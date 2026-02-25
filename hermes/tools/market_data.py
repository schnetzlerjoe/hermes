"""Market data tools using Yahoo Finance for quotes, historical prices, and screening.

Uses the public Yahoo Finance v8 chart API for price data.  There is no
official Yahoo Finance API, so these endpoints may change without notice.
A conservative rate limiter is applied to avoid throttling.
"""

from __future__ import annotations

import json
import logging
import time

from llama_index.core.tools import FunctionTool

from hermes.tools._base import YAHOO_BASE_URL, cached_request, yahoo_get

logger = logging.getLogger(__name__)

# Short TTL for market data that updates frequently but does not need to be
# fetched on every single call within the same agent turn.
_QUOTE_TTL: float = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def get_quote(symbol: str) -> dict:
    """Get current quote data for a stock or ETF.

    Returns the most recent price along with key fundamental and trading
    metrics: market cap, P/E ratio, 52-week range, volume, etc.

    Args:
        symbol: Ticker symbol (e.g. ``"AAPL"``, ``"SPY"``).

    Returns:
        Dict with ``symbol``, ``name``, ``price``, ``change``,
        ``change_percent``, ``volume``, ``market_cap``, ``pe_ratio``,
        ``fifty_two_week_high``, ``fifty_two_week_low``, ``currency``,
        and ``exchange``.
    """
    logger.debug("Fetching quote for %s", symbol)
    url = f"{YAHOO_BASE_URL}/v8/finance/chart/{symbol.upper()}"
    params = {"range": "1d", "interval": "1d", "includePrePost": "false"}

    data = await yahoo_get(url, params=params)

    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        raise ValueError(f"No quote data returned for symbol '{symbol}'.")

    result = results[0]
    meta = result.get("meta", {})

    return {
        "symbol": meta.get("symbol", symbol.upper()),
        "name": meta.get("longName", meta.get("shortName", "")),
        "price": meta.get("regularMarketPrice", 0.0),
        "previous_close": meta.get("chartPreviousClose", meta.get("previousClose", 0.0)),
        "change": round(
            meta.get("regularMarketPrice", 0.0)
            - meta.get("chartPreviousClose", meta.get("previousClose", 0.0)),
            4,
        ),
        "volume": meta.get("regularMarketVolume", 0),
        "currency": meta.get("currency", "USD"),
        "exchange": meta.get("exchangeName", ""),
        "instrument_type": meta.get("instrumentType", ""),
        "regular_market_time": meta.get("regularMarketTime", 0),
        "fifty_two_week_high": meta.get("fiftyTwoWeekHigh", None),
        "fifty_two_week_low": meta.get("fiftyTwoWeekLow", None),
    }


async def get_historical(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> list[dict]:
    """Get historical OHLCV (open/high/low/close/volume) price data.

    Args:
        symbol: Ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period.  Valid values: ``1d``, ``5d``, ``1mo``,
            ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``max``.
        interval: Bar interval.  Valid values: ``1m``, ``5m``, ``15m``,
            ``1h``, ``1d``, ``1wk``, ``1mo``.

    Returns:
        List of dicts, each with ``date`` (ISO string), ``open``, ``high``,
        ``low``, ``close``, ``volume``, and ``adj_close``.
    """
    logger.debug("Fetching history for %r, period=%s", symbol, period)
    url = f"{YAHOO_BASE_URL}/v8/finance/chart/{symbol.upper()}"
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
    }

    # Cache non-intraday data briefly; intraday data is always live.
    intraday_intervals = {"1m", "5m", "15m"}
    ttl = _QUOTE_TTL if interval not in intraday_intervals else None

    if ttl is not None:
        cache_key = f"hist_{symbol.upper()}_{period}_{interval}"

        async def _fetch() -> bytes:
            result = await yahoo_get(url, params=params)
            return json.dumps(result).encode()

        raw = await cached_request("yahoo_historical", cache_key, _fetch, ttl=ttl)
        data = json.loads(raw)
    else:
        data = await yahoo_get(url, params=params)

    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        raise ValueError(f"No historical data returned for symbol '{symbol}'.")

    result = results[0]
    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    adj_close_list = (
        indicators.get("adjclose", [{}])[0].get("adjclose", [])
        if indicators.get("adjclose")
        else []
    )

    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])

    bars = []
    for i, ts in enumerate(timestamps):
        # Skip bars where all price fields are None (non-trading days in some feeds).
        if i < len(closes) and closes[i] is None:
            continue

        bars.append({
            "date": time.strftime("%Y-%m-%d", time.gmtime(ts)),
            "open": opens[i] if i < len(opens) else None,
            "high": highs[i] if i < len(highs) else None,
            "low": lows[i] if i < len(lows) else None,
            "close": closes[i] if i < len(closes) else None,
            "adj_close": adj_close_list[i] if i < len(adj_close_list) else None,
            "volume": volumes[i] if i < len(volumes) else None,
        })

    return bars


async def get_multiple_quotes(symbols: list[str]) -> list[dict]:
    """Get quotes for multiple symbols in one batch.

    Convenience wrapper that calls :func:`get_quote` for each symbol and
    collects the results.  Errors for individual symbols are captured
    rather than aborting the entire batch.

    Args:
        symbols: List of ticker symbols (e.g. ``["AAPL", "MSFT", "GOOGL"]``).

    Returns:
        List of quote dicts.  Failed lookups include an ``error`` key
        instead of price data.
    """
    results = []
    for sym in symbols:
        try:
            quote = await get_quote(sym)
            results.append(quote)
        except Exception as exc:
            logger.warning("Failed to fetch quote for %s: %s", sym, exc)
            results.append({"symbol": sym.upper(), "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def create_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for all market data tools."""
    return [
        FunctionTool.from_defaults(
            async_fn=get_quote,
            name="get_stock_quote",
            description=(
                "Get the current quote for a stock or ETF including price, volume, "
                "and key metrics. Real-time data from Yahoo Finance."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_historical,
            name="get_historical_prices",
            description=(
                "Get historical OHLCV price data for a symbol. Supports periods "
                "from 1 day to max history, with intervals from 1 minute to monthly."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_multiple_quotes,
            name="get_multiple_quotes",
            description=(
                "Get current quotes for multiple symbols at once. Provide a list "
                "of ticker symbols. Failed lookups are returned with error messages "
                "rather than aborting the batch."
            ),
        ),
    ]
