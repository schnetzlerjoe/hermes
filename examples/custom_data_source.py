"""How to add a custom data source to Hermes.

Demonstrates wrapping an external API or local data as a Hermes-compatible
tool and registering it for agent use.

This example shows a hypothetical Bloomberg Terminal API wrapper.  The same
pattern works for any data source: wrap your API calls in plain async
functions, create FunctionTool instances from them, and register with Hermes.
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core.tools import FunctionTool

from hermes import Hermes, configure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Write async functions that call your data source.
#
# Each function should:
#   - Accept simple typed arguments (str, int, float, list, dict)
#   - Return JSON-serialisable data (dict or list)
#   - Have a clear docstring (LlamaIndex uses it as the tool description)
#   - Handle errors gracefully with informative messages
# ---------------------------------------------------------------------------


async def bloomberg_quote(ticker: str) -> dict[str, Any]:
    """Get a real-time quote from the Bloomberg Terminal.

    Returns last price, bid/ask, volume, and market cap for the
    given security identifier.

    Args:
        ticker: Bloomberg security identifier (e.g. "AAPL US Equity").

    Returns:
        Dict with ``last_price``, ``bid``, ``ask``, ``volume``,
        ``market_cap``, and ``currency``.
    """
    # In production, replace this stub with actual Bloomberg API calls.
    # For example, using blpapi:
    #
    #   import blpapi
    #   session = blpapi.Session()
    #   session.start()
    #   refDataService = session.getService("//blp/refdata")
    #   request = refDataService.createRequest("ReferenceDataRequest")
    #   request.append("securities", ticker)
    #   request.append("fields", "PX_LAST")
    #   ...
    logger.info("Bloomberg quote request for %s", ticker)
    return {
        "ticker": ticker,
        "last_price": 195.50,
        "bid": 195.48,
        "ask": 195.52,
        "volume": 42_000_000,
        "market_cap": 3_020_000_000_000,
        "currency": "USD",
    }


async def bloomberg_history(
    ticker: str,
    field: str = "PX_LAST",
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
) -> dict[str, Any]:
    """Get historical time series data from Bloomberg.

    Args:
        ticker: Bloomberg security identifier.
        field: Bloomberg field mnemonic (e.g. "PX_LAST", "VOLUME").
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        Dict with ``ticker``, ``field``, and ``data`` (list of
        ``{"date": str, "value": float}`` dicts).
    """
    logger.info("Bloomberg history: %s %s [%s to %s]", ticker, field, start_date, end_date)
    return {
        "ticker": ticker,
        "field": field,
        "data": [
            {"date": "2024-01-02", "value": 185.10},
            {"date": "2024-06-28", "value": 210.75},
            {"date": "2024-12-31", "value": 195.50},
        ],
    }


async def bloomberg_screening(
    universe: str = "SPX Index",
    min_market_cap: float | None = None,
    max_pe_ratio: float | None = None,
    sector: str | None = None,
) -> list[dict[str, Any]]:
    """Screen securities using Bloomberg criteria.

    Args:
        universe: Bloomberg index or universe identifier.
        min_market_cap: Minimum market cap in USD (e.g. 1e10 for $10B).
        max_pe_ratio: Maximum trailing P/E ratio.
        sector: GICS sector filter (e.g. "Information Technology").

    Returns:
        List of dicts with ``ticker``, ``name``, ``market_cap``,
        ``pe_ratio``, and ``sector``.
    """
    logger.info(
        "Bloomberg screen: universe=%s, min_mcap=%s, max_pe=%s, sector=%s",
        universe,
        min_market_cap,
        max_pe_ratio,
        sector,
    )
    return [
        {
            "ticker": "AAPL US Equity",
            "name": "Apple Inc",
            "market_cap": 3_020_000_000_000,
            "pe_ratio": 32.5,
            "sector": "Information Technology",
        },
        {
            "ticker": "MSFT US Equity",
            "name": "Microsoft Corp",
            "market_cap": 2_950_000_000_000,
            "pe_ratio": 35.2,
            "sector": "Information Technology",
        },
    ]


# ---------------------------------------------------------------------------
# Step 2: Wrap each function as a FunctionTool.
# ---------------------------------------------------------------------------


def create_bloomberg_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for the Bloomberg data source."""
    return [
        FunctionTool.from_defaults(
            async_fn=bloomberg_quote,
            name="bloomberg_quote",
            description=(
                "Get a real-time Bloomberg quote for a security. "
                "Returns last price, bid/ask, volume, and market cap."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=bloomberg_history,
            name="bloomberg_history",
            description=(
                "Get historical time series data from Bloomberg for a security "
                "and field over a date range."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=bloomberg_screening,
            name="bloomberg_screening",
            description=(
                "Screen a Bloomberg universe of securities by market cap, "
                "P/E ratio, sector, and other criteria."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Step 3: Register the tools with Hermes.
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate registering custom data source tools."""

    configure(
        llm_provider="anthropic",
        sec_user_agent="HermesDemo demo@example.com",
        verbose=True,
    )

    h = Hermes()

    # Register each Bloomberg tool individually.
    # Tags help the orchestrator find relevant tools by capability.
    for tool in create_bloomberg_tools():
        h.register_tool(
            name=tool.metadata.name,
            tool=tool,
            tags=["bloomberg", "market_data"],
            description=tool.metadata.description,
        )

    # Verify they appear in the tool list.
    print("Registered tools:")
    for name, tags in h.list_tools().items():
        print(f"  {name}: {tags}")


if __name__ == "__main__":
    main()
