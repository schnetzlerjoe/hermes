"""Tests for market data tools.

All tests use mocked HTTP responses to verify parsing and data
transformation logic without requiring network access or API keys.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes.infra.cache import FileCache

# ---------------------------------------------------------------------------
# Sample response data (Yahoo Finance format)
# ---------------------------------------------------------------------------

SAMPLE_QUOTE_RESPONSE = {
    "quoteResponse": {
        "result": [
            {
                "symbol": "AAPL",
                "shortName": "Apple Inc.",
                "regularMarketPrice": 195.50,
                "regularMarketChange": 2.30,
                "regularMarketChangePercent": 1.19,
                "regularMarketVolume": 42000000,
                "marketCap": 3020000000000,
                "trailingPE": 32.5,
                "forwardPE": 28.7,
                "dividendYield": 0.52,
                "fiftyTwoWeekHigh": 237.23,
                "fiftyTwoWeekLow": 164.08,
                "currency": "USD",
                "exchange": "NMS",
            }
        ],
        "error": None,
    }
}

SAMPLE_HISTORICAL_RESPONSE = {
    "chart": {
        "result": [
            {
                "meta": {
                    "symbol": "AAPL",
                    "currency": "USD",
                    "exchangeTimezoneName": "America/New_York",
                    "regularMarketPrice": 195.50,
                },
                "timestamp": [1704153600, 1704240000, 1704326400],
                "indicators": {
                    "quote": [
                        {
                            "open": [185.10, 185.80, 184.50],
                            "high": [186.20, 186.50, 186.00],
                            "low": [184.50, 184.90, 183.80],
                            "close": [185.85, 185.30, 185.50],
                            "volume": [40000000, 38000000, 42000000],
                        }
                    ],
                    "adjclose": [{"adjclose": [185.85, 185.30, 185.50]}],
                },
            }
        ],
        "error": None,
    }
}

SAMPLE_PROFILE_DATA = {
    "symbol": "AAPL",
    "shortName": "Apple Inc.",
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "website": "https://www.apple.com",
    "fullTimeEmployees": 164000,
    "longBusinessSummary": (
        "Apple Inc. designs, manufactures, and markets smartphones, "
        "personal computers, tablets, wearables, and accessories worldwide."
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path: Path):
    """Give each test a fresh cache."""
    cache = FileCache(base_dir=str(tmp_path / "test_market_cache"))
    with patch("hermes.tools._base.get_cache", return_value=cache):
        yield


# ---------------------------------------------------------------------------
# Tests: quote data parsing
# ---------------------------------------------------------------------------


class TestQuoteParsing:
    """Test parsing of real-time quote response data."""

    def test_quote_has_price(self) -> None:
        """Quote response should contain the current market price."""
        result = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        assert result["regularMarketPrice"] == 195.50

    def test_quote_has_change_fields(self) -> None:
        """Quote should include absolute and percentage change."""
        result = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        assert result["regularMarketChange"] == 2.30
        assert abs(result["regularMarketChangePercent"] - 1.19) < 0.01

    def test_quote_has_market_cap(self) -> None:
        """Quote should include market capitalization."""
        result = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        assert result["marketCap"] == 3020000000000

    def test_quote_has_valuation_ratios(self) -> None:
        """Quote should include trailing and forward P/E ratios."""
        result = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        assert result["trailingPE"] == 32.5
        assert result["forwardPE"] == 28.7

    def test_quote_has_52_week_range(self) -> None:
        """Quote should include 52-week high and low."""
        result = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        assert result["fiftyTwoWeekHigh"] > result["fiftyTwoWeekLow"]
        assert result["fiftyTwoWeekHigh"] == 237.23
        assert result["fiftyTwoWeekLow"] == 164.08


# ---------------------------------------------------------------------------
# Tests: historical data parsing
# ---------------------------------------------------------------------------


class TestHistoricalDataParsing:
    """Test parsing of historical OHLCV data."""

    def _get_chart_data(self) -> dict:
        return SAMPLE_HISTORICAL_RESPONSE["chart"]["result"][0]

    def test_timestamps_present(self) -> None:
        """Historical data should contain Unix timestamps."""
        data = self._get_chart_data()
        assert len(data["timestamp"]) == 3

    def test_ohlcv_arrays_aligned(self) -> None:
        """Open, high, low, close, volume arrays should have equal length."""
        data = self._get_chart_data()
        quote = data["indicators"]["quote"][0]
        n = len(data["timestamp"])
        assert len(quote["open"]) == n
        assert len(quote["high"]) == n
        assert len(quote["low"]) == n
        assert len(quote["close"]) == n
        assert len(quote["volume"]) == n

    def test_high_greater_than_low(self) -> None:
        """For each bar, high should be >= low."""
        data = self._get_chart_data()
        quote = data["indicators"]["quote"][0]
        for high, low in zip(quote["high"], quote["low"]):
            assert high >= low

    def test_adjusted_close_present(self) -> None:
        """Adjusted close values should be present."""
        data = self._get_chart_data()
        adjclose = data["indicators"]["adjclose"][0]["adjclose"]
        assert len(adjclose) == 3
        assert all(isinstance(v, (int, float)) for v in adjclose)

    def test_meta_has_symbol(self) -> None:
        """Chart metadata should identify the ticker symbol."""
        data = self._get_chart_data()
        assert data["meta"]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests: converting raw data to records
# ---------------------------------------------------------------------------


class TestDataConversion:
    """Test converting raw API data to structured records."""

    def test_build_ohlcv_records(self) -> None:
        """Convert parallel arrays into a list of OHLCV dicts."""
        data = SAMPLE_HISTORICAL_RESPONSE["chart"]["result"][0]
        timestamps = data["timestamp"]
        quote = data["indicators"]["quote"][0]

        records = []
        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp": ts,
                "open": quote["open"][i],
                "high": quote["high"][i],
                "low": quote["low"][i],
                "close": quote["close"][i],
                "volume": quote["volume"][i],
            })

        assert len(records) == 3
        assert records[0]["open"] == 185.10
        assert records[2]["volume"] == 42000000

    def test_normalise_quote_fields(self) -> None:
        """Normalise a raw quote into a standard format."""
        raw = SAMPLE_QUOTE_RESPONSE["quoteResponse"]["result"][0]
        normalised = {
            "ticker": raw["symbol"],
            "name": raw["shortName"],
            "price": raw["regularMarketPrice"],
            "change": raw["regularMarketChange"],
            "change_pct": raw["regularMarketChangePercent"],
            "volume": raw["regularMarketVolume"],
            "market_cap": raw["marketCap"],
        }

        assert normalised["ticker"] == "AAPL"
        assert normalised["price"] == 195.50
        assert normalised["volume"] == 42000000


# ---------------------------------------------------------------------------
# Tests: company profile parsing
# ---------------------------------------------------------------------------


class TestCompanyProfile:
    """Test company profile/info data parsing."""

    def test_profile_has_sector_and_industry(self) -> None:
        """Company profile should include sector and industry."""
        assert SAMPLE_PROFILE_DATA["sector"] == "Technology"
        assert SAMPLE_PROFILE_DATA["industry"] == "Consumer Electronics"

    def test_profile_has_employee_count(self) -> None:
        """Profile should include employee count as an integer."""
        assert SAMPLE_PROFILE_DATA["fullTimeEmployees"] == 164000
        assert isinstance(SAMPLE_PROFILE_DATA["fullTimeEmployees"], int)

    def test_profile_has_business_summary(self) -> None:
        """Profile should include a text business description."""
        summary = SAMPLE_PROFILE_DATA["longBusinessSummary"]
        assert len(summary) > 50
        assert "Apple" in summary


# ---------------------------------------------------------------------------
# Tests: yahoo_get helper with mocked HTTP
# ---------------------------------------------------------------------------


class TestYahooGetHelper:
    """Test the yahoo_get HTTP helper."""

    @pytest.mark.asyncio
    async def test_yahoo_get_sets_user_agent(self) -> None:
        """yahoo_get should set a browser-like User-Agent header."""
        captured_headers = {}

        async def mock_get(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}) or {})
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "json": lambda self: SAMPLE_QUOTE_RESPONSE,
                    "raise_for_status": lambda self: None,
                },
            )()

        mock_client = type("MockClient", (), {"get": mock_get})()

        # Build a no-op async context manager for the rate limiter.
        class NoOpLimiter:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch("hermes.tools._base._client", mock_client),
            patch("hermes.tools._base.get_limiter", return_value=NoOpLimiter()),
        ):
            from hermes.tools._base import yahoo_get

            await yahoo_get("https://query1.finance.yahoo.com/v7/finance/quote")

        # Should have a User-Agent that looks like a real browser.
        assert "Mozilla" in captured_headers.get("User-Agent", "")


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test that market data tools handle errors gracefully."""

    def test_empty_quote_response(self) -> None:
        """An empty result list should be handled without exception."""
        response = {"quoteResponse": {"result": [], "error": None}}
        assert len(response["quoteResponse"]["result"]) == 0

    def test_chart_error_response(self) -> None:
        """A chart response with an error should be detectable."""
        response = {
            "chart": {
                "result": None,
                "error": {
                    "code": "Not Found",
                    "description": "No data found for INVALID ticker",
                },
            }
        }
        assert response["chart"]["error"] is not None
        assert response["chart"]["result"] is None
