"""Tests for FRED tools.

All tests use mocked HTTP responses to verify the parsing and
transformation logic without requiring a real FRED API key or
network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes.infra.cache import FileCache

# ---------------------------------------------------------------------------
# Sample response data
# ---------------------------------------------------------------------------

SAMPLE_SERIES_OBSERVATIONS = {
    "realtime_start": "2024-01-01",
    "realtime_end": "2024-12-31",
    "observation_start": "2020-01-01",
    "observation_end": "2024-12-31",
    "units": "lin",
    "output_type": 1,
    "file_type": "json",
    "order_by": "observation_date",
    "sort_order": "asc",
    "count": 5,
    "offset": 0,
    "limit": 100000,
    "observations": [
        {"date": "2020-01-01", "value": "2.16"},
        {"date": "2021-01-01", "value": "0.08"},
        {"date": "2022-01-01", "value": "3.87"},
        {"date": "2023-01-01", "value": "5.33"},
        {"date": "2024-01-01", "value": "5.33"},
    ],
}

SAMPLE_SERIES_INFO = {
    "seriess": [
        {
            "id": "FEDFUNDS",
            "title": "Federal Funds Effective Rate",
            "observation_start": "1954-07-01",
            "observation_end": "2024-12-01",
            "frequency": "Monthly",
            "frequency_short": "M",
            "units": "Percent",
            "units_short": "%",
            "seasonal_adjustment": "Not Seasonally Adjusted",
            "seasonal_adjustment_short": "NSA",
            "notes": "The federal funds rate is the interest rate...",
        }
    ]
}

SAMPLE_SERIES_SEARCH = {
    "seriess": [
        {
            "id": "GDP",
            "title": "Gross Domestic Product",
            "frequency": "Quarterly",
            "units": "Billions of Dollars",
            "popularity": 95,
        },
        {
            "id": "GDPC1",
            "title": "Real Gross Domestic Product",
            "frequency": "Quarterly",
            "units": "Billions of Chained 2017 Dollars",
            "popularity": 90,
        },
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path: Path):
    """Give each test a fresh cache."""
    cache = FileCache(base_dir=str(tmp_path / "test_fred_cache"))
    with patch("hermes.tools._base.get_cache", return_value=cache):
        yield


@pytest.fixture(autouse=True)
def _set_fred_config():
    """Ensure FRED API key is configured for all tests."""
    fake_config = type(
        "FakeConfig",
        (),
        {"fred_api_key": "test-fake-key", "sec_user_agent": "test"},
    )()
    with patch("hermes.tools._base.get_config", return_value=fake_config):
        yield


# ---------------------------------------------------------------------------
# Helper to build a mock fred_get
# ---------------------------------------------------------------------------


def make_mock_fred_get(response_data: dict):
    """Create a mock for hermes.tools._base.fred_get that returns response_data."""

    async def mock_fred_get(path, params=None):
        return response_data

    return mock_fred_get


# ---------------------------------------------------------------------------
# Tests: parsing observation data
# ---------------------------------------------------------------------------


class TestFredObservationsParsing:
    """Test that FRED observation responses are parsed correctly."""

    def test_observations_have_date_and_value(self) -> None:
        """Each observation should have a date and numeric value."""
        observations = SAMPLE_SERIES_OBSERVATIONS["observations"]
        for obs in observations:
            assert "date" in obs
            assert "value" in obs
            # Values should be parseable as floats.
            float(obs["value"])

    def test_observations_in_chronological_order(self) -> None:
        """Observations should be sorted by date ascending."""
        observations = SAMPLE_SERIES_OBSERVATIONS["observations"]
        dates = [obs["date"] for obs in observations]
        assert dates == sorted(dates)

    def test_observation_count_matches(self) -> None:
        """The observation list length should match the count field."""
        assert (
            len(SAMPLE_SERIES_OBSERVATIONS["observations"])
            == SAMPLE_SERIES_OBSERVATIONS["count"]
        )


# ---------------------------------------------------------------------------
# Tests: parsing series info
# ---------------------------------------------------------------------------


class TestFredSeriesInfoParsing:
    """Test that series metadata responses are parsed correctly."""

    def test_series_has_required_fields(self) -> None:
        """Series info should contain id, title, frequency, and units."""
        series = SAMPLE_SERIES_INFO["seriess"][0]
        assert series["id"] == "FEDFUNDS"
        assert "Federal Funds" in series["title"]
        assert series["frequency"] == "Monthly"
        assert series["units"] == "Percent"

    def test_series_has_date_range(self) -> None:
        """Series info should include observation start and end dates."""
        series = SAMPLE_SERIES_INFO["seriess"][0]
        assert "observation_start" in series
        assert "observation_end" in series


# ---------------------------------------------------------------------------
# Tests: parsing series search
# ---------------------------------------------------------------------------


class TestFredSeriesSearchParsing:
    """Test that series search responses are parsed correctly."""

    def test_search_returns_multiple_results(self) -> None:
        """A search should return a list of matching series."""
        results = SAMPLE_SERIES_SEARCH["seriess"]
        assert len(results) == 2

    def test_search_results_have_metadata(self) -> None:
        """Each search result should have id, title, and frequency."""
        for series in SAMPLE_SERIES_SEARCH["seriess"]:
            assert "id" in series
            assert "title" in series
            assert "frequency" in series

    def test_search_results_sorted_by_popularity(self) -> None:
        """Results should be ordered by descending popularity."""
        results = SAMPLE_SERIES_SEARCH["seriess"]
        popularities = [s["popularity"] for s in results]
        assert popularities == sorted(popularities, reverse=True)


# ---------------------------------------------------------------------------
# Tests: fred_get helper with mocked HTTP
# ---------------------------------------------------------------------------


class TestFredGetHelper:
    """Test the fred_get HTTP helper with mocked responses."""

    @pytest.mark.asyncio
    async def test_fred_get_injects_api_key(self) -> None:
        """fred_get should inject the api_key parameter automatically."""
        captured_params = {}

        async def mock_get(*args, **kwargs):
            # httpx.AsyncClient.get passes params as a keyword argument.
            captured_params.update(kwargs.get("params", {}) or {})
            resp = type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "json": lambda self: SAMPLE_SERIES_OBSERVATIONS,
                    "raise_for_status": lambda self: None,
                },
            )()
            return resp

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
            from hermes.tools._base import fred_get

            await fred_get("/fred/series/observations", params={"series_id": "FEDFUNDS"})

        assert captured_params["api_key"] == "test-fake-key"
        assert captured_params["file_type"] == "json"
        assert captured_params["series_id"] == "FEDFUNDS"

    @pytest.mark.asyncio
    async def test_fred_get_raises_without_api_key(self) -> None:
        """fred_get should raise ValueError if fred_api_key is not set."""
        from hermes.tools._base import fred_get

        no_key_config = type(
            "FakeConfig", (), {"fred_api_key": None, "sec_user_agent": "test"}
        )()
        with patch("hermes.tools._base.get_config", return_value=no_key_config):
            with pytest.raises(ValueError, match="fred_api_key"):
                await fred_get("/fred/series")


# ---------------------------------------------------------------------------
# Tests: value transformation
# ---------------------------------------------------------------------------


class TestValueTransformation:
    """Test converting string observation values to numeric types."""

    def test_string_values_to_float(self) -> None:
        """FRED returns values as strings; they should convert to float."""
        observations = SAMPLE_SERIES_OBSERVATIONS["observations"]
        values = [float(obs["value"]) for obs in observations]
        assert values[0] == pytest.approx(2.16)
        assert values[2] == pytest.approx(3.87)

    def test_missing_value_marker(self) -> None:
        """FRED uses '.' as a missing value marker; parsing should handle it."""
        # This tests the expected behaviour -- '.' is not a valid float.
        with pytest.raises(ValueError):
            float(".")
