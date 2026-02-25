"""FRED (Federal Reserve Economic Data) tools for macroeconomic data retrieval.

Provides tools for fetching economic time series, searching for data series,
and retrieving series metadata from the Federal Reserve Bank of St. Louis
FRED API.
"""

from __future__ import annotations

import json
import logging

from llama_index.core.tools import FunctionTool

from hermes.infra.cache import TTL_1_HOUR
from hermes.tools._base import cached_request, fred_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def get_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Fetch observations for a FRED data series.

    Returns the time-series data points for a given FRED series identifier.
    Common series include ``GDP``, ``CPIAUCSL`` (CPI), ``DFF`` (fed funds
    rate), ``UNRATE`` (unemployment), and ``DGS10`` (10-year Treasury).

    Args:
        series_id: FRED series identifier (e.g. ``"GDP"``, ``"CPIAUCSL"``).
        start_date: Optional start date in ``YYYY-MM-DD`` format.
        end_date: Optional end date in ``YYYY-MM-DD`` format.

    Returns:
        List of dicts with ``date`` and ``value`` keys.  Values are strings
        because FRED returns ``"."`` for missing observations.
    """
    logger.debug("Fetching FRED series %r", series_id)
    params: dict[str, str] = {"series_id": series_id.upper()}
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    cache_key = json.dumps(params, sort_keys=True)

    async def _fetch() -> bytes:
        data = await fred_get("/fred/series/observations", params=params)
        return json.dumps(data).encode()

    raw = await cached_request("fred_series", cache_key, _fetch, ttl=TTL_1_HOUR)
    data = json.loads(raw)

    observations = data.get("observations", [])
    return [
        {"date": obs["date"], "value": obs["value"]}
        for obs in observations
    ]


async def search_series(query: str, limit: int = 10) -> list[dict]:
    """Search for FRED data series by keyword.

    Useful for discovering which series are available when you know the
    economic concept but not the exact series ID.

    Args:
        query: Search terms (e.g. ``"consumer price index"``).
        limit: Maximum number of results to return (1-100).

    Returns:
        List of dicts with ``id``, ``title``, ``frequency``, ``units``,
        ``seasonal_adjustment``, and ``last_updated``.
    """
    logger.debug("Searching FRED series: %r", query)
    limit = max(1, min(limit, 100))

    params: dict[str, str | int] = {
        "search_text": query,
        "limit": limit,
    }

    # No caching for search -- results should reflect current catalog.
    data = await fred_get("/fred/series/search", params=params)

    series_list = data.get("seriess", [])
    return [
        {
            "id": s.get("id", ""),
            "title": s.get("title", ""),
            "frequency": s.get("frequency", ""),
            "units": s.get("units", ""),
            "seasonal_adjustment": s.get("seasonal_adjustment", ""),
            "last_updated": s.get("last_updated", ""),
            "popularity": s.get("popularity", 0),
        }
        for s in series_list
    ]


async def get_series_info(series_id: str) -> dict:
    """Get metadata about a FRED series.

    Returns descriptive information including the full title, units of
    measurement, reporting frequency, seasonal adjustment status, and
    the date of the most recent observation.

    Args:
        series_id: FRED series identifier (e.g. ``"DGS10"``).

    Returns:
        Dict with ``id``, ``title``, ``units``, ``frequency``,
        ``seasonal_adjustment``, ``last_updated``, ``observation_start``,
        ``observation_end``, and ``notes``.
    """
    logger.debug("Fetching FRED series info: %r", series_id)
    params = {"series_id": series_id.upper()}

    cache_key = f"info_{series_id.upper()}"

    async def _fetch() -> bytes:
        data = await fred_get("/fred/series", params=params)
        return json.dumps(data).encode()

    raw = await cached_request("fred_info", cache_key, _fetch, ttl=TTL_1_HOUR)
    data = json.loads(raw)

    series_list = data.get("seriess", [])
    if not series_list:
        raise ValueError(f"FRED series '{series_id}' not found.")

    s = series_list[0]
    return {
        "id": s.get("id", ""),
        "title": s.get("title", ""),
        "units": s.get("units", ""),
        "frequency": s.get("frequency", ""),
        "seasonal_adjustment": s.get("seasonal_adjustment", ""),
        "last_updated": s.get("last_updated", ""),
        "observation_start": s.get("observation_start", ""),
        "observation_end": s.get("observation_end", ""),
        "notes": s.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def create_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for all FRED tools."""
    return [
        FunctionTool.from_defaults(
            async_fn=get_series,
            name="get_fred_series",
            description=(
                "Fetch time-series observations from FRED. Provide a series ID "
                "(e.g. 'GDP', 'CPIAUCSL', 'DFF', 'UNRATE', 'DGS10') and optional "
                "date range. Returns date/value pairs."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=search_series,
            name="search_fred_series",
            description=(
                "Search for FRED data series by keyword. Use this to discover "
                "available economic data series when you know the concept but "
                "not the exact series ID."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_series_info,
            name="get_fred_series_info",
            description=(
                "Get metadata for a FRED series including title, units, frequency, "
                "and the date range of available observations."
            ),
        ),
    ]
