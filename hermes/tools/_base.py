"""Shared HTTP client, caching helpers, and request utilities.

All network-facing tools use the helpers in this module so that rate limiting
and caching are applied consistently.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import httpx

from hermes.config import get_config
from hermes.infra.cache import FileCache
from hermes.infra.rate_limiter import get_limiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------------

SEC_BASE_URL = "https://data.sec.gov"
SEC_EFTS_URL = "https://efts.sec.gov/LATEST"
FRED_BASE_URL = "https://api.stlouisfed.org"
YAHOO_BASE_URL = "https://query1.finance.yahoo.com"

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None
_cache: FileCache | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return a shared :class:`httpx.AsyncClient` with sensible defaults.

    The client is created once and reused for connection pooling.  It sets a
    generous timeout for the large filing downloads that SEC EDGAR can produce.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"Accept": "application/json"},
        )
    return _client


def get_cache() -> FileCache:
    """Return a shared :class:`FileCache` rooted at the configured cache_dir."""
    global _cache  # noqa: PLW0603
    if _cache is None:
        cfg = get_config()
        _cache = FileCache(base_dir=cfg.cache_dir)
    return _cache


# ---------------------------------------------------------------------------
# API-specific GET helpers
# ---------------------------------------------------------------------------


async def sec_get(path: str, params: dict | None = None) -> dict:
    """Rate-limited GET to SEC EDGAR (``data.sec.gov``).

    The SEC requires a descriptive ``User-Agent`` header identifying the
    caller.  This is pulled from ``config.sec_user_agent``.

    Args:
        path: URL path appended to :data:`SEC_BASE_URL` (e.g. ``"/api/xbrl/..."``).
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.

    Raises:
        ValueError: If ``sec_user_agent`` is not configured.
        httpx.HTTPStatusError: On non-2xx responses.
    """
    cfg = get_config()
    if not cfg.sec_user_agent:
        raise ValueError(
            "sec_user_agent must be configured before calling SEC EDGAR APIs. "
            "Set HERMES_SEC_USER_AGENT or call configure(sec_user_agent=...)."
        )

    client = get_http_client()
    limiter = get_limiter("sec_edgar")

    async with limiter:
        url = f"{SEC_BASE_URL}{path}"
        logger.debug("SEC GET %s", url)
        response = await client.get(
            url,
            params=params,
            headers={"User-Agent": cfg.sec_user_agent},
        )
        response.raise_for_status()
        return response.json()


async def sec_efts_get(path: str, params: dict | None = None) -> dict:
    """Rate-limited GET to SEC EDGAR full-text search (``efts.sec.gov``).

    Uses the same rate limiter and User-Agent as :func:`sec_get` but targets
    the EFTS host instead.

    Args:
        path: URL path appended to :data:`SEC_EFTS_URL`.
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.
    """
    cfg = get_config()
    if not cfg.sec_user_agent:
        raise ValueError(
            "sec_user_agent must be configured before calling SEC EDGAR APIs."
        )

    client = get_http_client()
    limiter = get_limiter("sec_edgar")

    async with limiter:
        url = f"{SEC_EFTS_URL}{path}"
        logger.debug("SEC EFTS GET %s", url)
        response = await client.get(
            url,
            params=params,
            headers={"User-Agent": cfg.sec_user_agent},
        )
        response.raise_for_status()
        return response.json()


async def fred_get(path: str, params: dict | None = None) -> dict:
    """Rate-limited GET to FRED (``api.stlouisfed.org``).

    Automatically injects the ``api_key`` and ``file_type=json`` parameters.

    Args:
        path: URL path (e.g. ``"/fred/series/observations"``).
        params: Additional query parameters.

    Returns:
        Parsed JSON response body.

    Raises:
        ValueError: If ``fred_api_key`` is not configured.
        httpx.HTTPStatusError: On non-2xx responses.
    """
    cfg = get_config()
    if not cfg.fred_api_key:
        raise ValueError(
            "fred_api_key must be configured before calling FRED APIs. "
            "Set HERMES_FRED_API_KEY or call configure(fred_api_key=...)."
        )

    merged_params = {"api_key": cfg.fred_api_key, "file_type": "json"}
    if params:
        merged_params.update(params)

    client = get_http_client()
    limiter = get_limiter("fred")

    async with limiter:
        url = f"{FRED_BASE_URL}{path}"
        logger.debug("FRED GET %s", url)
        response = await client.get(url, params=merged_params)
        response.raise_for_status()
        return response.json()


async def yahoo_get(url: str, params: dict | None = None) -> dict:
    """Rate-limited GET to Yahoo Finance.

    Yahoo Finance has no official API and may throttle aggressively, so a
    conservative rate limiter is applied.

    Args:
        url: Full URL (the Yahoo endpoints vary across versions).
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.
    """
    client = get_http_client()
    limiter = get_limiter("yahoo_finance")

    async with limiter:
        logger.debug("Yahoo GET %s", url)
        response = await client.get(
            url,
            params=params,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Cached request helper
# ---------------------------------------------------------------------------


async def cached_request(
    namespace: str,
    key: str,
    fetch_fn: Callable[[], Awaitable[bytes]],
    ttl: float | None = None,
) -> bytes:
    """Return cached data or call *fetch_fn* to populate the cache.

    This is the standard pattern for all cacheable tool calls: try the disk
    cache first, fall back to the network, then store the result for next time.

    Args:
        namespace: Cache namespace (e.g. ``"sec_facts"``).
        key: Cache key (e.g. a URL or composite identifier).
        fetch_fn: Async callable that returns raw bytes on cache miss.
        ttl: Time-to-live in seconds.  ``None`` means permanent.

    Returns:
        The cached or freshly fetched bytes.
    """
    cache = get_cache()

    cached = cache.get(namespace, key)
    if cached is not None:
        logger.debug("Cache hit: %s/%s", namespace, key)
        return cached

    logger.debug("Cache miss: %s/%s -- fetching", namespace, key)
    data = await fetch_fn()
    cache.put(namespace, key, data, ttl_seconds=ttl)
    return data
