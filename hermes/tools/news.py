"""News retrieval tools for financial news and web research.

For v0.1 this uses free, publicly accessible sources: Yahoo Finance RSS
feeds and the Google News RSS feed.  No API keys are required.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from html import unescape

import httpx
from llama_index.core.tools import FunctionTool

from hermes.tools._base import get_http_client

logger = logging.getLogger(__name__)

# Yahoo Finance RSS feed for company-specific news.
_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

# Google News RSS for general queries.
_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_rss_items(xml_bytes: bytes, limit: int) -> list[dict]:
    """Parse RSS XML into a list of article dicts.

    Handles both standard RSS 2.0 and Atom-style feeds by looking for
    ``<item>`` or ``<entry>`` elements.

    Args:
        xml_bytes: Raw RSS/Atom XML content.
        limit: Maximum number of items to return.

    Returns:
        List of dicts with ``title``, ``link``, ``published``, and ``source``.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        logger.warning("Failed to parse RSS XML")
        return []

    # RSS 2.0 uses <item>, Atom uses <entry>.
    items = root.findall(".//item")
    if not items:
        # Try Atom namespace.
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//atom:entry", ns)

    results = []
    for item in items[:limit]:
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate") or item.find("published")
        source_el = item.find("source")

        title = unescape(title_el.text or "") if title_el is not None else ""
        # Atom <link> stores URL in href attribute; RSS in text.
        if link_el is not None:
            link = link_el.get("href", link_el.text or "")
        else:
            link = ""
        published = pub_el.text or "" if pub_el is not None else ""
        source = source_el.text or "" if source_el is not None else ""

        results.append({
            "title": title.strip(),
            "link": link.strip(),
            "published": published.strip(),
            "source": source.strip(),
        })

    return results


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def search_company_news(ticker: str, limit: int = 10) -> list[dict]:
    """Search for recent news articles about a company.

    Uses the Yahoo Finance RSS feed which provides headline-level news
    for publicly traded companies at no cost.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        limit: Maximum number of articles to return (default 10).

    Returns:
        List of dicts with ``title``, ``link``, ``published``, and ``source``.
    """
    limit = max(1, min(limit, 50))
    client = get_http_client()

    params = {"s": ticker.upper(), "region": "US", "lang": "en-US"}

    try:
        response = await client.get(
            _YAHOO_RSS_URL,
            params=params,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("Yahoo Finance RSS returned %s for %s", exc.response.status_code, ticker)
        return []
    except httpx.RequestError as exc:
        logger.warning("Network error fetching news for %s: %s", ticker, exc)
        return []

    return _parse_rss_items(response.content, limit)


async def search_financial_news(query: str, limit: int = 10) -> list[dict]:
    """General financial news search.

    Uses Google News RSS to find recent news articles matching a free-text
    query.  Useful for topic-level research (e.g. ``"interest rate hike"``
    or ``"semiconductor shortage"``).

    Args:
        query: Search terms (e.g. ``"Federal Reserve rate decision"``).
        limit: Maximum number of articles to return (default 10).

    Returns:
        List of dicts with ``title``, ``link``, ``published``, and ``source``.
    """
    limit = max(1, min(limit, 50))
    client = get_http_client()

    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    try:
        response = await client.get(
            _GOOGLE_NEWS_RSS_URL,
            params=params,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Google News RSS returned %s for query '%s'",
            exc.response.status_code, query,
        )
        return []
    except httpx.RequestError as exc:
        logger.warning("Network error fetching news for '%s': %s", query, exc)
        return []

    return _parse_rss_items(response.content, limit)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def create_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for all news tools."""
    return [
        FunctionTool.from_defaults(
            async_fn=search_company_news,
            name="search_company_news",
            description=(
                "Search for recent news articles about a specific company by ticker. "
                "Returns headlines with links and publication dates from Yahoo Finance."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=search_financial_news,
            name="search_financial_news",
            description=(
                "Search for financial news by keyword query. Use for topic-level "
                "research like 'interest rate hike' or 'semiconductor shortage'. "
                "Returns headlines from Google News."
            ),
        ),
    ]
