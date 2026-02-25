"""SEC EDGAR tools for retrieving filings, financial data, and company information.

Uses the ``edgartools`` library for structured XBRL financial data, filing
retrieval, and insider/institutional holdings.  Full-text search still uses
the SEC EFTS API directly since edgartools does not cover it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from llama_index.core.tools import FunctionTool

from hermes.config import get_config
from hermes.infra.cache import TTL_1_HOUR
from hermes.tools._base import cached_request, sec_efts_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# edgartools identity bridge
# ---------------------------------------------------------------------------

_identity_set = False


def _ensure_identity() -> None:
    """Set edgartools identity from hermes config (once)."""
    global _identity_set  # noqa: PLW0603
    if _identity_set:
        return
    from edgar import set_identity

    cfg = get_config()
    if cfg.sec_user_agent:
        set_identity(cfg.sec_user_agent)
    else:
        raise ValueError(
            "sec_user_agent must be configured before calling SEC EDGAR APIs."
        )
    _identity_set = True


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def get_company_facts(ticker: str) -> dict:
    """Fetch structured XBRL financial data for a company.

    Returns standardized financial statements (income statement, balance sheet,
    cash flow) with historical values parsed from XBRL filings.  Data is
    sourced from the company's most recent annual and quarterly filings.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).

    Returns:
        Dict with ``entityName`` and ``financials`` containing income statement,
        balance sheet, and cash flow statement data as text tables.
    """
    _ensure_identity()
    logger.debug("Fetching financials for %r via edgartools", ticker)

    def _fetch() -> dict:
        from edgar import Company

        company = Company(ticker)
        financials = company.get_financials()

        result: dict[str, Any] = {
            "entityName": company.name,
            "cik": company.cik,
            "tickers": company.tickers,
            "financials": {},
        }

        for stmt_name in ("income_statement", "balance_sheet", "cashflow_statement"):
            stmt_fn = getattr(financials, stmt_name, None)
            if stmt_fn is None:
                continue
            stmt = stmt_fn()
            if stmt is not None:
                result["financials"][stmt_name] = str(stmt)

        return result

    return await asyncio.to_thread(_fetch)


async def search_filings(
    query: str,
    ticker: str | None = None,
    filing_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Full-text search across SEC filings.

    Uses the EDGAR Full-Text Search System (EFTS) to find filings matching
    a keyword query.  Results include filing metadata and text snippets
    showing where the match occurred.

    Args:
        query: Free-text search query (e.g. ``"revenue recognition"``).
        ticker: Optional ticker to restrict results to a single filer.
        filing_type: Optional filing type filter (e.g. ``"10-K"``, ``"8-K"``).
        date_from: Start date in ``YYYY-MM-DD`` format.
        date_to: End date in ``YYYY-MM-DD`` format.

    Returns:
        List of dicts, each containing ``file_date``, ``entity_name``,
        ``file_type``, ``file_url``, and ``snippet``.
    """
    logger.debug("Searching filings: query=%r", query)
    params: dict[str, str] = {"q": query}
    if ticker:
        params["dateRange"] = "custom"
        params["q"] = f'"{ticker}" {query}'
    if filing_type:
        params["forms"] = filing_type
    if date_from:
        params["startdt"] = date_from
    if date_to:
        params["enddt"] = date_to

    cache_key = json.dumps(params, sort_keys=True)

    async def _fetch() -> bytes:
        data = await sec_efts_get("/search-index", params=params)
        return json.dumps(data).encode()

    raw = await cached_request("sec_search", cache_key, _fetch, ttl=TTL_1_HOUR)
    data = json.loads(raw)

    hits = data.get("hits", {}).get("hits", [])
    results = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append({
            "entity_name": source.get("entity_name", ""),
            "file_date": source.get("file_date", ""),
            "file_type": source.get("file_type", ""),
            "file_url": source.get("file_url", ""),
            "period_of_report": source.get("period_of_report", ""),
            "snippet": " ".join(
                hit.get("highlight", {}).get("_content", [""])
            ),
        })

    return results


async def get_submissions(ticker: str, limit: int = 200) -> dict:
    """Get filing history and metadata for a company.

    Returns the company's CIK, name, SIC code, and a list of recent
    filings with dates, types, accession numbers, and document URLs.

    Args:
        ticker: Stock ticker symbol (e.g. ``"MSFT"``).
        limit: Maximum number of recent filings to return (default 200).

    Returns:
        Dict with company info and ``recentFilings`` list.
    """
    _ensure_identity()
    logger.debug("Fetching submissions for %r via edgartools", ticker)

    def _fetch() -> dict:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings()

        result: dict[str, Any] = {
            "cik": company.cik,
            "name": company.name,
            "sic": company.sic,
            "tickers": company.tickers,
            "fiscalYearEnd": company.fiscal_year_end,
        }

        filings_list: list[dict] = []
        for filing in filings.head(limit):
            filings_list.append({
                "form": filing.form,
                "filingDate": filing.filing_date,
                "accessionNumber": filing.accession_no,
                "url": filing.filing_url,
            })

        result["recentFilings"] = filings_list
        return result

    return await asyncio.to_thread(_fetch)


async def get_filing_urls(
    ticker: str,
    filing_types: str = "10-K,10-Q",
    limit: int = 17,
) -> list[dict]:
    """Discover available filing URLs for a company filtered by type.

    Queries SEC EDGAR and returns a filtered list of filings matching the
    requested types (e.g. 10-K and 10-Q).

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        filing_types: Comma-separated filing types to include
            (default ``"10-K,10-Q"``).
        limit: Maximum number of filings to return (default 17).

    Returns:
        List of dicts with ``form``, ``filingDate``, ``accessionNumber``,
        and ``url`` keys, sorted by filing date descending.
    """
    _ensure_identity()
    logger.debug("Discovering %s filings for %r (limit=%d)", filing_types, ticker, limit)

    def _fetch() -> list[dict]:
        from edgar import Company

        company = Company(ticker)
        types_list = [t.strip() for t in filing_types.split(",")]
        filings = company.get_filings().filter(form=types_list).head(limit)

        matched: list[dict] = []
        for filing in filings:
            matched.append({
                "form": filing.form,
                "filingDate": filing.filing_date,
                "accessionNumber": filing.accession_no,
                "url": filing.filing_url,
            })

        matched.sort(key=lambda f: f.get("filingDate", ""), reverse=True)
        return matched

    return await asyncio.to_thread(_fetch)


async def get_filing_financial_tables(
    ticker: str,
    accession_number: str,
) -> str:
    """Extract structured financial statements from a SEC filing via XBRL.

    Retrieves the filing by accession number and extracts the income statement,
    balance sheet, and cash flow statement parsed from XBRL data.  Returns
    pre-classified, standardized financial tables — not raw HTML scraping.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        accession_number: SEC accession number (e.g.
            ``"0000320193-24-000123"``).

    Returns:
        Formatted text containing all available financial statements,
        separated by headers and dividers.
    """
    _ensure_identity()
    logger.debug(
        "Extracting financial tables for %r accession %s", ticker, accession_number
    )

    def _fetch() -> str:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings().filter(accession_number=accession_number)
        filing = filings.get_filing_at(0) if len(filings) > 0 else None

        if filing is None:
            return f"No filing found for accession number {accession_number}"

        financials = filing.obj()

        sections: list[str] = []
        for stmt_name, label in [
            ("income_statement", "INCOME STATEMENT"),
            ("balance_sheet", "BALANCE SHEET"),
            ("cashflow_statement", "CASH FLOW STATEMENT"),
        ]:
            stmt_fn = getattr(financials, stmt_name, None)
            if stmt_fn is None:
                continue
            try:
                stmt = stmt_fn()
            except Exception:
                stmt = None
            if stmt is not None:
                sections.append(f"=== {label} ===\n\n{stmt}")

        if not sections:
            return f"No financial statements found in filing {accession_number}"

        return "\n\n---\n\n".join(sections)

    return await asyncio.to_thread(_fetch)


async def get_filing_text(url: str, max_chars: int = 150_000) -> str:
    """Extract structured text sections from a SEC filing.

    Downloads the filing and extracts the qualitative content (MD&A, Risk
    Factors, Business description, etc.) as clean readable text.  For 10-K
    and 10-Q filings, returns individually labeled sections.

    Args:
        url: Full URL to the filing document on SEC EDGAR.
        max_chars: Maximum characters to return (default 150 000).

    Returns:
        Clean text content of the filing with section headers.
    """
    _ensure_identity()
    logger.debug("Extracting filing text from %s", url)

    def _fetch() -> str:
        from edgar import get_by_accession_number

        accession_no = _extract_accession_number(url)
        if not accession_no:
            # Fall back to Filing.markdown() via URL lookup
            from edgar import Filing as EdgarFiling

            filing = EdgarFiling.load(url)
            if filing is None:
                return f"Could not load filing from URL: {url}"
            text = filing.markdown() or filing.text() or ""
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... truncated ...]"
            return text

        filing = get_by_accession_number(accession_no)
        if filing is None:
            return f"Could not load filing for accession number: {accession_no}"

        # Try structured section extraction for 10-K/10-Q
        try:
            report = filing.obj()
            if hasattr(report, "items") and report.items:
                sections: list[str] = []
                for item_name in report.items:
                    try:
                        section = report[item_name]
                        if section is not None:
                            section_text = str(section)
                            if section_text.strip():
                                sections.append(f"=== {item_name} ===\n\n{section_text}")
                    except Exception:
                        continue

                if sections:
                    text = "\n\n---\n\n".join(sections)
                    if len(text) > max_chars:
                        text = text[:max_chars] + "\n\n[... truncated ...]"
                    return text
        except Exception:
            pass

        # Fall back to full markdown/text
        text = filing.markdown() or filing.text() or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text

    return await asyncio.to_thread(_fetch)


async def get_filing_content(url: str, max_chars: int = 80_000) -> str:
    """Download a filing and extract readable text.

    Retrieves the filing and returns clean text content including both
    tables and qualitative sections.  Prefer ``get_filing_financial_tables``
    and ``get_filing_text`` for structured extraction.

    Args:
        url: Full URL to the filing document on SEC EDGAR.
        max_chars: Maximum characters to return (default 80 000).

    Returns:
        Cleaned text content of the filing.
    """
    _ensure_identity()
    logger.debug("Fetching filing content from %s", url)

    def _fetch() -> str:
        from edgar import get_by_accession_number

        accession_no = _extract_accession_number(url)
        if accession_no:
            filing = get_by_accession_number(accession_no)
            if filing is not None:
                text = filing.markdown() or filing.text() or ""
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[... truncated ...]"
                return text

        # If accession number extraction fails, try direct load
        from edgar import Filing as EdgarFiling

        filing = EdgarFiling.load(url)
        if filing is None:
            return f"Could not load filing from URL: {url}"
        text = filing.markdown() or filing.text() or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text

    return await asyncio.to_thread(_fetch)


async def get_insider_transactions(ticker: str) -> list[dict]:
    """Get recent insider transactions (Form 3/4/5 filings) for a company.

    Retrieves the company's recent insider transaction filings and extracts
    structured transaction data including owner names, dates, transaction
    types, shares, and prices where available.

    Args:
        ticker: Stock ticker symbol (e.g. ``"TSLA"``).

    Returns:
        List of dicts with keys ``owner``, ``transaction_date``,
        ``form``, ``accessionNumber``, and ``url``.
    """
    _ensure_identity()
    logger.debug("Fetching insider transactions for ticker %r", ticker)

    def _fetch() -> list[dict]:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings().filter(form=[3, 4, 5]).head(50)

        transactions: list[dict] = []
        for filing in filings:
            transactions.append({
                "form": filing.form,
                "transaction_date": filing.filing_date,
                "accessionNumber": filing.accession_no,
                "url": filing.filing_url,
            })

        return transactions

    return await asyncio.to_thread(_fetch)


async def get_institutional_holdings(ticker: str) -> list[dict]:
    """Get institutional holdings from 13F filings for a company.

    Retrieves recent 13F-HR filings that reference the company to
    identify major institutional holders.

    Args:
        ticker: Stock ticker symbol (e.g. ``"GOOGL"``).

    Returns:
        List of dicts with institutional holder information including
        ``filer_name``, ``filing_date``, and ``filing_url``.
    """
    logger.debug("Fetching 13F holders for ticker %r", ticker)
    # 13F search still uses EFTS since edgartools' 13F support works
    # from the filer side, not the held-company side.
    params = {
        "q": ticker.upper(),
        "forms": "13F-HR",
    }

    async def _fetch() -> bytes:
        data = await sec_efts_get("/search-index", params=params)
        return json.dumps(data).encode()

    raw = await cached_request(
        "sec_13f", f"13f_{ticker.upper()}", _fetch, ttl=TTL_1_HOUR
    )
    data = json.loads(raw)

    hits = data.get("hits", {}).get("hits", [])
    holdings = []

    for hit in hits:
        source = hit.get("_source", {})
        holdings.append({
            "filer_name": source.get("entity_name", ""),
            "filing_date": source.get("file_date", ""),
            "period_of_report": source.get("period_of_report", ""),
            "file_url": source.get("file_url", ""),
        })

        if len(holdings) >= 50:
            break

    return holdings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")


def _extract_accession_number(url: str) -> str | None:
    """Extract an SEC accession number from a filing URL.

    Accession numbers appear in EDGAR URLs in either dashed format
    (``0000320193-24-000123``) or as a run of 18 digits.

    Args:
        url: An SEC EDGAR filing URL.

    Returns:
        The accession number in dashed format, or ``None`` if not found.
    """
    match = _ACCESSION_RE.search(url)
    if match:
        return match.group(0)

    # Try the 18-digit contiguous format and insert dashes.
    digits_match = re.search(r"/(\d{18})/", url)
    if digits_match:
        d = digits_match.group(1)
        return f"{d[:10]}-{d[10:12]}-{d[12:]}"

    return None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def create_tools() -> list[FunctionTool]:
    """Create LlamaIndex FunctionTool instances for all SEC EDGAR tools."""
    return [
        FunctionTool.from_defaults(
            async_fn=get_company_facts,
            name="get_company_facts",
            description=(
                "Fetch structured XBRL financial data for a company from SEC EDGAR. "
                "Returns standardized income statement, balance sheet, and cash flow "
                "statement data parsed from XBRL filings."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=search_filings,
            name="search_filings",
            description=(
                "Full-text search across SEC filings. Accepts a query string and "
                "optional filters for ticker, filing type, and date range. Returns "
                "matching filings with text snippets."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_submissions,
            name="get_submissions",
            description=(
                "Get the complete filing history and company metadata for a ticker "
                "from SEC EDGAR. Returns CIK, SIC code, and recent filings."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_filing_content,
            name="get_filing_content",
            description=(
                "Download and extract the full text content of a specific SEC filing "
                "by URL. Returns cleaned text including tables. "
                "Prefer get_filing_financial_tables and "
                "get_filing_text for structured extraction."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_filing_urls,
            name="get_filing_urls",
            description=(
                "Discover available SEC filing URLs for a company filtered by type. "
                "Use this first to find 10-K and 10-Q filing URLs before extracting "
                "data with get_filing_financial_tables or get_filing_text. "
                "Returns a list with form type, filing date, accession number, and "
                "URL for each filing."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_filing_financial_tables,
            name="get_filing_financial_tables",
            description=(
                "Extract structured financial statements (income statement, balance "
                "sheet, cash flow) from a SEC filing via XBRL parsing. Requires "
                "ticker and accession number (from get_filing_urls). Returns "
                "pre-classified, standardized financial tables."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_filing_text,
            name="get_filing_text",
            description=(
                "Extract qualitative text sections from a single SEC filing. For "
                "10-K/10-Q filings, returns individually labeled sections (MD&A, "
                "Risk Factors, Business description, etc.). Pass one filing URL "
                "at a time."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_insider_transactions,
            name="get_insider_transactions",
            description=(
                "Get recent insider transactions (Form 3/4/5) for a company. Returns "
                "transaction dates, form types, and filing URLs."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=get_institutional_holdings,
            name="get_institutional_holdings",
            description=(
                "Get institutional holdings from 13F filings mentioning a company. "
                "Returns filer names, filing dates, and URLs."
            ),
        ),
    ]
