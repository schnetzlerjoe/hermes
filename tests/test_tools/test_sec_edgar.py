"""Tests for SEC EDGAR tools.

Tests that require real network access are marked with @pytest.mark.network
and skipped by default.  All other tests use mocked edgartools objects and
EFTS responses to verify the logic in isolation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes.infra.cache import FileCache

# ---------------------------------------------------------------------------
# Stub out llama_index if it is not installed so the sec_edgar module can
# be imported in test environments that lack the full dependency tree.
# ---------------------------------------------------------------------------
if "llama_index" not in sys.modules:
    _li = types.ModuleType("llama_index")
    _li_core = types.ModuleType("llama_index.core")
    _li_tools = types.ModuleType("llama_index.core.tools")

    class _FakeFunctionTool:
        """Minimal stand-in for FunctionTool used only to allow import."""

        @classmethod
        def from_defaults(cls, **kwargs):
            return cls()

    _li_tools.FunctionTool = _FakeFunctionTool  # type: ignore[attr-defined]
    _li_core.tools = _li_tools  # type: ignore[attr-defined]
    _li.core = _li_core  # type: ignore[attr-defined]
    sys.modules["llama_index"] = _li
    sys.modules["llama_index.core"] = _li_core
    sys.modules["llama_index.core.tools"] = _li_tools


# ---------------------------------------------------------------------------
# Sample response data
# ---------------------------------------------------------------------------

SAMPLE_EFTS_RESPONSE = {
    "hits": {
        "total": {"value": 2},
        "hits": [
            {
                "_source": {
                    "entity_name": "Apple Inc.",
                    "file_date": "2024-11-01",
                    "file_type": "10-K",
                    "file_url": "https://www.sec.gov/Archives/edgar/data/320193/filing.htm",
                    "period_of_report": "2024-09-28",
                },
                "highlight": {
                    "_content": ["...revenue recognition policy..."],
                },
            },
            {
                "_source": {
                    "entity_name": "Apple Inc.",
                    "file_date": "2023-11-03",
                    "file_type": "10-K",
                    "file_url": "https://www.sec.gov/Archives/edgar/data/320193/filing2.htm",
                    "period_of_report": "2023-09-30",
                },
                "highlight": {
                    "_content": ["...revenue growth discussion..."],
                },
            },
        ],
    }
}


# ---------------------------------------------------------------------------
# Helpers to build mock edgartools objects
# ---------------------------------------------------------------------------


def _make_mock_company(
    name: str = "Apple Inc.",
    cik: int = 320193,
    tickers: list[str] | None = None,
    sic: str = "3571",
    fiscal_year_end: str = "0928",
    filings: list | None = None,
):
    """Return a mock ``edgar.Company`` instance."""
    company = MagicMock()
    company.name = name
    company.cik = cik
    company.tickers = tickers or ["AAPL"]
    company.sic = sic
    company.fiscal_year_end = fiscal_year_end

    mock_filings = _make_mock_filings(filings or [])
    company.get_filings.return_value = mock_filings

    mock_financials = _make_mock_multi_financials()
    company.get_financials.return_value = mock_financials

    return company


def _make_mock_filing(
    form: str = "10-K",
    filing_date: str = "2024-11-01",
    accession_no: str = "0000320193-24-000123",
    filing_url: str = "https://www.sec.gov/Archives/edgar/data/320193/filing.htm",
):
    """Return a mock ``edgar.Filing`` instance."""
    filing = MagicMock()
    filing.form = form
    filing.filing_date = filing_date
    filing.accession_no = accession_no
    filing.filing_url = filing_url
    filing.url = filing_url

    # Mock obj() for financial tables
    report = MagicMock()
    report.income_statement.return_value = MagicMock(
        __str__=lambda self: "Revenue | 394,328\nNet Income | 93,736"
    )
    report.balance_sheet.return_value = MagicMock(
        __str__=lambda self: "Total Assets | 352,583\nTotal Liabilities | 290,437"
    )
    report.cashflow_statement.return_value = MagicMock(
        __str__=lambda self: "Operating Cash Flow | 118,254"
    )
    filing.obj.return_value = report

    # Mock text/markdown
    filing.markdown.return_value = "# Apple Inc 10-K\n\nSample filing text."
    filing.text.return_value = "Apple Inc 10-K\n\nSample filing text."

    return filing


def _make_mock_filings(filing_list: list | None = None):
    """Return a mock ``edgar.Filings`` collection."""
    if filing_list is None:
        filing_list = [_make_mock_filing()]

    filings = MagicMock()
    filings.__iter__ = MagicMock(return_value=iter(filing_list))
    filings.__len__ = MagicMock(return_value=len(filing_list))
    filings.head.return_value = filings
    filings.filter.return_value = filings

    def get_filing_at(idx):
        if idx < len(filing_list):
            return filing_list[idx]
        return None

    filings.get_filing_at = get_filing_at

    return filings


def _make_mock_multi_financials():
    """Return a mock ``edgar.MultiFinancials`` instance."""
    mf = MagicMock()

    income = MagicMock()
    income.__str__ = lambda self: "Revenue | 394,328\nNet Income | 93,736"
    mf.income_statement.return_value = income

    balance = MagicMock()
    balance.__str__ = lambda self: "Total Assets | 352,583"
    mf.balance_sheet.return_value = balance

    cashflow = MagicMock()
    cashflow.__str__ = lambda self: "Operating Cash Flow | 118,254"
    mf.cashflow_statement.return_value = cashflow

    return mf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache(tmp_path: Path):
    """Ensure each test gets a fresh cache so mocked data does not leak."""
    cache = FileCache(base_dir=str(tmp_path / "test_sec_cache"))
    with patch("hermes.tools._base.get_cache", return_value=cache):
        yield


@pytest.fixture(autouse=True)
def _set_sec_user_agent():
    """Ensure sec_user_agent is configured for all tests."""
    with patch(
        "hermes.tools.sec_edgar.get_config",
        return_value=type(
            "FakeConfig", (), {"sec_user_agent": "TestSuite test@example.com"}
        )(),
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_identity():
    """Reset the edgartools identity flag between tests."""
    import hermes.tools.sec_edgar as mod

    mod._identity_set = False
    yield
    mod._identity_set = False


# ---------------------------------------------------------------------------
# Tests: _ensure_identity
# ---------------------------------------------------------------------------


class TestEnsureIdentity:
    """Test the edgartools identity initialization."""

    def test_sets_identity_once(self) -> None:
        """_ensure_identity should call set_identity on first invocation."""
        with patch("edgar.set_identity") as mock_set:
            from hermes.tools.sec_edgar import _ensure_identity

            _ensure_identity()
            mock_set.assert_called_once_with("TestSuite test@example.com")

    def test_idempotent(self) -> None:
        """Repeated calls should not call set_identity again."""
        with patch("edgar.set_identity") as mock_set:
            from hermes.tools.sec_edgar import _ensure_identity

            _ensure_identity()
            _ensure_identity()
            mock_set.assert_called_once()

    def test_raises_without_user_agent(self) -> None:
        """Should raise ValueError when sec_user_agent is not set."""
        import hermes.tools.sec_edgar as mod

        mod._identity_set = False
        with patch(
            "hermes.tools.sec_edgar.get_config",
            return_value=type("FakeConfig", (), {"sec_user_agent": ""})(),
        ):
            with pytest.raises(ValueError, match="sec_user_agent"):
                mod._ensure_identity()


# ---------------------------------------------------------------------------
# Tests: _extract_accession_number
# ---------------------------------------------------------------------------


class TestExtractAccessionNumber:
    """Test accession number extraction from URLs."""

    def test_dashed_format(self) -> None:
        from hermes.tools.sec_edgar import _extract_accession_number

        url = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-24-000123/doc.htm"
        assert _extract_accession_number(url) == "0000320193-24-000123"

    def test_contiguous_digits(self) -> None:
        from hermes.tools.sec_edgar import _extract_accession_number

        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/doc.htm"
        assert _extract_accession_number(url) == "0000320193-24-000123"

    def test_no_match(self) -> None:
        from hermes.tools.sec_edgar import _extract_accession_number

        assert _extract_accession_number("https://example.com/noaccession") is None


# ---------------------------------------------------------------------------
# Tests: get_company_facts
# ---------------------------------------------------------------------------


class TestGetCompanyFacts:
    """Test structured financial data retrieval."""

    @pytest.mark.asyncio
    async def test_returns_financials(self) -> None:
        """get_company_facts should return parsed financial statements."""
        from hermes.tools.sec_edgar import get_company_facts

        mock_company = _make_mock_company()

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_company_facts("AAPL")

        assert result["entityName"] == "Apple Inc."
        assert "income_statement" in result["financials"]
        assert "balance_sheet" in result["financials"]
        assert "cashflow_statement" in result["financials"]
        assert "394,328" in result["financials"]["income_statement"]

    @pytest.mark.asyncio
    async def test_returns_company_metadata(self) -> None:
        """Result should include CIK and tickers."""
        from hermes.tools.sec_edgar import get_company_facts

        mock_company = _make_mock_company()

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_company_facts("AAPL")

        assert result["cik"] == 320193
        assert result["tickers"] == ["AAPL"]


# ---------------------------------------------------------------------------
# Tests: search_filings (still EFTS-based)
# ---------------------------------------------------------------------------


class TestSearchFilings:
    """Test full-text search across SEC filings."""

    @pytest.mark.asyncio
    async def test_returns_normalised_results(self) -> None:
        """search_filings should normalise EFTS response into a clean list."""
        from hermes.tools.sec_edgar import search_filings

        async def mock_efts_get(path, params=None):
            return SAMPLE_EFTS_RESPONSE

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            results = await search_filings("revenue recognition")

        assert len(results) == 2
        assert results[0]["entity_name"] == "Apple Inc."
        assert results[0]["file_type"] == "10-K"
        assert results[0]["file_date"] == "2024-11-01"

    @pytest.mark.asyncio
    async def test_search_includes_snippets(self) -> None:
        """Search results should include text snippets from highlights."""
        from hermes.tools.sec_edgar import search_filings

        async def mock_efts_get(path, params=None):
            return SAMPLE_EFTS_RESPONSE

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            results = await search_filings("revenue recognition")

        assert "revenue recognition policy" in results[0]["snippet"]

    @pytest.mark.asyncio
    async def test_search_empty_results(self) -> None:
        """An empty EFTS response should return an empty list."""
        from hermes.tools.sec_edgar import search_filings

        async def mock_efts_get(path, params=None):
            return {"hits": {"total": {"value": 0}, "hits": []}}

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            results = await search_filings("nonexistent obscure query 12345")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_ticker_filter(self) -> None:
        """When a ticker is provided, it should be included in the query."""
        from hermes.tools.sec_edgar import search_filings

        captured_params = {}

        async def mock_efts_get(path, params=None):
            captured_params.update(params or {})
            return {"hits": {"total": {"value": 0}, "hits": []}}

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            await search_filings("revenue", ticker="AAPL")

        assert "AAPL" in captured_params.get("q", "")

    @pytest.mark.asyncio
    async def test_search_with_filing_type_filter(self) -> None:
        """A filing_type filter should be passed to the EFTS API."""
        from hermes.tools.sec_edgar import search_filings

        captured_params = {}

        async def mock_efts_get(path, params=None):
            captured_params.update(params or {})
            return {"hits": {"total": {"value": 0}, "hits": []}}

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            await search_filings("revenue", filing_type="10-K")

        assert captured_params.get("forms") == "10-K"


# ---------------------------------------------------------------------------
# Tests: get_submissions
# ---------------------------------------------------------------------------


class TestGetSubmissions:
    """Test filing history retrieval."""

    @pytest.mark.asyncio
    async def test_returns_company_info_and_filings(self) -> None:
        """get_submissions should return company metadata and filing list."""
        from hermes.tools.sec_edgar import get_submissions

        filing1 = _make_mock_filing(form="10-K", filing_date="2024-11-01")
        filing2 = _make_mock_filing(
            form="10-Q",
            filing_date="2024-08-02",
            accession_no="0000320193-24-000456",
        )
        mock_company = _make_mock_company(filings=[filing1, filing2])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_submissions("AAPL")

        assert result["name"] == "Apple Inc."
        assert result["cik"] == 320193
        assert len(result["recentFilings"]) == 2
        assert result["recentFilings"][0]["form"] == "10-K"
        assert result["recentFilings"][1]["form"] == "10-Q"


# ---------------------------------------------------------------------------
# Tests: get_filing_urls
# ---------------------------------------------------------------------------


class TestGetFilingUrls:
    """Test filing URL discovery."""

    @pytest.mark.asyncio
    async def test_filters_by_form_type(self) -> None:
        """get_filing_urls should filter filings by form type."""
        from hermes.tools.sec_edgar import get_filing_urls

        filing1 = _make_mock_filing(form="10-K", filing_date="2024-11-01")
        filing2 = _make_mock_filing(form="10-Q", filing_date="2024-08-02")
        mock_company = _make_mock_company(filings=[filing1, filing2])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_filing_urls("AAPL", "10-K,10-Q", limit=10)

        assert len(result) == 2
        assert all("form" in f for f in result)
        assert all("accessionNumber" in f for f in result)
        assert all("url" in f for f in result)

    @pytest.mark.asyncio
    async def test_sorted_by_date_descending(self) -> None:
        """Results should be sorted by filing date, most recent first."""
        from hermes.tools.sec_edgar import get_filing_urls

        filing1 = _make_mock_filing(form="10-K", filing_date="2023-11-01")
        filing2 = _make_mock_filing(form="10-K", filing_date="2024-11-01")
        mock_company = _make_mock_company(filings=[filing1, filing2])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_filing_urls("AAPL", "10-K")

        assert result[0]["filingDate"] == "2024-11-01"
        assert result[1]["filingDate"] == "2023-11-01"


# ---------------------------------------------------------------------------
# Tests: get_filing_financial_tables
# ---------------------------------------------------------------------------


class TestGetFilingFinancialTables:
    """Test XBRL financial statement extraction."""

    @pytest.mark.asyncio
    async def test_returns_classified_statements(self) -> None:
        """Should return pre-classified income, balance, cash flow sections."""
        from hermes.tools.sec_edgar import get_filing_financial_tables

        filing = _make_mock_filing()
        mock_company = _make_mock_company(filings=[filing])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_filing_financial_tables("AAPL", "0000320193-24-000123")

        assert "INCOME STATEMENT" in result
        assert "BALANCE SHEET" in result
        assert "CASH FLOW STATEMENT" in result
        assert "394,328" in result

    @pytest.mark.asyncio
    async def test_not_found_accession(self) -> None:
        """Should return an error message for an unknown accession number."""
        from hermes.tools.sec_edgar import get_filing_financial_tables

        mock_company = _make_mock_company(filings=[])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_filing_financial_tables("AAPL", "9999999999-99-999999")

        assert "No filing found" in result


# ---------------------------------------------------------------------------
# Tests: get_filing_text
# ---------------------------------------------------------------------------


class TestGetFilingText:
    """Test qualitative text extraction."""

    @pytest.mark.asyncio
    async def test_returns_text_content(self) -> None:
        """get_filing_text should return filing text."""
        from hermes.tools.sec_edgar import get_filing_text

        mock_filing = _make_mock_filing()
        # Simulate obj() returning a report with items
        report = MagicMock()
        report.items = ["Item 1", "Item 7"]
        section1 = MagicMock()
        section1.__str__ = lambda self: "Apple designs and manufactures consumer electronics."
        section7 = MagicMock()
        section7.__str__ = lambda self: "Revenue increased 5% year-over-year."
        report.__getitem__ = lambda self, key: {
            "Item 1": section1,
            "Item 7": section7,
        }.get(key)
        mock_filing.obj.return_value = report

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch(
                "edgar.get_by_accession_number",
                return_value=mock_filing,
            ),
        ):
            url = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-24-000123/doc.htm"
            result = await get_filing_text(url)

        assert "Item 1" in result
        assert "Item 7" in result
        assert "consumer electronics" in result

    @pytest.mark.asyncio
    async def test_truncates_long_text(self) -> None:
        """Should truncate text exceeding max_chars."""
        from hermes.tools.sec_edgar import get_filing_text

        mock_filing = MagicMock()
        mock_filing.obj.side_effect = Exception("no structured data")
        mock_filing.markdown.return_value = "x" * 200
        mock_filing.text.return_value = "x" * 200

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.get_by_accession_number", return_value=mock_filing),
        ):
            url = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-24-000123/doc.htm"
            result = await get_filing_text(url, max_chars=100)

        assert len(result) < 200
        assert "[... truncated ...]" in result


# ---------------------------------------------------------------------------
# Tests: get_filing_content
# ---------------------------------------------------------------------------


class TestGetFilingContent:
    """Test full filing content retrieval."""

    @pytest.mark.asyncio
    async def test_returns_markdown_content(self) -> None:
        """get_filing_content should return markdown text from the filing."""
        from hermes.tools.sec_edgar import get_filing_content

        mock_filing = _make_mock_filing()

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.get_by_accession_number", return_value=mock_filing),
        ):
            url = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-24-000123/doc.htm"
            result = await get_filing_content(url)

        assert "Apple Inc 10-K" in result


# ---------------------------------------------------------------------------
# Tests: get_insider_transactions
# ---------------------------------------------------------------------------


class TestGetInsiderTransactions:
    """Test insider transaction retrieval."""

    @pytest.mark.asyncio
    async def test_returns_form_4_filings(self) -> None:
        """Should return insider transaction filing metadata."""
        from hermes.tools.sec_edgar import get_insider_transactions

        form4 = _make_mock_filing(
            form="4",
            filing_date="2024-12-15",
            accession_no="0000320193-24-000789",
        )
        mock_company = _make_mock_company(filings=[form4])

        with (
            patch("hermes.tools.sec_edgar._ensure_identity"),
            patch("edgar.Company", return_value=mock_company),
        ):
            result = await get_insider_transactions("AAPL")

        assert len(result) == 1
        assert result[0]["form"] == "4"
        assert result[0]["transaction_date"] == "2024-12-15"


# ---------------------------------------------------------------------------
# Tests: get_institutional_holdings (still EFTS-based)
# ---------------------------------------------------------------------------


class TestGetInstitutionalHoldings:
    """Test institutional holdings retrieval."""

    @pytest.mark.asyncio
    async def test_returns_13f_filers(self) -> None:
        """Should return 13F filer information from EFTS search."""
        from hermes.tools.sec_edgar import get_institutional_holdings

        efts_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "entity_name": "Vanguard Group Inc",
                            "file_date": "2024-08-14",
                            "period_of_report": "2024-06-30",
                            "file_url": "https://www.sec.gov/Archives/edgar/data/102909/13f.htm",
                        },
                    },
                ],
            }
        }

        async def mock_efts_get(path, params=None):
            return efts_response

        with patch("hermes.tools.sec_edgar.sec_efts_get", side_effect=mock_efts_get):
            result = await get_institutional_holdings("AAPL")

        assert len(result) == 1
        assert result[0]["filer_name"] == "Vanguard Group Inc"


# ---------------------------------------------------------------------------
# Tests: create_tools
# ---------------------------------------------------------------------------


class TestCreateTools:
    """Test tool registration factory."""

    def test_creates_all_tools(self) -> None:
        """create_tools should return 9 tool instances."""
        from hermes.tools.sec_edgar import create_tools

        tools = create_tools()
        assert len(tools) == 9


# ---------------------------------------------------------------------------
# Network tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestSecEdgarNetwork:
    """Integration tests that hit the real SEC EDGAR API.

    Run with: pytest -m network
    These require internet access and a valid sec_user_agent config.
    """

    @pytest.mark.asyncio
    async def test_live_get_company_facts(self) -> None:
        """Fetch AAPL financials against the live SEC API."""
        from hermes.tools.sec_edgar import get_company_facts

        result = await get_company_facts("AAPL")
        assert result["entityName"]
        assert "income_statement" in result["financials"]
