# Hermes Financial

Multi-agent financial research framework built on [LlamaIndex](https://docs.llamaindex.ai/).

Hermes gives AI engineers pre-built financial data tools, document ingestion
pipelines, output generation (Excel models, Word/PDF reports), and composable
agents for equity research. Use the agents out of the box, swap in your own,
or extend with custom tools and data sources.

## Quick Start

```bash
# Install with uv (recommended)
uv add hermes-financial

# Or with pip
pip install hermes-financial
```

```python
import asyncio
import hermes

hermes.configure(
    llm_provider="anthropic",
    sec_user_agent="MyApp/1.0 (me@company.com)",  # required by SEC
)

h = hermes.Hermes()

# List available tools and agents
print(h.list_tools())
print(h.list_agents())
```

## What's Included

### Data Tools (35 total)

| Module | Tools | Source |
|--------|-------|--------|
| **SEC EDGAR** | Company facts (XBRL), filing search, submissions, filing URLs, structured financial tables, labeled text sections (MD&A, Risk Factors, etc.), insider transactions, institutional holdings | [edgartools](https://github.com/dgunning/edgartools) + `efts.sec.gov` — free, no key |
| **FRED** | Economic time series, series search, metadata | `api.stlouisfed.org` — free key |
| **Market Data** | Quotes, historical OHLCV, batch quotes | Yahoo Finance — free, no key |
| **News** | Company news, general financial news | Yahoo/Google RSS — free, no key |
| **Excel** | Create, write, read, format, formulas, charts, save `.xlsx` | `openpyxl` |
| **Documents** | Create, headings, paragraphs, tables, images, save `.docx`, export PDF | `python-docx` + LibreOffice |
| **Charts** | Line, bar, waterfall, scatter, heatmap → PNG | `matplotlib` |

### Specialist Agents (7)

| Agent | Type | Role |
|-------|------|------|
| `SecFilingsAgent` | FunctionAgent | SEC filing retrieval and analysis |
| `MacroAgent` | FunctionAgent | FRED macroeconomic data |
| `MarketDataAgent` | FunctionAgent | Quotes and historical prices |
| `NewsAgent` | FunctionAgent | Financial news and sentiment |
| `ModelingAgent` | ReActAgent | Excel financial model construction |
| `ReportAgent` | ReActAgent | Word/PDF report generation |
| `ResearchOrchestrator` | ReActAgent | Multi-agent coordination |

### Infrastructure

- **File-based cache** with per-item TTL (filings cached permanently, quotes never cached)
- **Async rate limiter** (token bucket) for SEC EDGAR, FRED, Yahoo Finance
- **Streaming events** for real-time progress reporting
- **ChromaDB vector indices** for semantic search over financial documents

## Configuration

All settings can be passed to `configure()` or set via environment variables
with the `HERMES_` prefix:

```python
hermes.configure(
    llm_provider="anthropic",           # or "openai"
    llm_model="claude-sonnet-4-5-20250514",
    sec_user_agent="MyApp/1.0 (me@company.com)",
    fred_api_key="your_key",
    output_dir="./output",
    verbose=True,
)
```

Environment variables: `HERMES_LLM_PROVIDER`, `HERMES_FRED_API_KEY`, etc.

## Extending Hermes

### Register a Custom Tool

```python
from llama_index.core.tools import FunctionTool

def my_bloomberg_lookup(ticker: str) -> dict:
    """Look up data from Bloomberg."""
    ...

tool = FunctionTool.from_defaults(fn=my_bloomberg_lookup, name="bbg_lookup")

h = hermes.Hermes()
h.register_tool("bbg_lookup", tool, tags=["market_data"])
```

### Register a Custom Agent

```python
from hermes.agents import HermesAgent

class OptionsFlowAgent(HermesAgent):
    name = "options_flow"
    description = "Analyzes unusual options activity and flow data."
    system_prompt = "You are an options flow specialist..."
    agent_type = "function"

    def get_tools(self):
        return [...]

h = hermes.Hermes()
h.register_agent("options_flow", OptionsFlowAgent)
```

## Using Tools Directly

You don't need agents to use the tools. Every tool module exposes async
functions that work standalone:

```python
from hermes.tools.sec_edgar import (
    get_company_facts,
    get_filing_urls,
    get_filing_financial_tables,
    get_filing_text,
)
from hermes.tools.fred import get_series
from hermes.tools.excel import excel_create_workbook, excel_write_cells, excel_save

# Fetch Apple's multi-period XBRL financial data
facts = await get_company_facts("AAPL")

# Discover recent 10-K and 10-Q filings
filings = await get_filing_urls("AAPL", filing_types="10-K,10-Q", limit=10)

# Extract XBRL-parsed, pre-classified financial statements from a filing
tables = await get_filing_financial_tables("AAPL", filings[0]["accessionNumber"])

# Extract labeled qualitative sections (MD&A, Risk Factors, Business, etc.)
text = await get_filing_text(filings[0]["url"])

# Pull GDP data from FRED
gdp = await get_series("GDP", start_date="2020-01-01")

# Build an Excel workbook
wb_id = excel_create_workbook("model", sheets=["Assumptions", "Income Statement"])
excel_write_cells(wb_id, "Assumptions", {"A1": "Revenue Growth", "B1": 0.12})
filepath = excel_save(wb_id, "my_model.xlsx")
```

## Project Structure

```
hermes/
├── __init__.py              # Public API: Hermes, configure
├── config.py                # Pydantic configuration
├── registry.py              # Tool/agent registry
├── core.py                  # Hermes facade class
├── tools/
│   ├── _base.py             # HTTP client, rate limiter, cache helpers
│   ├── sec_edgar.py         # SEC EDGAR (XBRL, EFTS, submissions)
│   ├── fred.py              # FRED macroeconomic data
│   ├── market_data.py       # Yahoo Finance quotes and history
│   ├── news.py              # RSS-based news retrieval
│   ├── excel.py             # Excel workbook manipulation
│   ├── documents.py         # Word document generation + PDF export
│   └── charts.py            # matplotlib chart generation
├── agents/
│   ├── base.py              # HermesAgent abstract base class
│   ├── sec_filings.py       # SEC filings specialist
│   ├── macro.py             # Macroeconomic data specialist
│   ├── market.py            # Market data specialist
│   ├── news.py              # News/sentiment specialist
│   ├── modeling.py          # Financial modeling (Excel)
│   ├── report.py            # Report writing (Word/PDF)
│   └── orchestrator.py      # Multi-agent coordinator
├── ingestion/
│   ├── sec_parser.py        # SEC filing HTML → LlamaIndex nodes
│   ├── transcript_parser.py # Earnings transcript parser
│   └── index_manager.py     # ChromaDB index management
└── infra/
    ├── cache.py             # File-based cache with TTL
    ├── rate_limiter.py      # Async token bucket rate limiter
    └── streaming.py         # Streaming event types
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/hermes-financial/hermes-financial.git
cd hermes-financial
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check hermes/

# Type check
uv run mypy hermes/
```

## Requirements

- Python >= 3.10
- An LLM API key (Anthropic or OpenAI) for agent functionality
- SEC EDGAR requires a User-Agent string identifying your application
- FRED requires a free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
- LibreOffice for PDF export (optional, available in Docker image)

## Professional Services

Need this deployed internally, integrated with your data sources, or extended
with custom agents and models? [joe@pital.dev](mailto:joe@pital.dev)

## Acknowledgments

SEC EDGAR data retrieval is powered by [edgartools](https://github.com/dgunning/edgartools)
(MIT), an excellent Python library by Dwight Gunning that provides high-level access to
SEC EDGAR filings, XBRL-parsed financial statements, and structured document sections.
If you find Hermes useful for SEC research, please consider starring the edgartools repo.

## License

MIT
