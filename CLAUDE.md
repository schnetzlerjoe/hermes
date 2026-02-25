# CLAUDE.md — Agent Instructions for hermes-financial

This file provides context for AI coding assistants working on this codebase.

## Project Overview

**hermes-financial** is an open-source Python library (MIT) providing a
multi-agent financial research framework built on LlamaIndex. It ships with
data retrieval tools, document ingestion pipelines, output generation (Excel,
Word/PDF, charts), and composable agents for equity research.

A separate closed-source chatbot product (hermes-app) is built on top of this
library. This repo contains only the open-source library.

## Tech Stack

- **Python >= 3.10** — modern type hints (`X | None`, not `Optional[X]`)
- **LlamaIndex** — agent framework, tool wrappers, RAG pipelines
- **Pydantic v2** — configuration and data models
- **httpx** — async HTTP client for all API calls
- **openpyxl** — Excel workbook creation and manipulation
- **python-docx** — Word document generation
- **matplotlib** — chart generation (static PNG)
- **ChromaDB** — local vector store for document retrieval
- **BeautifulSoup4** — SEC filing HTML parsing

## Package Manager

We use **uv** as the primary package manager. All commands should use `uv run`.
The project must also remain pip-installable via `pyproject.toml`.

```bash
uv sync --all-extras    # install everything
uv run pytest           # run tests
uv run ruff check       # lint
uv run mypy hermes/     # type check
```

## Architecture

### Module Layout

```
hermes/
├── config.py        — Pydantic config with HERMES_ env prefix
├── registry.py      — Tool/agent registry (no global state)
├── core.py          — Hermes facade class
├── tools/           — Data + output tools (each exposes create_tools())
├── agents/          — Specialist agents (each inherits HermesAgent)
├── ingestion/       — Document parsing and ChromaDB indexing
└── infra/           — Cache, rate limiter, streaming events
```

### Key Patterns

- **Tool modules** expose a `create_tools() -> list[FunctionTool]` factory.
  Each tool function is a standalone async function that can be used without
  the agent layer.
- **Agent classes** inherit from `HermesAgent` (ABC) and implement `get_tools()`.
  The `agent_type` attribute selects between `FunctionAgent` (tool-calling)
  and `ReActAgent` (chain-of-thought).
- **Registry** — tools and agents are registered by name with optional tags.
  The `Hermes` class owns a `Registry` instance (no singletons).
- **Config** — `HermesConfig` is a Pydantic model with env var support.
  `configure()` creates/updates a module-level instance; `get_config()`
  provides lazy access.
- **Rate limiting** — all HTTP tools use `get_limiter(name)` from
  `infra/rate_limiter.py`. SEC EDGAR, FRED, and Yahoo Finance each have
  pre-configured token bucket limits.
- **Caching** — `infra/cache.py` provides a file-based cache with per-item TTL.
  The `cached_request()` helper in `tools/_base.py` wraps the fetch-or-cache
  pattern. Filing HTML is cached permanently; metadata has short TTLs.

### Agent Types

| Agent | Type | Rationale |
|-------|------|-----------|
| SecFilingsAgent, MacroAgent, MarketDataAgent, NewsAgent | FunctionAgent | Straightforward data retrieval |
| ModelingAgent, ReportAgent, ResearchOrchestrator | ReActAgent | Multi-step reasoning required |

Default to `FunctionAgent`. Only use `ReActAgent` where the reasoning chain helps.

## Conventions

### Code Style

- **ruff** for linting and formatting (line-length 99, target py310)
- **Type hints everywhere** — use `X | None` not `Optional[X]`
- **`from __future__ import annotations`** at the top of every module
- **Docstrings** — use Google/NumPy style, include Args/Returns sections
- **Logging** — `logger = logging.getLogger(__name__)` per module
- **No global mutable state** except the config singleton and rate limiter pool
- Async by default for all network-facing code

### Naming

- Tool functions: `verb_noun` (e.g., `get_company_facts`, `excel_write_cells`)
- Agent classes: `{Domain}Agent` (e.g., `SecFilingsAgent`, `MacroAgent`)
- Config fields: `snake_case`, env vars are `HERMES_UPPER_SNAKE`
- Cache namespaces: `source_type` (e.g., `sec_facts`, `fred_series`)

### Testing

- Tests in `tests/` mirroring the package structure
- Use `pytest` with `pytest-asyncio` (mode=auto)
- Mock HTTP calls for unit tests; mark live network tests with `@pytest.mark.network`
- Test fixtures in `tests/conftest.py`
- Tests should create real files (Excel, Word) in temp dirs and verify contents

### Dependencies

- Core deps in `[project.dependencies]` — keep minimal
- Optional extras: `llamaparse`, `web`, `pandas`, `dev`
- Never add a dependency without checking if an existing one covers the need

## Data Sources

### Tier 1 (built-in, free)

- **SEC EDGAR** — XBRL facts, full-text search, submissions, filings (`data.sec.gov`)
- **FRED** — 800k+ economic series (`api.stlouisfed.org`, free key)
- **Yahoo Finance** — quotes, OHLCV history (public endpoints, no key)

### Tier 2 (optional, user provides key)

- Polygon.io, Financial Modeling Prep, Alpha Vantage, Benzinga

### SEC EDGAR Rules

- **Must set `sec_user_agent`** — SEC blocks requests without a descriptive User-Agent
- **Rate limit: 10 req/sec** — enforced by the built-in rate limiter
- Filing HTML is cached permanently (filings never change once filed)
- XBRL data cached 24h, submissions cached 1h, ticker maps cached 7d

## Common Tasks

### Adding a New Data Tool

1. Create `hermes/tools/my_source.py`
2. Write async tool functions with proper docstrings
3. Add a `create_tools() -> list[FunctionTool]` factory at the bottom
4. Add tests in `tests/test_tools/test_my_source.py`
5. Wire into `_register_builtin_tools()` in `core.py` when ready

### Adding a New Agent

1. Create `hermes/agents/my_agent.py`
2. Inherit from `HermesAgent`, set name/description/system_prompt/agent_type
3. Implement `get_tools()` returning the relevant tool list
4. Add to the orchestrator's `build_workflow()` if it should be auto-included

### Version Scope

- **v0.1** (current): All tools, agents, infra, ingestion. No templates.
- **v0.2** (planned): Excel/doc templates, plugin architecture, custom template
  registration, HuggingFace embeddings for local-only setup.
