"""Agent definitions for the Hermes research framework.

Each agent is a specialist with a focused set of tools and a system prompt
tailored to its domain.  The orchestrator coordinates these specialists to
produce complete financial research deliverables.

Available agents:

* :class:`SecFilingsAgent` -- SEC EDGAR filings retrieval and analysis
* :class:`MacroAgent` -- FRED macroeconomic data
* :class:`MarketDataAgent` -- Prices, volumes, and screening
* :class:`NewsAgent` -- Financial news search and analysis
* :class:`ModelingAgent` -- Excel financial model construction
* :class:`ReportAgent` -- Research report generation
* :class:`ResearchOrchestrator` -- Top-level coordinator
"""

from hermes.agents.base import HermesAgent

__all__ = ["HermesAgent"]
