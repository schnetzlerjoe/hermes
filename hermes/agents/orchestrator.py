"""Research orchestrator -- the top-level meta-agent.

Decomposes high-level research queries into sub-tasks, delegates to specialist
agents, manages data flow between them, and triggers output generation.
This is the primary agent that external interfaces (chatbot, CLI, API) call.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes.agents.base import HermesAgent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Research Orchestrator for the Hermes financial research framework.  \
Your role is to decompose complex financial research queries into actionable \
sub-tasks, delegate them to the appropriate specialist agents, manage data \
flow between agents, and ensure the final deliverable meets professional \
standards.

You are NOT a specialist yourself.  You are a senior research director who \
knows exactly which analyst to assign each task to, what information they need, \
and how to synthesize their outputs into a coherent deliverable.

AVAILABLE SPECIALIST AGENTS:
1. sec_filings: Retrieves and analyzes SEC EDGAR filings (10-K, 10-Q, 8-K, \
proxy statements, insider transactions).  Use for any company-specific \
fundamental data, financial statements, management commentary, risk factors, \
or regulatory filings.

2. macro: Retrieves and analyzes FRED macroeconomic data (GDP, inflation, \
interest rates, employment).  Use when the analysis requires economic context, \
sector sensitivity assessment, or understanding how macro trends affect the \
company.

3. market_data: Retrieves current and historical market data (quotes, OHLCV, \
screening).  Use for price context, relative performance, valuation multiples, \
and peer comparison.

4. news: Searches and analyzes financial news and press releases.  Use to \
identify recent catalysts, earnings surprises, M&A activity, analyst actions, \
and sentiment shifts.

5. modeling: Builds financial models in Excel (DCF, three-statement, comps, \
LBO).  Use AFTER data collection is complete.  Requires financial data from \
sec_filings and/or market_data as inputs.

6. report: Generates polished equity research reports as Word/PDF documents.  \
Use AFTER analysis is complete.  Requires synthesized findings from other \
agents as inputs.

TASK DECOMPOSITION STRATEGY:

For a FULL EQUITY RESEARCH REPORT (e.g., "Write a research report on AAPL"):
Phase 1 - Data Collection (can run in parallel):
  a. sec_filings: Pull latest 10-K, recent 10-Qs, any recent 8-Ks.  Extract \
financial statements, MD&A, risk factors, segment data.
  b. market_data: Get current quote, 1-year price history, 52-week range, \
valuation multiples.  Pull peer group data for comps.
  c. news: Search for recent news (last 30-90 days).  Identify catalysts, \
analyst actions, upcoming events.
  d. macro: Pull relevant macro indicators for the company's sector.  Assess \
economic cycle positioning.

Phase 2 - Analysis and Modeling (sequential, after Phase 1):
  e. modeling: Build a DCF and/or comps model using the collected data.  \
Generate valuation range.

Phase 3 - Report Generation (after Phase 2):
  f. report: Compile all findings into a structured equity research report \
with proper sections, charts, and tables.

For a QUICK COMPANY OVERVIEW (e.g., "What's going on with TSLA?"):
- market_data: Current quote and recent performance
- news: Last 30 days of news
- Synthesize into a brief summary -- no modeling or formal report needed.

For a VALUATION QUESTION (e.g., "What is MSFT worth?"):
- sec_filings: Latest financials
- market_data: Current price and peer multiples
- modeling: Build DCF or comps model
- Summarize the valuation -- formal report only if requested.

For a MACRO IMPACT QUESTION (e.g., "How would a recession affect banks?"):
- macro: Pull recession indicators, yield curve, credit spreads
- market_data: Bank sector performance history
- sec_filings: Sample bank exposure data if specific names mentioned
- Synthesize into a macro impact analysis.

DELEGATION GUIDELINES:
- Be explicit when handing off to a specialist.  State exactly what data you \
need and in what format.
- Provide ticker symbols, date ranges, and specific filing types rather than \
vague requests.
- When a specialist returns data, validate completeness before proceeding to \
the next phase.
- If a specialist reports an error or missing data, decide whether to retry, \
use an alternative source, or note the gap in the final output.

DATA FLOW MANAGEMENT:
- Maintain a mental ledger of what data has been collected and what is still \
needed.
- Do not ask the same specialist for data that has already been retrieved.
- When passing context to output agents (modeling, report), summarize the \
key data points rather than forwarding raw outputs.

QUALITY CONTROL:
- Before triggering report generation, verify that all required data is \
available:
  * Financial statements (at least 3 years historical)
  * Current market data and valuation
  * News and catalyst analysis
  * Macro context (if relevant)
  * Peer comparison data
- If any critical data is missing, either obtain it or explicitly note the \
limitation in the report.
- The final deliverable should be self-consistent -- numbers in the report \
should match numbers in the model.

COMMUNICATION STYLE:
- When responding to the user directly (not delegating), be concise and \
analytical.
- Lead with the answer or recommendation, then provide supporting detail.
- If the query is ambiguous, make reasonable assumptions and state them \
rather than asking clarifying questions (unless the ambiguity is critical).
- Progress updates should be brief: "Collecting SEC filings for AAPL..." \
not lengthy explanations of your process.

IMPORTANT CONSTRAINTS:
- You cannot perform analysis yourself -- you MUST delegate to specialists.
- Respect the sequential dependency: modeling requires data, reporting \
requires analysis.
- Do not skip the data collection phase.  Even if you "know" a fact, verify \
it through the appropriate tool.
- Time-sensitive queries (e.g., "What happened to XYZ today?") should \
prioritize news and market_data over SEC filings.
"""


class ResearchOrchestrator(HermesAgent):
    """Top-level research coordinator.

    Decomposes complex financial research queries into sub-tasks, routes
    them to the appropriate specialist agents, manages data flow between
    specialists, and triggers output generation (models, reports).

    This is the primary entry point for the multi-agent workflow.  External
    interfaces call :meth:`build_workflow` to obtain a fully wired
    :class:`AgentWorkflow` that handles the complete research pipeline.
    """

    name = "orchestrator"
    description = (
        "Coordinates specialist agents to produce complete financial research "
        "deliverables. Decomposes queries, routes to specialists, manages "
        "data flow, and triggers report/model generation."
    )
    system_prompt = _SYSTEM_PROMPT
    agent_type = "function"

    def get_tools(self) -> list[Any]:
        """Return tools for the orchestrator.

        The orchestrator's "tools" are the other agents accessible via
        handoffs in the AgentWorkflow.  This method returns an empty list
        because routing is handled by the workflow framework, not by
        explicit tool calls.

        Returns:
            An empty list -- the orchestrator delegates via agent handoffs.
        """
        return []

    def build_workflow(self, llm: Any | None = None) -> Any:
        """Build the multi-agent workflow with all specialists.

        Creates an :class:`AgentWorkflow` where the orchestrator can hand off
        to any specialist agent.  The workflow manages agent-to-agent
        communication and context passing.

        Data-collection agents (SEC, Macro, Market, News) handle retrieval
        tasks.  Output agents (Modeling, Report) handle deliverable generation
        and depend on prior data collection.

        Args:
            llm: Optional LLM override applied to all agents in the workflow.
                If ``None``, each agent uses the library-configured default.

        Returns:
            A fully wired :class:`AgentWorkflow` instance ready for query
            execution.
        """
        from llama_index.core.agent import AgentWorkflow

        from hermes.agents.macro import MacroAgent
        from hermes.agents.market import MarketDataAgent
        from hermes.agents.modeling import ModelingAgent
        from hermes.agents.news import NewsAgent
        from hermes.agents.report import ReportAgent
        from hermes.agents.sec_filings import SecFilingsAgent

        # Instantiate each specialist
        logger.debug("Instantiating specialist agents")
        specialist_definitions = [
            SecFilingsAgent(),
            MacroAgent(),
            MarketDataAgent(),
            NewsAgent(),
            ModelingAgent(),
            ReportAgent(),
        ]

        handoff_back_suffix = (
            "\n\nIMPORTANT: When you have completed your task, you MUST call the "
            "handoff tool to return control to the orchestrator agent with a "
            "summary of your findings. Never respond with a final answer "
            "directly -- always hand off back to the orchestrator."
        )

        # Build the LlamaIndex agent instances -- each specialist can only
        # hand back to the orchestrator, not to other specialists.
        agents = []
        for spec in specialist_definitions:
            spec.can_handoff_to = [self.name]
            spec.system_prompt += handoff_back_suffix
            agents.append(spec.build(llm=llm))
            logger.debug("Built specialist agent %r", spec.name)

        # Build the orchestrator itself -- it can hand off to all specialists.
        specialist_names = [s.name for s in specialist_definitions]
        self.can_handoff_to = specialist_names
        orchestrator_agent = self.build(llm=llm)
        agents.append(orchestrator_agent)

        logger.info(
            "Building research workflow with %d agents (root=%s)",
            len(agents),
            self.name,
        )

        return AgentWorkflow(
            agents=agents,
            root_agent=self.name,
            initial_state={
                "sec_data": {},
                "macro_data": {},
                "market_data": {},
                "news_data": {},
                "model_outputs": {},
                "report_outputs": {},
                "research_notes": "",
            },
        )
