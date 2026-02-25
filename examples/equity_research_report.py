"""Generate a full equity research report with financial model.

End-to-end example using the orchestrator to produce a complete
research deliverable from a single query.  The orchestrator coordinates
specialist agents (SEC filings, macro, market data, modeling, report)
to build the output.

This example demonstrates:
    1. Configuring Hermes with all required API keys.
    2. Running a streamed query and handling events in real time.
    3. Inspecting the final result for generated files and analysis.
"""

from __future__ import annotations

import asyncio
import os
import sys

from hermes import Hermes, HermesConfig, configure
from hermes.core import StreamEvent


async def generate_report(ticker: str) -> None:
    """Generate a full equity research report for the given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "MSFT").
    """

    # ------------------------------------------------------------------
    # Step 1: Configure Hermes with all required credentials.
    # In production, set these as environment variables:
    #   ANTHROPIC_API_KEY, HERMES_SEC_USER_AGENT, HERMES_FRED_API_KEY, etc.
    # ------------------------------------------------------------------
    config = configure(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250514",
        sec_user_agent="HermesResearch research@example.com",
        # fred_api_key=os.getenv("HERMES_FRED_API_KEY"),  # Optional for macro data
        output_dir=f"./output/{ticker.upper()}_report",
        verbose=True,
    )

    print(f"Generating equity research report for {ticker.upper()}")
    print(f"Output directory: {config.output_dir}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Step 2: Create the Hermes instance.
    # ------------------------------------------------------------------
    h = Hermes(config=config)

    # ------------------------------------------------------------------
    # Step 3: Stream the research query.
    #
    # Hermes.stream() yields StreamEvent objects as the orchestrator
    # works.  Each event has a `kind` field:
    #   - "agent_switch"  : orchestrator delegated to a sub-agent
    #   - "tool_call"     : a tool is being invoked
    #   - "token"         : incremental text from the LLM
    #   - "file_created"  : a file (Excel model, report, chart) was saved
    #   - "error"         : something went wrong
    #
    # This lets you build rich UIs with real-time progress indicators.
    # ------------------------------------------------------------------
    query = (
        f"Produce a comprehensive equity research report for {ticker.upper()}. "
        "Include: (1) business overview and competitive positioning, "
        "(2) analysis of the last 3 years of financial statements from SEC filings, "
        "(3) a DCF valuation model in Excel with sensitivity analysis, "
        "(4) relevant macroeconomic context, "
        "(5) a buy/hold/sell recommendation with price target. "
        "Output a formatted Word document and supporting Excel model."
    )

    try:
        # NOTE: stream() is not yet implemented in v0.1.0.  This code
        # shows the intended API for v0.2+.
        generated_files: list[str] = []

        async for event in h.stream(query, ticker=ticker):
            _handle_event(event, generated_files)

        # Print final summary.
        print("\n" + "=" * 60)
        print("Report generation complete!")
        if generated_files:
            print("\nGenerated files:")
            for path in generated_files:
                print(f"  {path}")

    except NotImplementedError:
        # Expected in v0.1.0 -- fall back to showing what would happen.
        print("\nstream() is not yet available in v0.1.0.")
        print("When the orchestrator ships in v0.2, this script will:")
        print(f"  1. Query SEC EDGAR for {ticker.upper()} filings")
        print("  2. Retrieve macro indicators from FRED")
        print("  3. Fetch current market data and pricing")
        print("  4. Build a DCF model in Excel")
        print("  5. Generate a Word document research report")
        print("  6. Create supporting charts and visualisations")

    # ------------------------------------------------------------------
    # Step 4 (alternative): Use run() for non-streamed execution.
    # ------------------------------------------------------------------
    try:
        result = await h.run(query, ticker=ticker)
        print("\nAnswer:", result.get("answer", "N/A"))
        print("Sources:", result.get("sources", []))
        print("Files:", result.get("files", []))
    except NotImplementedError:
        print("\nrun() is also not yet available in v0.1.0.")


def _handle_event(event: StreamEvent, generated_files: list[str]) -> None:
    """Process a single stream event and print progress to stdout.

    Args:
        event: The stream event to handle.
        generated_files: Accumulator list for generated file paths.
    """
    kind = event.kind
    data = event.data

    if kind == "agent_switch":
        agent_name = data.get("agent", "unknown") if isinstance(data, dict) else data
        print(f"\n[Agent] Delegating to: {agent_name}")

    elif kind == "tool_call":
        tool_name = data.get("tool", "unknown") if isinstance(data, dict) else data
        print(f"  [Tool] Calling: {tool_name}")

    elif kind == "token":
        # Incremental text -- print without newline for streaming effect.
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        sys.stdout.write(text)
        sys.stdout.flush()

    elif kind == "file_created":
        path = data.get("path", "") if isinstance(data, dict) else str(data)
        generated_files.append(path)
        print(f"\n  [File] Created: {path}")

    elif kind == "error":
        message = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        print(f"\n  [Error] {message}")


def main() -> None:
    """Entry point -- parse ticker from argv or use default."""
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(generate_report(ticker))


if __name__ == "__main__":
    main()
