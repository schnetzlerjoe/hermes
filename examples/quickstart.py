"""Hermes quickstart -- minimal usage example.

Demonstrates basic configuration and running a simple research query.
Requires: ANTHROPIC_API_KEY environment variable set.
"""

import asyncio

# Step 1: Import the library.
# Hermes exposes three main objects at the top level:
#   - configure()   : set global config (API keys, providers, paths)
#   - HermesConfig   : the config dataclass (for advanced usage)
#   - Hermes         : the main facade that owns tools, agents, and execution
from hermes import Hermes, configure


def main() -> None:
    """Run a minimal Hermes session."""

    # Step 2: Configure the framework.
    # At minimum you need to tell Hermes which LLM provider to use and supply
    # a SEC User-Agent string (required by SEC EDGAR for all API calls).
    # API keys are read from environment variables automatically
    # (HERMES_FRED_API_KEY, ANTHROPIC_API_KEY, etc.) but can also be passed
    # here as keyword arguments.
    configure(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5-20250514",
        sec_user_agent="HermesDemo demo@example.com",
        verbose=True,
    )

    # Step 3: Create a Hermes instance.
    # This lazily registers all built-in tools and agents on first use.
    h = Hermes()

    # Step 4: Inspect what is available.
    # list_tools() returns {tool_name: [tags]} for every registered tool.
    print("Registered tools:")
    for name, tags in h.list_tools().items():
        print(f"  {name}: {tags}")

    # list_agents() returns a flat list of agent names.
    print("\nRegistered agents:")
    for agent_name in h.list_agents():
        print(f"  {agent_name}")

    # Step 5: Run a research query.
    # Hermes.run() is async, so we use asyncio.run() from a sync context.
    # The orchestrator delegates to specialist agents (SEC filings, macro,
    # market data, etc.) and returns a dict with at least:
    #   {"answer": str, "sources": list[str]}
    #
    # NOTE: run() is not yet implemented in v0.1.0 -- this will raise
    # NotImplementedError until v0.2 ships the orchestrator agent.
    try:
        result = asyncio.run(
            h.run("What were Apple's revenue trends over the last 3 years?")
        )
        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])
    except NotImplementedError as exc:
        print(f"\n(Expected in v0.1.0) {exc}")


if __name__ == "__main__":
    main()
