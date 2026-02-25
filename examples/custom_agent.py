"""How to build and register a custom agent with Hermes.

Demonstrates creating a specialist agent with custom tools and
registering it with the framework for orchestrator access.

This example builds an OptionsFlowAgent that could analyse unusual
options activity.  The pattern shown here -- subclass HermesAgent,
define metadata, implement get_tools() -- is the same for any
domain-specific agent you want to add.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.tools import FunctionTool

from hermes import Hermes, configure
from hermes.agents.base import HermesAgent


# ---------------------------------------------------------------------------
# Step 1: Define your custom tool functions.
# These are plain Python functions (sync or async) that the agent will call.
# Each should have a clear docstring -- LlamaIndex uses it as the tool
# description the LLM sees when deciding which tool to invoke.
# ---------------------------------------------------------------------------


async def fetch_options_chain(ticker: str, expiry: str | None = None) -> dict:
    """Fetch the options chain for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        expiry: Optional expiration date in YYYY-MM-DD format.
            If omitted, returns the nearest monthly expiry.

    Returns:
        Dict with ``calls`` and ``puts`` lists, each containing
        strike, bid, ask, volume, open_interest, and implied_vol.
    """
    # In a real implementation you would call an options data API
    # (e.g. Polygon, Tradier, CBOE).  Here we return a stub.
    return {
        "ticker": ticker.upper(),
        "expiry": expiry or "2025-06-20",
        "calls": [
            {
                "strike": 200.0,
                "bid": 5.10,
                "ask": 5.30,
                "volume": 12500,
                "open_interest": 45000,
                "implied_vol": 0.28,
            },
        ],
        "puts": [
            {
                "strike": 200.0,
                "bid": 4.80,
                "ask": 5.00,
                "volume": 8700,
                "open_interest": 31000,
                "implied_vol": 0.30,
            },
        ],
    }


async def detect_unusual_activity(
    ticker: str, volume_threshold: int = 10000
) -> list[dict]:
    """Detect unusual options activity based on volume spikes.

    Scans recent options trades for contracts where volume significantly
    exceeds open interest, which can indicate institutional positioning.

    Args:
        ticker: Stock ticker symbol.
        volume_threshold: Minimum volume to flag as unusual.

    Returns:
        List of dicts describing unusual contracts with ``strike``,
        ``expiry``, ``call_put``, ``volume``, ``open_interest``,
        and ``vol_oi_ratio``.
    """
    # Stub implementation.  A real version would query a streaming
    # options feed and apply statistical filters.
    return [
        {
            "strike": 210.0,
            "expiry": "2025-06-20",
            "call_put": "call",
            "volume": 25000,
            "open_interest": 3000,
            "vol_oi_ratio": 8.33,
        },
    ]


# ---------------------------------------------------------------------------
# Step 2: Create the agent class.
# Inherit from HermesAgent and set the four required class attributes:
#   name          -- unique identifier (used for routing)
#   description   -- what the orchestrator sees when choosing agents
#   system_prompt -- injected into the LLM conversation for this agent
#   agent_type    -- "function" (tool-calling) or "react" (chain-of-thought)
# Then implement get_tools() to return your FunctionTool list.
# ---------------------------------------------------------------------------


class OptionsFlowAgent(HermesAgent):
    """Specialist agent for options flow analysis.

    Analyses options chains and detects unusual activity that may signal
    institutional positioning or event-driven trades.
    """

    name = "options_flow"
    description = (
        "Analyses options chains, detects unusual activity, and identifies "
        "potential institutional positioning based on volume/OI patterns."
    )
    system_prompt = (
        "You are an options flow analyst. Your job is to:\n"
        "1. Fetch options chain data for requested tickers.\n"
        "2. Identify unusual volume patterns that may signal institutional activity.\n"
        "3. Summarise put/call ratios, skew, and notable strikes.\n"
        "4. Always caveat that unusual flow is not predictive and may reflect "
        "hedging rather than directional bets.\n"
        "Be precise with numbers. Quote exact strikes, volumes, and ratios."
    )
    agent_type = "function"

    def get_tools(self) -> list[Any]:
        """Return FunctionTool instances wrapping our options helpers."""
        return [
            FunctionTool.from_defaults(
                async_fn=fetch_options_chain,
                name="fetch_options_chain",
                description=(
                    "Fetch the full options chain (calls and puts) for a ticker, "
                    "optionally filtered by expiry date."
                ),
            ),
            FunctionTool.from_defaults(
                async_fn=detect_unusual_activity,
                name="detect_unusual_activity",
                description=(
                    "Scan recent options activity for a ticker and flag contracts "
                    "with unusually high volume relative to open interest."
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# Step 3: Register the agent with a Hermes instance.
# After registration the orchestrator can delegate options-related
# sub-tasks to this agent automatically.
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate registering and listing a custom agent."""

    # Configure Hermes (minimal settings for this demo).
    configure(
        llm_provider="anthropic",
        sec_user_agent="HermesDemo demo@example.com",
        verbose=True,
    )

    h = Hermes()

    # Register our custom agent class.
    h.register_agent(
        name=OptionsFlowAgent.name,
        agent_cls=OptionsFlowAgent,
        description=OptionsFlowAgent.description,
    )

    # Verify it appears in the agent list.
    print("Registered agents:", h.list_agents())


if __name__ == "__main__":
    main()
