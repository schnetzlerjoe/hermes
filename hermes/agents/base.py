"""Base class for all Hermes agents.

Provides the standard interface that the registry and orchestrator expect.
Custom agents should inherit from :class:`HermesAgent` and implement
:meth:`get_tools`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HermesAgent(ABC):
    """Base class for Hermes specialist agents.

    Subclasses must define class-level attributes and implement
    :meth:`get_tools`.  The orchestrator uses ``name`` and ``description``
    for routing decisions -- both should be concise but descriptive enough
    for an LLM to decide whether to delegate a sub-task to this agent.

    Class Attributes:
        name: Machine-readable identifier used for agent routing and
            registration.  Must be unique across all agents.
        description: Natural-language summary of the agent's capabilities.
            Used by the orchestrator when deciding which specialist to invoke.
        system_prompt: The full system prompt injected into the underlying
            LLM conversation.  Should contain domain-specific instructions,
            output format guidance, and guardrails.
        agent_type: Either ``"function"`` (tool-calling) or ``"react"``
            (chain-of-thought with tool use).  Data-retrieval agents typically
            use ``"function"``; multi-step reasoning agents use ``"react"``.
    """

    # Subclasses must set these
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    agent_type: str = "function"  # "function" or "react"
    can_handoff_to: list[str] | None = None  # None = unrestricted

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """Return the LlamaIndex tools this agent has access to.

        Returns:
            A list of :class:`FunctionTool`, :class:`QueryEngineTool`, or
            other LlamaIndex-compatible tool instances.  The orchestrator
            passes these to the underlying agent framework at build time.
        """
        ...

    def get_query_engines(self) -> list[Any]:
        """Return query engines for RAG-capable agents.

        Override this in agents that need vector store retrieval
        (e.g., :class:`SecFilingsAgent` for Q&A over indexed filings).
        Defaults to an empty list.

        Returns:
            A list of LlamaIndex query engine instances.
        """
        return []

    def build(self, llm: Any | None = None) -> Any:
        """Construct the underlying LlamaIndex agent.

        Uses :class:`FunctionAgent` for ``agent_type="function"`` and
        :class:`ReActAgent` for ``agent_type="react"``.  Accepts an optional
        LLM override; otherwise the agent framework uses its configured
        default.

        Args:
            llm: An optional LLM instance to override the library default.
                Must be a LlamaIndex-compatible LLM (e.g.,
                ``llama_index.llms.anthropic.Anthropic``).

        Returns:
            A constructed LlamaIndex agent instance ready for use in a
            workflow.

        Raises:
            ValueError: If ``agent_type`` is not ``"function"`` or
                ``"react"``.
        """
        from llama_index.core.agent import FunctionAgent, ReActAgent

        if self.agent_type not in ("function", "react"):
            raise ValueError(
                f"Invalid agent_type {self.agent_type!r} for {self.name!r}. "
                "Must be 'function' or 'react'."
            )

        tools = self.get_tools()
        logger.debug("Agent %r get_tools() returned %d tools", self.name, len(tools))

        agent_cls = FunctionAgent if self.agent_type == "function" else ReActAgent

        kwargs: dict[str, Any] = {
            "tools": tools,
            "system_prompt": self.system_prompt,
            "name": self.name,
            "description": self.description,
        }
        if self.can_handoff_to is not None:
            kwargs["can_handoff_to"] = self.can_handoff_to
        if llm is not None:
            kwargs["llm"] = llm

        logger.debug(
            "Building %s agent %r with %d tools",
            self.agent_type,
            self.name,
            len(tools),
        )

        return agent_cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"type={self.agent_type!r})"
        )
