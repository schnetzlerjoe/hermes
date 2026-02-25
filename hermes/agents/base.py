"""Base class for all Hermes agents.

Provides the standard interface that the registry and orchestrator expect.
Custom agents should inherit from :class:`HermesAgent` and implement
:meth:`get_tools`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini schema compatibility helpers
# ---------------------------------------------------------------------------


def _strip_additional_properties(schema: dict) -> None:
    """Remove ``additionalProperties`` from a JSON schema dict in-place.

    Recursively processes all nested schemas.  The Gemini API rejects schemas
    that contain ``additionalProperties``, which Pydantic v2 emits for
    ``dict[str, Any]`` parameters (as ``additionalProperties: true``).

    Args:
        schema: A JSON Schema dict, modified in-place.
    """
    schema.pop("additionalProperties", None)
    for value in schema.values():
        if isinstance(value, dict):
            _strip_additional_properties(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strip_additional_properties(item)


def _is_google_llm(llm: Any) -> bool:
    """Return True if *llm* is a GoogleGenAI instance.

    Checked by class name to avoid a hard import of the optional package.
    """
    return type(llm).__name__ == "GoogleGenAI"


def _make_gemini_safe_schema(orig_mjs: Any) -> Any:
    """Wrap a ``model_json_schema`` classmethod to strip ``additionalProperties``.

    Args:
        orig_mjs: The original bound ``model_json_schema`` classmethod.

    Returns:
        A replacement classmethod that calls the original and then strips
        ``additionalProperties`` from the result.
    """

    @classmethod  # type: ignore[misc]
    def _safe_mjs(cls, **kwargs: Any) -> dict:
        schema = orig_mjs(**kwargs)
        _strip_additional_properties(schema)
        return schema

    return _safe_mjs


def patch_tools_for_google(tools: list[Any]) -> None:
    """Patch tool fn_schema classes to strip ``additionalProperties`` for Gemini.

    Pydantic v2 emits ``additionalProperties: true`` for ``dict[str, Any]``
    parameters.  The standard Gemini API rejects any schema containing this
    keyword (even nested within a property definition).  This patches each
    tool's schema class in-place so subsequent calls to ``model_json_schema()``
    return a Gemini-compatible schema.

    Args:
        tools: List of LlamaIndex tool instances to patch.
    """
    for tool in tools:
        try:
            schema_cls = tool.metadata.fn_schema
            schema_cls.model_json_schema = _make_gemini_safe_schema(
                schema_cls.model_json_schema
            )
        except (AttributeError, TypeError):
            pass


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

        if llm is not None and _is_google_llm(llm):
            logger.debug("Patching %d tool schemas for Gemini API compatibility", len(tools))
            patch_tools_for_google(tools)

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
