"""Main entry point for the Hermes library.

The :class:`Hermes` facade provides a single object through which callers
interact with the entire framework -- registering tools and agents, launching
queries, and streaming results.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from hermes.config import HermesConfig, configure, get_config
from hermes.infra.streaming import EventType, StreamEvent
from hermes.llm_providers import build_llm, detect_provider
from hermes.registry import Registry

logger = logging.getLogger(__name__)


def _extract_text(value: Any) -> str | None:
    """Return plain text from a value that may be a str, ChatMessage, ToolOutput, etc.

    LlamaIndex wraps LLM responses in ChatMessage objects and tool results in
    ToolOutput objects.  Both carry the actual text in a ``.content`` attribute.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # ChatMessage / ToolOutput and similar objects expose .content
    content = getattr(value, "content", None)
    if content is not None:
        return str(content)
    # Some objects use .text
    text_attr = getattr(value, "text", None)
    if text_attr is not None:
        return str(text_attr)
    return str(value)


class Hermes:
    """High-level facade for the Hermes multi-agent financial research framework.

    Usage::

        from hermes import Hermes, configure

        configure(sec_user_agent="MyApp admin@x.com")
        h = Hermes()
        result = h.invoke("Summarise Apple's latest 10-K filing.")

    Power-user constructor::

        h = Hermes(model="gpt-4o", tools=[my_tool], agents=[MyAgent])
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        tools: list[Any] | None = None,
        agents: list[type] | None = None,
        config: HermesConfig | None = None,
        verbose: bool = False,
        **config_kwargs: Any,
    ) -> None:
        # Build or adopt config
        if config is not None:
            self._config = config
        elif config_kwargs or model or provider or verbose:
            overrides: dict[str, Any] = {**config_kwargs}
            if verbose:
                overrides["verbose"] = True
            if model:
                overrides["llm_model"] = model
            if provider:
                overrides["llm_provider"] = provider
            elif model:
                overrides["llm_provider"] = detect_provider(model)
            self._config = configure(**overrides)
        else:
            self._config = get_config()

        self._registry: Registry = Registry()
        self._initialized: bool = False
        self._extra_tools: list[Any] = tools or []
        self._extra_agents: list[type] = agents or []

    # -- Public API -----------------------------------------------------------

    def register_tool(
        self,
        name: str,
        tool: Any,
        tags: list[str] | None = None,
        description: str = "",
    ) -> None:
        """Register a custom tool.

        Args:
            name: Unique identifier.
            tool: A ``FunctionTool`` (or compatible) instance.
            tags: Optional tags for capability-based lookup.
            description: Human-readable summary.
        """
        self._registry.register_tool(name, tool, tags=tags, description=description)

    def register_agent(
        self,
        name: str,
        agent_cls: type,
        description: str = "",
        *,
        override: bool = False,
    ) -> None:
        """Register a custom agent class.

        Args:
            name: Unique identifier.
            agent_cls: The agent class to register.
            description: Human-readable summary.
            override: Replace an existing agent with the same name.
        """
        self._registry.register_agent(
            name, agent_cls, description=description, override=override
        )

    def list_tools(self) -> dict[str, list[str]]:
        """Return registered tools as ``{name: [tags, ...]}``."""
        self._ensure_initialized()
        return self._registry.list_tools()

    def list_agents(self) -> list[str]:
        """Return the names of all registered agents."""
        self._ensure_initialized()
        return self._registry.list_agents()

    async def run(
        self,
        query: str,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a research query through the orchestrator.

        Args:
            query: Natural-language research question.
            max_iterations: Maximum agent steps before stopping
                (default 100, LlamaIndex default is 20).
            **kwargs: Forwarded to the orchestrator (e.g. ``ticker``,
                ``filing_type``).

        Returns:
            A dict containing at minimum ``{"response": str}``.
        """
        logger.info("Running query: %s", query[:120])
        self._ensure_initialized()
        llm = self._get_llm()

        from hermes.agents.orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator()
        workflow = orchestrator.build_workflow(llm=llm)
        handler = workflow.run(
            user_msg=query, max_iterations=max_iterations, **kwargs
        )
        result = await handler

        logger.info("Query complete")
        return {"response": str(result)}

    async def stream(
        self,
        query: str,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream execution events for a research query.

        Yields :class:`StreamEvent` instances as the orchestrator delegates
        work to sub-agents and tools.

        Args:
            query: Natural-language research question.
            max_iterations: Maximum agent steps before stopping
                (default 100, LlamaIndex default is 20).
            **kwargs: Forwarded to the orchestrator.
        """
        self._ensure_initialized()
        llm = self._get_llm()

        from hermes.agents.orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator()
        workflow = orchestrator.build_workflow(llm=llm)
        handler = workflow.run(
            user_msg=query, max_iterations=max_iterations, **kwargs
        )

        yield StreamEvent(type=EventType.WORKFLOW_START, text=query)

        async for ev in handler.stream_events():
            ev_type = type(ev).__name__
            if ev_type == "AgentStream":
                delta = getattr(ev, "delta", "")
                yield StreamEvent(
                    type=EventType.TOKEN,
                    agent_name=getattr(ev, "agent_name", None),
                    text=_extract_text(delta) or "",
                )
            elif ev_type == "ToolCall":
                yield StreamEvent(
                    type=EventType.TOOL_CALL,
                    agent_name=getattr(ev, "agent_name", None),
                    tool_name=getattr(ev, "tool_name", None),
                )
            elif ev_type == "ToolCallResult":
                raw_output = getattr(ev, "tool_output", None)
                yield StreamEvent(
                    type=EventType.TOOL_RESULT,
                    agent_name=getattr(ev, "agent_name", None),
                    tool_name=getattr(ev, "tool_name", None),
                    text=_extract_text(raw_output),
                )
            elif ev_type == "AgentOutput":
                raw_response = getattr(ev, "response", None)
                yield StreamEvent(
                    type=EventType.AGENT_OUTPUT,
                    agent_name=getattr(ev, "agent_name", None),
                    text=_extract_text(raw_response),
                )

        final_result = await handler
        yield StreamEvent(
            type=EventType.WORKFLOW_COMPLETE,
            text=_extract_text(final_result) or "",
        )

    def invoke(
        self,
        query: str,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous convenience wrapper around :meth:`run`.

        Args:
            query: Natural-language research question.
            max_iterations: Maximum agent steps before stopping
                (default 100, LlamaIndex default is 20).
            **kwargs: Forwarded to :meth:`run`.

        Returns:
            A dict containing at minimum ``{"response": str}``.
        """
        return asyncio.run(self.run(query, max_iterations=max_iterations, **kwargs))

    # -- LLM instantiation ----------------------------------------------------

    def _get_llm(self) -> Any:
        """Instantiate a LlamaIndex LLM from the current config."""
        return build_llm(self._config.llm_provider, self._config.llm_model, self._config)

    # -- Lazy initialisation --------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Perform one-time setup: register built-in tools and agents."""
        if self._initialized:
            return
        self._register_builtin_tools()
        logger.debug("Registered %d built-in tools", len(self._registry.list_tools()))
        self._register_builtin_agents()
        logger.debug("Registered %d built-in agents", len(self._registry.list_agents()))

        # Register user-provided extras
        for tool in self._extra_tools:
            name = getattr(tool, "metadata", None)
            if name is not None:
                name = getattr(name, "name", None)
            if name is None:
                name = getattr(tool, "name", f"custom_tool_{id(tool)}")
            self._registry.register_tool(
                name, tool, tags=["custom"], description="User-provided tool"
            )

        for agent_cls in self._extra_agents:
            agent_name = getattr(agent_cls, "name", agent_cls.__name__)
            desc = getattr(agent_cls, "description", "")
            self._registry.register_agent(agent_name, agent_cls, description=desc)

        self._initialized = True
        if self._config.verbose:
            logger.info(
                "Hermes initialised with %d tools and %d agents",
                len(self._registry.list_tools()),
                len(self._registry.list_agents()),
            )

    def _register_builtin_tools(self) -> None:
        """Import and register every tool shipped with hermes."""
        from hermes.tools import (
            charts,
            documents,
            excel,
            fred,
            market_data,
            news,
            sec_edgar,
        )

        tag_map: dict[str, list[str]] = {
            "sec_edgar": ["sec", "data"],
            "fred": ["macro", "data"],
            "market_data": ["market", "data"],
            "news": ["news", "data"],
            "excel": ["output", "excel"],
            "documents": ["output", "documents"],
            "charts": ["output", "charts"],
        }

        for module_name, module in [
            ("sec_edgar", sec_edgar),
            ("fred", fred),
            ("market_data", market_data),
            ("news", news),
            ("excel", excel),
            ("documents", documents),
            ("charts", charts),
        ]:
            tags = tag_map.get(module_name, [])
            for tool in module.create_tools():
                self._registry.register_tool(
                    tool.metadata.name,
                    tool,
                    tags=tags,
                    description=tool.metadata.description,
                )

    def _register_builtin_agents(self) -> None:
        """Import and register every agent shipped with hermes."""
        from hermes.agents.macro import MacroAgent
        from hermes.agents.market import MarketDataAgent
        from hermes.agents.modeling import ModelingAgent
        from hermes.agents.news import NewsAgent
        from hermes.agents.orchestrator import ResearchOrchestrator
        from hermes.agents.report import ReportAgent
        from hermes.agents.sec_filings import SecFilingsAgent

        for cls in [
            SecFilingsAgent,
            MacroAgent,
            MarketDataAgent,
            NewsAgent,
            ModelingAgent,
            ReportAgent,
            ResearchOrchestrator,
        ]:
            self._registry.register_agent(
                cls.name, cls, description=cls.description
            )

    # -- Dunder helpers -------------------------------------------------------

    def __repr__(self) -> str:
        provider = self._config.llm_provider
        model = self._config.llm_model
        n_tools = len(self._registry.list_tools())
        n_agents = len(self._registry.list_agents())
        return (
            f"Hermes(provider={provider!r}, model={model!r}, "
            f"tools={n_tools}, agents={n_agents})"
        )


