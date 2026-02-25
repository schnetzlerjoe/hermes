"""Central registry for tools and agents.

Supports registration, lookup, tagging, and override.  Every :class:`Hermes`
instance owns a :class:`Registry` that maps human-readable names to tool
entries (wrapping LlamaIndex ``FunctionTool`` objects) and agent class entries.

Tools can be tagged with arbitrary strings (e.g. ``"sec"``, ``"macro"``) so
that the orchestrator can filter by capability at planning time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolEntry:
    """A registered tool and its metadata."""

    name: str
    tool: Any  # llama_index.core.tools.FunctionTool at runtime
    tags: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(slots=True)
class AgentEntry:
    """A registered agent class and its metadata."""

    name: str
    agent_cls: type
    description: str = ""


class Registry:
    """In-memory store for tools and agents.

    The registry is intentionally *not* a global singleton -- each
    :class:`~hermes.core.Hermes` instance carries its own so that tests and
    multiple concurrent configurations stay isolated.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._agents: dict[str, AgentEntry] = {}

    # -- Tool operations ------------------------------------------------------

    def register_tool(
        self,
        name: str,
        tool: Any,
        tags: list[str] | None = None,
        description: str = "",
        *,
        override: bool = False,
    ) -> None:
        """Register a tool under *name*.

        Args:
            name: Unique identifier for the tool.
            tool: A ``FunctionTool`` (or compatible) instance.
            tags: Optional tags for capability-based lookup.
            description: Human-readable summary of what the tool does.
            override: If ``True``, silently replace an existing entry with the
                same *name*.  Otherwise raise :class:`KeyError`.
        """
        if name in self._tools and not override:
            raise KeyError(
                f"Tool '{name}' is already registered. "
                "Pass override=True to replace it."
            )
        self._tools[name] = ToolEntry(
            name=name,
            tool=tool,
            tags=tags or [],
            description=description,
        )
        logger.debug("Registered tool %r (tags=%s)", name, tags or [])

    def get_tool(self, name: str) -> ToolEntry:
        """Return the :class:`ToolEntry` for *name* or raise :class:`KeyError`."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"No tool registered with name '{name}'.") from None

    def find_tools_by_tag(self, tag: str) -> list[ToolEntry]:
        """Return every tool whose tags include *tag*."""
        return [entry for entry in self._tools.values() if tag in entry.tags]

    def list_tools(self) -> dict[str, list[str]]:
        """Return a mapping of tool names to their tags."""
        return {name: entry.tags for name, entry in self._tools.items()}

    def remove_tool(self, name: str) -> None:
        """Remove a tool by *name*.  Raises :class:`KeyError` if not found."""
        try:
            del self._tools[name]
        except KeyError:
            raise KeyError(f"No tool registered with name '{name}'.") from None

    # -- Agent operations -----------------------------------------------------

    def register_agent(
        self,
        name: str,
        agent_cls: type,
        description: str = "",
        *,
        override: bool = False,
    ) -> None:
        """Register an agent class under *name*.

        Args:
            name: Unique identifier for the agent.
            agent_cls: The agent class (not an instance).
            description: Human-readable summary of the agent's role.
            override: If ``True``, silently replace an existing entry.
        """
        if name in self._agents and not override:
            raise KeyError(
                f"Agent '{name}' is already registered. "
                "Pass override=True to replace it."
            )
        self._agents[name] = AgentEntry(
            name=name,
            agent_cls=agent_cls,
            description=description,
        )
        logger.debug("Registered agent %r", name)

    def get_agent(self, name: str) -> AgentEntry:
        """Return the :class:`AgentEntry` for *name* or raise :class:`KeyError`."""
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(f"No agent registered with name '{name}'.") from None

    def list_agents(self) -> list[str]:
        """Return a list of registered agent names."""
        return list(self._agents.keys())

    def remove_agent(self, name: str) -> None:
        """Remove an agent by *name*.  Raises :class:`KeyError` if not found."""
        try:
            del self._agents[name]
        except KeyError:
            raise KeyError(f"No agent registered with name '{name}'.") from None

    # -- Utilities ------------------------------------------------------------

    def clear(self) -> None:
        """Remove all registered tools and agents.  Intended for testing."""
        self._tools.clear()
        self._agents.clear()

    def __repr__(self) -> str:
        return (
            f"Registry(tools={len(self._tools)}, agents={len(self._agents)})"
        )
