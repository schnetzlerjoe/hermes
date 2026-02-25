"""Streaming event types for real-time progress reporting.

Agents and tools emit :class:`StreamEvent` instances so that callers (CLIs,
web UIs, logging pipelines) can observe workflow progress without polling.
Each event carries an :class:`EventType` discriminator plus optional fields
describing the context.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Discriminator for stream events."""

    AGENT_START = "agent_start"
    AGENT_OUTPUT = "agent_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOKEN = "token"
    FILE_CREATED = "file_created"
    ERROR = "error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"


class StreamEvent(BaseModel):
    """A single event emitted during agent execution.

    Attributes
    ----------
    type:
        The kind of event.
    agent_name:
        Name of the agent that produced the event (if applicable).
    tool_name:
        Name of the tool involved (for ``TOOL_CALL`` / ``TOOL_RESULT``).
    text:
        Free-form textual payload (agent output, error messages, etc.).
    file_path:
        Path to a file that was created or referenced.
    metadata:
        Arbitrary extra data consumers may find useful.
    timestamp:
        Unix epoch seconds when the event was created.
    """

    type: EventType
    agent_name: str | None = None
    tool_name: str | None = None
    text: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def agent_start(name: str) -> StreamEvent:
    """Create an ``AGENT_START`` event."""
    return StreamEvent(type=EventType.AGENT_START, agent_name=name)


def agent_output(name: str, text: str) -> StreamEvent:
    """Create an ``AGENT_OUTPUT`` event carrying *text*."""
    return StreamEvent(type=EventType.AGENT_OUTPUT, agent_name=name, text=text)


def tool_call(agent: str, tool: str) -> StreamEvent:
    """Create a ``TOOL_CALL`` event for *tool* invoked by *agent*."""
    return StreamEvent(type=EventType.TOOL_CALL, agent_name=agent, tool_name=tool)


def file_created(path: str, agent: str | None = None) -> StreamEvent:
    """Create a ``FILE_CREATED`` event for *path*."""
    return StreamEvent(type=EventType.FILE_CREATED, file_path=path, agent_name=agent)


def error(message: str, agent: str | None = None) -> StreamEvent:
    """Create an ``ERROR`` event with *message*."""
    return StreamEvent(type=EventType.ERROR, text=message, agent_name=agent)
