"""Tests for HermesAgent base class helpers.

Focuses on schema patching for Gemini API compatibility.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out llama_index if not installed
# ---------------------------------------------------------------------------
if "llama_index" not in sys.modules:
    _li = types.ModuleType("llama_index")
    _li_core = types.ModuleType("llama_index.core")
    _li_tools = types.ModuleType("llama_index.core.tools")
    _li_agent = types.ModuleType("llama_index.core.agent")

    class _FakeFunctionTool:
        @classmethod
        def from_defaults(cls, **kwargs):
            return cls()

    class _FakeAgent:
        def __init__(self, **kwargs):
            pass

    _li_tools.FunctionTool = _FakeFunctionTool  # type: ignore[attr-defined]
    _li_core.tools = _li_tools  # type: ignore[attr-defined]
    _li_agent.FunctionAgent = _FakeAgent  # type: ignore[attr-defined]
    _li_agent.ReActAgent = _FakeAgent  # type: ignore[attr-defined]
    _li_core.agent = _li_agent  # type: ignore[attr-defined]
    _li.core = _li_core  # type: ignore[attr-defined]
    sys.modules["llama_index"] = _li
    sys.modules["llama_index.core"] = _li_core
    sys.modules["llama_index.core.tools"] = _li_tools
    sys.modules["llama_index.core.agent"] = _li_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_with_schema(schema: dict) -> MagicMock:
    """Create a mock tool whose fn_schema returns the given schema dict."""

    class FakeSchema:
        @classmethod
        def model_json_schema(cls, **kwargs: Any) -> dict:
            import copy

            return copy.deepcopy(schema)

    tool = MagicMock()
    tool.metadata.fn_schema = FakeSchema
    return tool


def _make_google_llm() -> MagicMock:
    """Return a mock LLM whose class is named GoogleGenAI."""
    llm = MagicMock()
    llm.__class__.__name__ = "GoogleGenAI"
    return llm


# ---------------------------------------------------------------------------
# Tests: _strip_additional_properties
# ---------------------------------------------------------------------------


class TestStripAdditionalProperties:
    """Unit tests for the recursive schema stripping helper."""

    def test_removes_top_level(self) -> None:
        from hermes.agents.base import _strip_additional_properties

        schema = {"type": "object", "additionalProperties": True}
        _strip_additional_properties(schema)
        assert "additionalProperties" not in schema

    def test_removes_nested_in_property(self) -> None:
        from hermes.agents.base import _strip_additional_properties

        schema = {
            "type": "object",
            "properties": {
                "cell_data": {"type": "object", "additionalProperties": True}
            },
        }
        _strip_additional_properties(schema)
        assert "additionalProperties" not in schema["properties"]["cell_data"]

    def test_removes_inside_list(self) -> None:
        from hermes.agents.base import _strip_additional_properties

        schema = {
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "null"},
            ]
        }
        _strip_additional_properties(schema)
        assert "additionalProperties" not in schema["anyOf"][0]

    def test_no_op_when_absent(self) -> None:
        from hermes.agents.base import _strip_additional_properties

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        original = schema.copy()
        _strip_additional_properties(schema)
        assert schema == original

    def test_removes_false_value_too(self) -> None:
        """additionalProperties: false should also be removed."""
        from hermes.agents.base import _strip_additional_properties

        schema = {"type": "object", "additionalProperties": False}
        _strip_additional_properties(schema)
        assert "additionalProperties" not in schema


# ---------------------------------------------------------------------------
# Tests: _is_google_llm
# ---------------------------------------------------------------------------


class TestIsGoogleLlm:
    def test_true_for_google_genai(self) -> None:
        from hermes.agents.base import _is_google_llm

        llm = _make_google_llm()
        assert _is_google_llm(llm) is True

    def test_false_for_other_providers(self) -> None:
        from hermes.agents.base import _is_google_llm

        for name in ("Anthropic", "OpenAI", "MistralAI", "Groq"):
            llm = MagicMock()
            llm.__class__.__name__ = name
            assert _is_google_llm(llm) is False


# ---------------------------------------------------------------------------
# Tests: patch_tools_for_google
# ---------------------------------------------------------------------------


class TestPatchToolsForGoogle:
    def test_strips_additional_properties_from_schema(self) -> None:
        from hermes.agents.base import patch_tools_for_google

        schema = {
            "type": "object",
            "properties": {
                "cell_data": {"type": "object", "additionalProperties": True}
            },
        }
        tool = _make_tool_with_schema(schema)
        patch_tools_for_google([tool])

        result = tool.metadata.fn_schema.model_json_schema()
        assert "additionalProperties" not in result["properties"]["cell_data"]

    def test_original_schema_still_has_key(self) -> None:
        """Patching should not mutate the schema dict returned before patching."""
        from hermes.agents.base import patch_tools_for_google

        schema = {"type": "object", "additionalProperties": True}
        tool = _make_tool_with_schema(schema)

        # Get schema before patching
        before = tool.metadata.fn_schema.model_json_schema()
        assert "additionalProperties" in before

        patch_tools_for_google([tool])

        after = tool.metadata.fn_schema.model_json_schema()
        assert "additionalProperties" not in after

    def test_handles_tool_without_fn_schema(self) -> None:
        """Should not raise if a tool has no fn_schema."""
        from hermes.agents.base import patch_tools_for_google

        tool = MagicMock()
        del tool.metadata.fn_schema  # no fn_schema attribute

        patch_tools_for_google([tool])  # should not raise

    def test_patches_multiple_tools(self) -> None:
        from hermes.agents.base import patch_tools_for_google

        tools = [
            _make_tool_with_schema({"type": "object", "additionalProperties": True}),
            _make_tool_with_schema(
                {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "additionalProperties": True}
                    },
                }
            ),
        ]
        patch_tools_for_google(tools)

        assert "additionalProperties" not in tools[0].metadata.fn_schema.model_json_schema()
        assert (
            "additionalProperties"
            not in tools[1].metadata.fn_schema.model_json_schema()["properties"]["data"]
        )


# ---------------------------------------------------------------------------
# Tests: HermesAgent.build() patches tools for Google
# ---------------------------------------------------------------------------


class TestHermesAgentBuildGeminiPatching:
    """Verify that build() applies schema patching when LLM is GoogleGenAI."""

    def test_build_patches_tools_for_google(self) -> None:
        from hermes.agents.base import HermesAgent

        schema = {"type": "object", "additionalProperties": True}
        tool = _make_tool_with_schema(schema)

        class FakeAgent(HermesAgent):
            name = "fake"
            description = "fake"
            system_prompt = "fake"
            agent_type = "function"

            def get_tools(self):
                return [tool]

        agent = FakeAgent()
        llm = _make_google_llm()

        agent.build(llm=llm)

        # After build(), the tool schema should be sanitized
        result = tool.metadata.fn_schema.model_json_schema()
        assert "additionalProperties" not in result

    def test_build_does_not_patch_for_non_google(self) -> None:
        """Schema should not be modified when using non-Gemini providers."""
        from hermes.agents.base import HermesAgent

        schema = {"type": "object", "additionalProperties": True}
        tool = _make_tool_with_schema(schema)

        class FakeAgent(HermesAgent):
            name = "fake"
            description = "fake"
            system_prompt = "fake"
            agent_type = "function"

            def get_tools(self):
                return [tool]

        agent = FakeAgent()
        anthropic_llm = MagicMock()
        anthropic_llm.__class__.__name__ = "Anthropic"

        agent.build(llm=anthropic_llm)

        # Schema should still have additionalProperties
        result = tool.metadata.fn_schema.model_json_schema()
        assert "additionalProperties" in result
