"""Tests for hermes.infra.retry — provider-aware 429 retry logic."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes.infra.retry import (
    RetryConfig,
    _parse_go_duration,
    extract_retry_after,
    is_rate_limit_error,
)


# ---------------------------------------------------------------------------
# _parse_go_duration
# ---------------------------------------------------------------------------


def test_parse_go_duration_combined():
    assert _parse_go_duration("6m2s") == pytest.approx(362.0)


def test_parse_go_duration_seconds_only():
    assert _parse_go_duration("53s") == pytest.approx(53.0)


def test_parse_go_duration_hours_minutes():
    assert _parse_go_duration("1h30m") == pytest.approx(5400.0)


def test_parse_go_duration_milliseconds():
    assert _parse_go_duration("500ms") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# is_rate_limit_error — Anthropic
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_anthropic_true():
    anthropic = pytest.importorskip("anthropic")
    exc = MagicMock(spec=anthropic.RateLimitError)
    assert is_rate_limit_error(exc, "anthropic") is True


def test_is_rate_limit_error_anthropic_other_exception():
    exc = ValueError("something else")
    assert is_rate_limit_error(exc, "anthropic") is False


# ---------------------------------------------------------------------------
# is_rate_limit_error — OpenAI
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_openai_true():
    openai = pytest.importorskip("openai")
    exc = MagicMock(spec=openai.RateLimitError)
    assert is_rate_limit_error(exc, "openai") is True


def test_is_rate_limit_error_openai_false():
    exc = RuntimeError("not a rate limit")
    assert is_rate_limit_error(exc, "openai") is False


def test_is_rate_limit_error_xai_same_as_openai():
    openai = pytest.importorskip("openai")
    exc = MagicMock(spec=openai.RateLimitError)
    assert is_rate_limit_error(exc, "xai") is True


# ---------------------------------------------------------------------------
# is_rate_limit_error — Google
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_google_client_error_429():
    google_genai = pytest.importorskip("google.genai.errors")
    exc = MagicMock(spec=google_genai.ClientError)
    exc.code = 429
    assert is_rate_limit_error(exc, "google") is True


def test_is_rate_limit_error_google_client_error_403():
    google_genai = pytest.importorskip("google.genai.errors")
    exc = MagicMock(spec=google_genai.ClientError)
    exc.code = 403
    assert is_rate_limit_error(exc, "google") is False


# ---------------------------------------------------------------------------
# extract_retry_after — Anthropic
# ---------------------------------------------------------------------------


def test_extract_retry_after_anthropic():
    exc = MagicMock()
    exc.response.headers = {"retry-after": "20"}
    result = extract_retry_after(exc, "anthropic")
    assert result == pytest.approx(20.0)


def test_extract_retry_after_anthropic_missing_header():
    exc = MagicMock()
    exc.response.headers = {}
    cfg = RetryConfig(base_backoff=30.0)
    result = extract_retry_after(exc, "anthropic", cfg)
    assert result == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# extract_retry_after — OpenAI / Go duration header
# ---------------------------------------------------------------------------


def test_extract_retry_after_openai_retry_after_float():
    exc = MagicMock()
    exc.response.headers = {"retry-after": "15"}
    result = extract_retry_after(exc, "openai")
    assert result == pytest.approx(15.0)


def test_extract_retry_after_openai_go_duration():
    exc = MagicMock()
    exc.response.headers = {"x-ratelimit-reset-requests": "6m2s"}
    result = extract_retry_after(exc, "openai")
    assert result == pytest.approx(362.0)


def test_extract_retry_after_openai_tokens_header():
    exc = MagicMock()
    exc.response.headers = {"x-ratelimit-reset-tokens": "53s"}
    result = extract_retry_after(exc, "openai")
    assert result == pytest.approx(53.0)


# ---------------------------------------------------------------------------
# extract_retry_after — Google JSON body
# ---------------------------------------------------------------------------


def test_extract_retry_after_google_body():
    exc = MagicMock()
    exc.response.json.return_value = {
        "error": {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "53s",
                }
            ]
        }
    }
    # No retry_delay attribute
    del exc.retry_delay
    result = extract_retry_after(exc, "google")
    assert result == pytest.approx(53.0)


def test_extract_retry_after_google_proto_fallback():
    exc = MagicMock()
    exc.response.json.return_value = {"error": {"details": []}}
    exc.retry_delay.seconds = 45
    exc.retry_delay.nanos = 0
    result = extract_retry_after(exc, "google")
    assert result == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# extract_retry_after — fallback
# ---------------------------------------------------------------------------


def test_extract_retry_after_fallback_unknown_provider():
    exc = ValueError("some error")
    cfg = RetryConfig(base_backoff=30.0)
    result = extract_retry_after(exc, "mistral", cfg)
    assert result == pytest.approx(30.0)


def test_extract_retry_after_fallback_no_headers():
    exc = MagicMock()
    exc.response = None  # causes AttributeError on .headers
    cfg = RetryConfig(base_backoff=30.0)
    result = extract_retry_after(exc, "anthropic", cfg)
    assert result == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Hermes.run() retry integration
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    """Minimal stand-in for a provider RateLimitError."""

    def __init__(self) -> None:
        super().__init__("rate limited")
        self.response = SimpleNamespace(headers={"retry-after": "1"})


@pytest.mark.asyncio
async def test_run_retries_on_429():
    """Hermes.run() should retry twice then return the successful result."""
    from hermes import Hermes

    call_count = 0

    async def _fake_handler():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _FakeRateLimitError()
        return "Apple stock is $200"

    fake_handler = _fake_handler()

    # We need the workflow.run() to return awaitable objects each call
    call_returns = [
        _fake_handler(),
        _fake_handler(),
        _fake_handler(),
    ]
    idx = 0

    def _next_handler(*args, **kwargs):
        nonlocal idx
        h = call_returns[idx]
        idx += 1
        return h

    with (
        patch("hermes.core.is_rate_limit_error", side_effect=lambda exc, p: isinstance(exc, _FakeRateLimitError)),
        patch("hermes.core.extract_retry_after", return_value=0.001),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("hermes.agents.orchestrator.ResearchOrchestrator") as MockOrch,
        patch.object(Hermes, "_ensure_initialized"),
        patch.object(Hermes, "_get_llm", return_value=MagicMock()),
    ):
        mock_workflow = MagicMock()
        mock_workflow.run.side_effect = _next_handler
        MockOrch.return_value.build_workflow.return_value = mock_workflow

        h = Hermes()
        h._config = h._config.model_copy(update={"llm_max_retries": 3})
        result = await h.run("What is Apple's stock price?")

    assert "Apple stock is $200" in result["response"]
    assert mock_sleep.call_count == 2
