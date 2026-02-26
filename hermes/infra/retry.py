"""Provider-aware 429 retry logic for LLM API calls.

Provides :class:`RetryConfig`, :func:`is_rate_limit_error`, and
:func:`extract_retry_after` — the three primitives used by :class:`Hermes`
to handle transient rate-limit errors from any supported LLM provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    """Immutable configuration for LLM retry behaviour.

    Args:
        max_retries: Maximum number of retry attempts after the first failure.
        max_wait: Cap on the server-suggested wait time in seconds.
        base_backoff: Fallback wait time (seconds) when no header is present.
    """

    max_retries: int = 3
    max_wait: float = 120.0
    base_backoff: float = 30.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_go_duration(s: str) -> float:
    """Parse a Go-style duration string into seconds.

    Examples: ``"6m2s"`` → 362.0, ``"53s"`` → 53.0, ``"1h30m"`` → 5400.0.

    Args:
        s: Go duration string (e.g. ``"6m2s"``).

    Returns:
        Total duration in seconds as a float.
    """
    total = 0.0
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)([a-z]+)", s):
        v = float(val)
        match unit:
            case "h":
                total += v * 3600
            case "m":
                total += v * 60
            case "s":
                total += v
            case "ms":
                total += v / 1000
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_rate_limit_error(exc: BaseException, provider: str) -> bool:
    """Return True if *exc* is a rate-limit error from *provider*.

    Imports for each SDK are done inside try/except blocks so uninstalled
    provider SDKs do not break other providers.

    Args:
        exc: The exception to inspect.
        provider: LLM provider name (e.g. ``"anthropic"``, ``"openai"``).

    Returns:
        True if the exception represents an HTTP 429 rate-limit error.
    """
    if provider == "anthropic":
        try:
            import anthropic

            return isinstance(exc, anthropic.RateLimitError)
        except ImportError:
            return False

    if provider in ("openai", "xai", "deepseek"):
        try:
            import openai

            return isinstance(exc, openai.RateLimitError)
        except ImportError:
            return False

    if provider == "google":
        try:
            import google.genai.errors  # type: ignore[import-untyped]

            code = getattr(exc, "code", None)
            if isinstance(exc, google.genai.errors.ClientError) and code == 429:
                return True
        except ImportError:
            pass
        try:
            import google.api_core.exceptions  # type: ignore[import-untyped]

            if isinstance(exc, google.api_core.exceptions.ResourceExhausted):
                return True
        except ImportError:
            pass
        return False

    return False


def extract_retry_after(
    exc: BaseException, provider: str, config: RetryConfig | None = None
) -> float:
    """Return how many seconds to wait before retrying after a 429 error.

    Inspects provider-specific response headers and body fields to find the
    authoritative retry delay.  Falls back to ``config.base_backoff`` (30 s)
    when no value can be extracted.

    Args:
        exc: The rate-limit exception.
        provider: LLM provider name.
        config: Retry configuration (uses defaults if None).

    Returns:
        Number of seconds to wait (not capped by ``max_wait`` — callers apply
        the cap themselves).
    """
    cfg = config if config is not None else RetryConfig()

    if provider == "anthropic":
        try:
            headers = exc.response.headers  # type: ignore[union-attr]
            val = headers.get("retry-after")
            if val is not None:
                return float(val)
        except (AttributeError, TypeError, ValueError):
            pass

    elif provider in ("openai", "xai", "deepseek"):
        try:
            headers = exc.response.headers  # type: ignore[union-attr]
            # Try plain seconds first
            val = headers.get("retry-after")
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    return _parse_go_duration(val)
            # Try Go duration headers
            for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
                val = headers.get(header)
                if val:
                    return _parse_go_duration(val)
        except (AttributeError, TypeError):
            pass

    elif provider == "google":
        # Try JSON body RetryInfo
        try:
            body = exc.response.json()  # type: ignore[union-attr]
            details = body.get("error", {}).get("details", [])
            for detail in details:
                type_url = detail.get("@type", "")
                if type_url.endswith("RetryInfo"):
                    delay_str = detail.get("retryDelay", "")
                    if delay_str:
                        return _parse_go_duration(delay_str)
        except (AttributeError, TypeError, ValueError, KeyError):
            pass
        # Try proto Duration attribute
        try:
            retry_delay = exc.retry_delay  # type: ignore[union-attr]
            seconds = getattr(retry_delay, "seconds", 0) or 0
            nanos = getattr(retry_delay, "nanos", 0) or 0
            total = seconds + nanos / 1e9
            if total > 0:
                return total
        except (AttributeError, TypeError):
            pass

    return cfg.base_backoff
