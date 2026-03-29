"""LLM provider registry and factory for multi-provider support.

This module centralises the knowledge of how to instantiate LlamaIndex LLM
classes for every supported provider.  The :func:`build_llm` factory is the
single entry point used by :class:`hermes.core.Hermes`.

Providers that require extra packages (beyond the core anthropic/openai) are
imported lazily so that users only need to install what they use.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# LlamaIndex maps max_tokens → max_completion_tokens only for these model ids (exact match).
try:
    from llama_index.llms.openai.utils import O1_MODELS as _LI_O1_MODELS

    _OPENAI_REASONING_MODEL_IDS: frozenset[str] = frozenset(_LI_O1_MODELS.keys())
except ImportError:  # pragma: no cover - llama-index is a core dependency
    _OPENAI_REASONING_MODEL_IDS = frozenset()


def _openai_model_needs_canonical_id(model: str) -> bool:
    """Return True if *model* should be lowercased for LlamaIndex OpenAI token handling.

    OpenAI's chat completions API expects ``max_completion_tokens`` for reasoning models
    instead of ``max_tokens``. LlamaIndex's OpenAI LLM maps ``max_tokens`` only when
    ``self.model`` is an exact key in ``O1_MODELS`` (lowercase). Non-canonical casing
    (e.g. ``O3-mini``) skips that branch and the API may reject ``max_tokens``.

    Args:
        model: Raw model string from configuration.

    Returns:
        Whether the identifier should be normalized with :func:`_canonical_openai_model_id`.
    """
    ml = model.strip().lower()
    if ml in _OPENAI_REASONING_MODEL_IDS:
        return True
    # Forward-compatible: new reasoning-style ids not yet in the installed LlamaIndex.
    return ml.startswith(("o1", "o3", "o4", "gpt-5"))


def _canonical_openai_model_id(model: str) -> str:
    """Return lowercase model id when it is a known reasoning / O1_MODELS OpenAI model."""
    return model.strip().lower()


def _resolve_llm_model_id(spec: ProviderSpec, model: str) -> str:
    """Resolve the model string passed to the LlamaIndex LLM constructor.

    OpenAI-compatible classes that inherit ``llama_index.llms.openai.base.OpenAI`` rely on
    exact ``O1_MODELS`` keys for ``max_tokens`` → ``max_completion_tokens`` conversion.

    Args:
        spec: Provider specification.
        model: User-configured model name.

    Returns:
        Model id to pass to the LLM class (``model`` or ``model_name``, etc.).
    """
    if spec.class_name == "OpenAI" and spec.import_module == "llama_index.llms.openai":
        if _openai_model_needs_canonical_id(model):
            return _canonical_openai_model_id(model)
        return model.strip()
    # Groq subclasses OpenAILike → OpenAI; same ``model in O1_MODELS`` behaviour.
    if spec.class_name == "Groq":
        if _openai_model_needs_canonical_id(model):
            return _canonical_openai_model_id(model)
        return model.strip()
    return model.strip()


@dataclass(frozen=True)
class ProviderSpec:
    """Describes how to locate and instantiate a LlamaIndex LLM class.

    Attributes:
        name: Short provider identifier (e.g. ``"anthropic"``).
        import_module: Fully-qualified Python module containing the LLM class.
        class_name: Name of the LLM class inside *import_module*.
        model_kwarg: Keyword argument name for the model identifier
            (usually ``"model"``; HuggingFace uses ``"model_name"``).
        api_key_config_field: Name of the :class:`HermesConfig` field that
            holds an API key for this provider, or *None* if the provider's
            SDK reads its own env var.
        extra_kwargs: Additional keyword arguments passed to the LLM
            constructor (e.g. ``api_base`` for OpenAI-compatible proxies).
        detection_prefixes: Model-name prefixes used by :func:`detect_provider`
            to auto-select this provider.
        package: pip-installable package name shown in error messages.
    """

    name: str
    import_module: str
    class_name: str
    model_kwarg: str = "model"
    api_key_config_field: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    detection_prefixes: tuple[str, ...] = ()
    package: str = ""


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        import_module="llama_index.llms.anthropic",
        class_name="Anthropic",
        detection_prefixes=("claude",),
        package="llama-index-llms-anthropic",
    ),
    "openai": ProviderSpec(
        name="openai",
        import_module="llama_index.llms.openai",
        class_name="OpenAI",
        detection_prefixes=("gpt", "o1", "o3", "o4"),
        package="llama-index-llms-openai",
    ),
    "google": ProviderSpec(
        name="google",
        import_module="llama_index.llms.google_genai",
        class_name="GoogleGenAI",
        detection_prefixes=("gemini",),
        package="llama-index-llms-google-genai",
    ),
    "mistral": ProviderSpec(
        name="mistral",
        import_module="llama_index.llms.mistralai",
        class_name="MistralAI",
        detection_prefixes=("mistral", "mixtral"),
        package="llama-index-llms-mistralai",
    ),
    "groq": ProviderSpec(
        name="groq",
        import_module="llama_index.llms.groq",
        class_name="Groq",
        detection_prefixes=("llama", "gemma"),
        package="llama-index-llms-groq",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        import_module="llama_index.llms.ollama",
        class_name="Ollama",
        detection_prefixes=(),
        package="llama-index-llms-ollama",
    ),
    "huggingface": ProviderSpec(
        name="huggingface",
        import_module="llama_index.llms.huggingface_api",
        class_name="HuggingFaceInferenceAPI",
        model_kwarg="model_name",
        api_key_config_field="huggingface_api_key",
        detection_prefixes=(),
        package="llama-index-llms-huggingface-api",
    ),
    "xai": ProviderSpec(
        name="xai",
        import_module="llama_index.llms.openai",
        class_name="OpenAI",
        api_key_config_field="xai_api_key",
        extra_kwargs={"api_base": "https://api.x.ai/v1"},
        detection_prefixes=("grok",),
        package="llama-index-llms-openai",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        import_module="llama_index.llms.openai",
        class_name="OpenAI",
        api_key_config_field="deepseek_api_key",
        extra_kwargs={"api_base": "https://api.deepseek.com"},
        detection_prefixes=("deepseek",),
        package="llama-index-llms-openai",
    ),
    "cohere": ProviderSpec(
        name="cohere",
        import_module="llama_index.llms.cohere",
        class_name="Cohere",
        detection_prefixes=("command",),
        package="llama-index-llms-cohere",
    ),
}


def detect_provider(model: str) -> str:
    """Auto-detect the LLM provider from a model name string.

    Scans :data:`PROVIDER_REGISTRY` entries and returns the first provider
    whose ``detection_prefixes`` match the start of *model* (case-insensitive).
    Falls back to ``"openai"`` if no prefix matches.

    Args:
        model: A model identifier (e.g. ``"claude-sonnet-4-6"``).

    Returns:
        Provider name string suitable for :data:`PROVIDER_REGISTRY` lookup.
    """
    lower = model.lower()
    for spec in PROVIDER_REGISTRY.values():
        for prefix in spec.detection_prefixes:
            if lower.startswith(prefix):
                logger.debug(
                    "Detected provider %r for model %r (prefix=%r)",
                    spec.name, model, prefix,
                )
                return spec.name
    logger.debug("No prefix matched model %r, falling back to 'openai'", model)
    return "openai"


def build_llm(provider: str, model: str, config: Any) -> Any:
    """Construct a LlamaIndex LLM instance for the given provider.

    Args:
        provider: Key into :data:`PROVIDER_REGISTRY`.
        model: Model identifier passed to the LLM constructor.
        config: A :class:`~hermes.config.HermesConfig` instance used to
            read API keys.

    Returns:
        An instantiated LlamaIndex LLM object.

    Raises:
        ValueError: If *provider* is not in the registry.
        ImportError: If the required package is not installed.
    """
    if provider not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Available providers: {available}"
        )

    spec = PROVIDER_REGISTRY[provider]

    try:
        module = importlib.import_module(spec.import_module)
    except ImportError as exc:
        raise ImportError(
            f"Provider {provider!r} requires package {spec.package!r}. "
            f"Install it with:  pip install {spec.package}"
        ) from exc

    cls = getattr(module, spec.class_name)

    resolved_model = _resolve_llm_model_id(spec, model)
    kwargs: dict[str, Any] = {spec.model_kwarg: resolved_model}
    kwargs.update(spec.extra_kwargs)

    if spec.api_key_config_field:
        key = getattr(config, spec.api_key_config_field, None)
        if key:
            kwargs["api_key"] = key

    max_tokens = getattr(config, "llm_max_tokens", 8192)
    # LlamaIndex OpenAI maps this to max_completion_tokens for O1_MODELS when model id
    # matches; see _resolve_llm_model_id.
    kwargs["max_tokens"] = max_tokens

    # Enable Anthropic prompt caching.  cache_idx=-1 instructs LlamaIndex to
    # add cache_control breakpoints on all messages up to the last one, which
    # covers the system prompt and all accumulated tool context -- the largest
    # cost driver in multi-agent workflows.
    if provider == "anthropic":
        kwargs["cache_idx"] = -1

    # Enable Google GenAI cached content when configured.  The caller must
    # pre-create the cache via the Google GenAI SDK (including the system
    # instruction) and set HERMES_GOOGLE_CACHED_CONTENT to the cache name.
    if provider == "google":
        cached_content = getattr(config, "google_cached_content", None)
        if cached_content:
            kwargs["cached_content"] = cached_content
            logger.info("Google cached content enabled: %s", cached_content)

    logger.info(
        "Building LLM: provider=%r, model=%r, class=%s, max_tokens=%d",
        provider, resolved_model, spec.class_name, max_tokens,
    )
    llm = cls(**kwargs)
    return _wrap_with_retry(llm, provider)


# ---------------------------------------------------------------------------
# Per-call LLM retry
# ---------------------------------------------------------------------------

# Fixed backoff sequence: wait 5s, 15s, 30s between attempts 1→2, 2→3, 3→4.
_BACKOFF_SECONDS: list[float] = [5.0, 15.0, 30.0]


def _wrap_with_retry(llm: Any, provider: str) -> Any:
    """Wrap LLM async chat/complete methods with per-call exponential backoff.

    Retries transparently on rate-limit and transient errors so the calling
    workflow never sees the exception and does not restart.

    Args:
        llm: An instantiated LlamaIndex LLM object.
        provider: Provider name used for rate-limit error detection.

    Returns:
        The same *llm* object with its async methods patched in-place.
    """
    # Imported here to avoid circular imports (retry imports nothing from here).
    from hermes.infra.retry import is_rate_limit_error, is_transient_error

    def _make_retried(original: Any, method_name: str) -> Any:
        @functools.wraps(original)
        async def _retried(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(len(_BACKOFF_SECONDS) + 1):
                try:
                    return await original(*args, **kwargs)
                except Exception as exc:
                    if attempt < len(_BACKOFF_SECONDS) and (
                        is_rate_limit_error(exc, provider) or is_transient_error(exc)
                    ):
                        wait = _BACKOFF_SECONDS[attempt]
                        logger.warning(
                            "LLM %s error (attempt %d/%d) — retrying in %.0fs: %s: %s",
                            method_name,
                            attempt + 1,
                            len(_BACKOFF_SECONDS) + 1,
                            wait,
                            type(exc).__name__,
                            exc,
                            exc_info=True,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise

        return _retried

    for method_name in ("achat", "acomplete"):
        original = getattr(llm, method_name, None)
        if callable(original):
            try:
                # LlamaIndex LLMs are Pydantic models; setattr raises ValueError
                # for unknown fields.  object.__setattr__ writes directly to
                # __dict__ and shadows the class-level method (non-data descriptor)
                # via Python's normal attribute lookup order.
                object.__setattr__(llm, method_name, _make_retried(original, method_name))
            except (AttributeError, TypeError):
                logger.debug(
                    "Could not patch %s.%s for retry — skipping",
                    type(llm).__name__,
                    method_name,
                )

    return llm
