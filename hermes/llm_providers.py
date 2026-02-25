"""LLM provider registry and factory for multi-provider support.

This module centralises the knowledge of how to instantiate LlamaIndex LLM
classes for every supported provider.  The :func:`build_llm` factory is the
single entry point used by :class:`hermes.core.Hermes`.

Providers that require extra packages (beyond the core anthropic/openai) are
imported lazily so that users only need to install what they use.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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

    kwargs: dict[str, Any] = {spec.model_kwarg: model}
    kwargs.update(spec.extra_kwargs)

    if spec.api_key_config_field:
        key = getattr(config, spec.api_key_config_field, None)
        if key:
            kwargs["api_key"] = key

    logger.info("Building LLM: provider=%r, model=%r, class=%s", provider, model, spec.class_name)
    return cls(**kwargs)
