"""Tests for hermes.llm_providers — provider registry, detection, and factory."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hermes.config import HermesConfig
from hermes.llm_providers import PROVIDER_REGISTRY, ProviderSpec, build_llm, detect_provider

# ---------------------------------------------------------------------------
# TestDetectProvider
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """Verify that detect_provider maps model prefixes to the right provider."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # anthropic
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-3-opus-20240229", "anthropic"),
            ("Claude-Haiku", "anthropic"),
            # openai
            ("gpt-4o", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini", "openai"),
            # google
            ("gemini-2.0-flash", "google"),
            ("gemini-pro", "google"),
            # mistral
            ("mistral-large-latest", "mistral"),
            ("mixtral-8x7b", "mistral"),
            # groq
            ("llama-3.1-70b", "groq"),
            ("gemma-7b-it", "groq"),
            # xai
            ("grok-2", "xai"),
            ("grok-beta", "xai"),
            # deepseek
            ("deepseek-chat", "deepseek"),
            ("deepseek-coder", "deepseek"),
            # cohere
            ("command-r-plus", "cohere"),
            ("command-light", "cohere"),
            # fallback
            ("some-unknown-model", "openai"),
        ],
    )
    def test_prefix_detection(self, model: str, expected: str) -> None:
        assert detect_provider(model) == expected

    def test_case_insensitive(self) -> None:
        assert detect_provider("CLAUDE-SONNET-4-6") == "anthropic"
        assert detect_provider("GPT-4o") == "openai"
        assert detect_provider("Gemini-Pro") == "google"


# ---------------------------------------------------------------------------
# TestBuildLlm
# ---------------------------------------------------------------------------


class TestBuildLlm:
    """Verify build_llm constructs the right class with the right kwargs."""

    def _make_config(self, **overrides: Any) -> HermesConfig:
        return HermesConfig(
            llm_provider="anthropic",
            llm_model="test-model",
            chroma_persist_dir="/tmp/test_chroma",
            output_dir="/tmp/test_output",
            cache_dir="/tmp/test_cache",
            **overrides,
        )

    def test_unknown_provider_raises(self) -> None:
        config = self._make_config()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_llm("not-a-provider", "some-model", config)

    def test_missing_package_raises(self) -> None:
        config = self._make_config()
        with patch("hermes.llm_providers.importlib.import_module", side_effect=ImportError):
            with pytest.raises(ImportError, match="pip install"):
                build_llm("anthropic", "claude-sonnet-4-6", config)

    @pytest.mark.parametrize(
        ("provider", "expected_class_name", "expected_model_kwarg"),
        [
            ("anthropic", "Anthropic", "model"),
            ("openai", "OpenAI", "model"),
            ("google", "GoogleGenAI", "model"),
            ("mistral", "MistralAI", "model"),
            ("groq", "Groq", "model"),
            ("ollama", "Ollama", "model"),
            ("huggingface", "HuggingFaceInferenceAPI", "model_name"),
            ("xai", "OpenAI", "model"),
            ("deepseek", "OpenAI", "model"),
            ("cohere", "Cohere", "model"),
        ],
    )
    def test_builds_correct_class(
        self,
        provider: str,
        expected_class_name: str,
        expected_model_kwarg: str,
    ) -> None:
        config = self._make_config()
        mock_cls = MagicMock(name=expected_class_name)
        mock_module = MagicMock()
        setattr(mock_module, expected_class_name, mock_cls)

        with patch("hermes.llm_providers.importlib.import_module", return_value=mock_module):
            build_llm(provider, "test-model", config)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs[expected_model_kwarg] == "test-model"

    def test_xai_passes_api_base(self) -> None:
        config = self._make_config(xai_api_key="xai-key-123")
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OpenAI = mock_cls

        with patch("hermes.llm_providers.importlib.import_module", return_value=mock_module):
            build_llm("xai", "grok-2", config)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_base"] == "https://api.x.ai/v1"
        assert call_kwargs["api_key"] == "xai-key-123"

    def test_deepseek_passes_api_base(self) -> None:
        config = self._make_config(deepseek_api_key="ds-key-456")
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OpenAI = mock_cls

        with patch("hermes.llm_providers.importlib.import_module", return_value=mock_module):
            build_llm("deepseek", "deepseek-chat", config)

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_base"] == "https://api.deepseek.com"
        assert call_kwargs["api_key"] == "ds-key-456"

    def test_huggingface_uses_model_name_kwarg(self) -> None:
        config = self._make_config(huggingface_api_key="hf-key-789")
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.HuggingFaceInferenceAPI = mock_cls

        with patch("hermes.llm_providers.importlib.import_module", return_value=mock_module):
            build_llm("huggingface", "meta-llama/Llama-2-70b", config)

        call_kwargs = mock_cls.call_args[1]
        assert "model_name" in call_kwargs
        assert "model" not in call_kwargs
        assert call_kwargs["model_name"] == "meta-llama/Llama-2-70b"
        assert call_kwargs["api_key"] == "hf-key-789"

    def test_no_api_key_when_field_is_none(self) -> None:
        config = self._make_config()
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.OpenAI = mock_cls

        with patch("hermes.llm_providers.importlib.import_module", return_value=mock_module):
            build_llm("xai", "grok-2", config)

        call_kwargs = mock_cls.call_args[1]
        assert "api_key" not in call_kwargs


# ---------------------------------------------------------------------------
# TestProviderRegistry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    """Structural invariants on the provider registry."""

    def test_all_specs_are_provider_spec(self) -> None:
        for name, spec in PROVIDER_REGISTRY.items():
            assert isinstance(spec, ProviderSpec), f"{name} is not a ProviderSpec"

    def test_all_specs_have_required_fields(self) -> None:
        for name, spec in PROVIDER_REGISTRY.items():
            assert spec.name == name, f"Spec name {spec.name!r} != key {name!r}"
            assert spec.import_module, f"{name} missing import_module"
            assert spec.class_name, f"{name} missing class_name"
            assert spec.package, f"{name} missing package"

    def test_no_duplicate_detection_prefixes(self) -> None:
        seen: dict[str, str] = {}
        for name, spec in PROVIDER_REGISTRY.items():
            for prefix in spec.detection_prefixes:
                assert prefix not in seen, (
                    f"Duplicate prefix {prefix!r}: "
                    f"used by both {seen[prefix]!r} and {name!r}"
                )
                seen[prefix] = name

    def test_registry_has_ten_providers(self) -> None:
        assert len(PROVIDER_REGISTRY) == 10
