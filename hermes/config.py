"""Configuration management for the Hermes library.

All runtime configuration -- LLM provider selection, API keys, storage paths,
and parser preferences -- is centralised in :class:`HermesConfig`.  A single
module-level instance is lazily created on first access via :func:`get_config`
and can be explicitly set or updated with :func:`configure`.

Environment variables with the ``HERMES_`` prefix are read automatically
(e.g. ``HERMES_LLM_PROVIDER=openai``).
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class HermesConfig(BaseModel):
    """Central configuration for every Hermes component.

    Fields fall into four categories:

    1. **LLM** -- provider choice and model identifier.
    2. **Data-source keys** -- credentials for SEC EDGAR, FRED, Polygon, etc.
    3. **Parsing** -- which document parser backend to use.
    4. **Storage** -- local paths for ChromaDB persistence, output artefacts,
       and an HTTP cache.

    All string fields that represent filesystem paths have ``~`` expanded at
    validation time so callers never have to worry about it.
    """

    model_config = ConfigDict(
        env_prefix="HERMES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- LLM ------------------------------------------------------------------
    llm_provider: Literal[
        "anthropic",
        "openai",
        "google",
        "mistral",
        "groq",
        "ollama",
        "huggingface",
        "xai",
        "deepseek",
        "cohere",
    ] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"

    # -- Data-source API keys -------------------------------------------------
    sec_user_agent: str | None = None
    """SEC EDGAR requires a User-Agent identifying the caller."""

    fred_api_key: str | None = None
    polygon_api_key: str | None = None
    fmp_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    xai_api_key: str | None = None
    deepseek_api_key: str | None = None
    huggingface_api_key: str | None = None

    # -- Parsing --------------------------------------------------------------
    parser: Literal["html", "llamaparse"] = "html"
    llamaparse_api_key: str | None = None

    # -- Storage paths --------------------------------------------------------
    chroma_persist_dir: str = "./hermes_data"
    output_dir: str = "./output"
    cache_dir: str = "~/.hermes/cache"

    # -- Behaviour ------------------------------------------------------------
    verbose: bool = False
    llm_max_retries: int = 3
    llm_retry_max_wait: float = 120.0

    # -- Validators -----------------------------------------------------------

    @model_validator(mode="after")
    def _expand_home_in_paths(self) -> HermesConfig:
        """Expand ``~`` to the user's home directory in all path fields."""
        self.chroma_persist_dir = os.path.expanduser(self.chroma_persist_dir)
        self.output_dir = os.path.expanduser(self.output_dir)
        self.cache_dir = os.path.expanduser(self.cache_dir)
        return self

    @model_validator(mode="after")
    def _validate_llamaparse_key(self) -> HermesConfig:
        """Ensure a LlamaParse API key is present when that parser is selected."""
        if self.parser == "llamaparse" and not self.llamaparse_api_key:
            raise ValueError(
                "llamaparse_api_key must be set when parser is 'llamaparse'. "
                "Provide it directly or set the HERMES_LLAMAPARSE_API_KEY env var."
            )
        return self


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: HermesConfig | None = None


def _env_overrides() -> dict[str, object]:
    """Read HERMES_* environment variables and return as a field-name → value dict.

    HermesConfig is a plain BaseModel (not pydantic-settings BaseSettings), so
    env vars are not read automatically.  This helper bridges that gap so that
    :func:`configure` and :func:`get_config` honour the documented HERMES_ prefix.
    """
    overrides: dict[str, object] = {}
    for field_name in HermesConfig.model_fields:
        env_key = f"HERMES_{field_name.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            overrides[field_name] = val
    return overrides


def configure(**kwargs: object) -> HermesConfig:
    """Create or update the global :class:`HermesConfig`.

    Keyword arguments are forwarded to the ``HermesConfig`` constructor.  If a
    global config already exists its current values are used as defaults so that
    callers only need to specify the fields they want to change.

    Environment variables with the ``HERMES_`` prefix (e.g.
    ``HERMES_LLM_PROVIDER=google``) are applied before explicit *kwargs*, so
    kwargs always win.

    Returns:
        The newly created or updated :class:`HermesConfig` instance.
    """
    global _config  # noqa: PLW0603

    env = _env_overrides()

    if _config is not None:
        # Merge: existing config < env vars < explicit kwargs
        merged = {**_config.model_dump(), **env, **kwargs}
        _config = HermesConfig(**merged)
    else:
        _config = HermesConfig(**env, **kwargs)

    return _config


def get_config() -> HermesConfig:
    """Return the global config, creating a default instance if needed.

    This is the canonical way for internal modules to obtain configuration.
    End-users should prefer :func:`configure` to customise settings before
    calling any library code.
    """
    global _config  # noqa: PLW0603

    if _config is None:
        _config = HermesConfig(**_env_overrides())

    return _config
