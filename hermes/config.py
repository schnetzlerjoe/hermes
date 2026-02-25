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


def configure(**kwargs: object) -> HermesConfig:
    """Create or update the global :class:`HermesConfig`.

    Keyword arguments are forwarded to the ``HermesConfig`` constructor.  If a
    global config already exists its current values are used as defaults so that
    callers only need to specify the fields they want to change.

    Returns:
        The newly created or updated :class:`HermesConfig` instance.
    """
    global _config  # noqa: PLW0603

    if _config is not None:
        # Merge existing values with overrides so callers can do incremental
        # updates (e.g. ``configure(verbose=True)`` without losing keys).
        merged = {**_config.model_dump(), **kwargs}
        _config = HermesConfig(**merged)
    else:
        _config = HermesConfig(**kwargs)

    return _config


def get_config() -> HermesConfig:
    """Return the global config, creating a default instance if needed.

    This is the canonical way for internal modules to obtain configuration.
    End-users should prefer :func:`configure` to customise settings before
    calling any library code.
    """
    global _config  # noqa: PLW0603

    if _config is None:
        _config = HermesConfig()

    return _config
