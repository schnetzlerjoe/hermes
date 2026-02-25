"""Shared test fixtures for the Hermes test suite."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hermes.config import HermesConfig
from hermes.registry import Registry

# ---------------------------------------------------------------------------
# Configuration fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermes_config(tmp_path: Path) -> HermesConfig:
    """Create a HermesConfig with test-safe defaults.

    Uses temporary directories for all storage paths so tests never
    write to the user's home directory or working directory.
    """
    return HermesConfig(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        sec_user_agent="HermesTestSuite test@example.com",
        fred_api_key="test-fred-key-not-real",
        chroma_persist_dir=str(tmp_path / "chroma"),
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Registry fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> Registry:
    """Create a fresh, empty Registry instance."""
    return Registry()


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test output files.

    The directory is created automatically and cleaned up by pytest
    after the test completes.
    """
    output = tmp_path / "test_output"
    output.mkdir(parents=True, exist_ok=True)
    return output


# ---------------------------------------------------------------------------
# Mock HTTP client
# ---------------------------------------------------------------------------


class MockHTTPResponse:
    """A minimal mock for httpx.Response.

    Supports the subset of the Response API that Hermes tools use:
    status_code, json(), text, content, and raise_for_status().
    """

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (content.decode("utf-8", errors="replace") if content else "")
        self.content = content or text.encode("utf-8")

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        import json

        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"HTTP {self.status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )


class MockAsyncHTTPClient:
    """A programmable async HTTP client for offline testing.

    Register response handlers with ``add_response()`` keyed by URL
    substring.  When ``get()`` is called, the first matching handler is
    used to produce the response.

    Example::

        mock_client = MockAsyncHTTPClient()
        mock_client.add_response(
            "data.sec.gov",
            MockHTTPResponse(json_data={"cik": "0000320193"}),
        )
    """

    def __init__(self) -> None:
        self._responses: list[tuple[str, MockHTTPResponse]] = []
        self._default = MockHTTPResponse(
            status_code=404, json_data={"error": "no mock configured"}
        )
        self._call_log: list[dict[str, Any]] = []

    def add_response(self, url_contains: str, response: MockHTTPResponse) -> None:
        """Register a response for URLs containing the given substring."""
        self._responses.append((url_contains, response))

    async def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        **kwargs: Any,
    ) -> MockHTTPResponse:
        """Simulate an async GET request."""
        self._call_log.append(
            {"method": "GET", "url": url, "params": params, "headers": headers}
        )
        for substring, response in self._responses:
            if substring in url:
                return response
        return self._default

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        data: Any = None,
        headers: dict | None = None,
        **kwargs: Any,
    ) -> MockHTTPResponse:
        """Simulate an async POST request."""
        self._call_log.append(
            {"method": "POST", "url": url, "json": json, "headers": headers}
        )
        for substring, response in self._responses:
            if substring in url:
                return response
        return self._default

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return the log of all requests made to this mock."""
        return self._call_log


@pytest.fixture()
def mock_http_client() -> Generator[MockAsyncHTTPClient, None, None]:
    """Patch httpx.AsyncClient globally and return a MockAsyncHTTPClient.

    All code that calls ``hermes.tools._base.get_http_client()`` will
    receive this mock instead of a real HTTP client.  The mock is reset
    after each test.

    Usage in tests::

        def test_something(mock_http_client):
            mock_http_client.add_response(
                "data.sec.gov/api/xbrl",
                MockHTTPResponse(json_data={...}),
            )
            # Now call the tool under test -- it will hit the mock.
    """
    mock_client = MockAsyncHTTPClient()

    # Patch the module-level singleton in hermes.tools._base so that
    # get_http_client() returns our mock.
    with patch("hermes.tools._base._client", mock_client):
        yield mock_client

    # Reset the module-level singleton to None so the next test gets
    # a fresh state.
    import hermes.tools._base as base_mod

    base_mod._client = None
