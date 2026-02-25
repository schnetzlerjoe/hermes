"""File-based cache with per-item TTL.

Stores cached data on disk at ``~/.hermes/cache/`` (configurable).  Each entry
consists of two files inside a namespace directory:

* ``<sha256>.data`` -- the raw bytes
* ``<sha256>.meta`` -- a small JSON object with *created_at* and *ttl_seconds*

This keeps the implementation simple and inspectable; you can ``ls`` the cache
directory and see exactly what is stored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined TTL constants
# ---------------------------------------------------------------------------

TTL_PERMANENT: float | None = None
TTL_1_HOUR: float = 3600
TTL_24_HOURS: float = 86400
TTL_7_DAYS: float = 604800


# ---------------------------------------------------------------------------
# Internal model
# ---------------------------------------------------------------------------


class CacheEntry(BaseModel):
    """Metadata stored alongside each cached blob."""

    data: bytes
    created_at: float
    ttl_seconds: float | None = None


# ---------------------------------------------------------------------------
# FileCache
# ---------------------------------------------------------------------------


class FileCache:
    """Disk-backed cache with per-item TTL.

    Parameters
    ----------
    base_dir:
        Root directory for all cached data.  ``~`` is expanded automatically.
    """

    def __init__(self, base_dir: str = "~/.hermes/cache") -> None:
        self._base = Path(base_dir).expanduser()
        self._base.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------

    def get(self, namespace: str, key: str) -> bytes | None:
        """Return cached bytes for *key*, or ``None`` if missing / expired.

        Expired entries are deleted from disk as a side-effect.
        """
        data_path = self._entry_path(namespace, key)
        meta_path = self._meta_path(namespace, key)

        if not data_path.exists() or not meta_path.exists():
            logger.debug("Cache miss: %s/%s", namespace, key)
            return None

        meta = self._read_meta(meta_path)
        if meta is None:
            # Corrupt metadata -- treat as miss.
            self._remove_pair(data_path, meta_path)
            logger.debug("Cache miss: %s/%s", namespace, key)
            return None

        if self._is_expired(meta):
            self._remove_pair(data_path, meta_path)
            logger.debug("Cache expired: %s/%s", namespace, key)
            return None

        return data_path.read_bytes()

    def put(
        self,
        namespace: str,
        key: str,
        data: bytes,
        ttl_seconds: float | None = None,
    ) -> None:
        """Write *data* to the cache.

        Parameters
        ----------
        namespace:
            Logical grouping (e.g. ``"sec_filings"``).
        key:
            Arbitrary string key.  Hashed to SHA-256 for the filename.
        data:
            Raw bytes to store.
        ttl_seconds:
            Seconds until expiry.  ``None`` means the entry never expires.
        """
        data_path = self._entry_path(namespace, key)
        meta_path = self._meta_path(namespace, key)

        data_path.parent.mkdir(parents=True, exist_ok=True)

        data_path.write_bytes(data)

        meta = {"created_at": time.time(), "ttl_seconds": ttl_seconds}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        logger.debug("Cache put: %s/%s (ttl=%s)", namespace, key, ttl_seconds)

    def has(self, namespace: str, key: str) -> bool:
        """Return ``True`` if a valid (non-expired) entry exists for *key*."""
        data_path = self._entry_path(namespace, key)
        meta_path = self._meta_path(namespace, key)

        if not data_path.exists() or not meta_path.exists():
            return False

        meta = self._read_meta(meta_path)
        if meta is None:
            return False

        if self._is_expired(meta):
            self._remove_pair(data_path, meta_path)
            return False

        return True

    def delete(self, namespace: str, key: str) -> bool:
        """Remove a single cached entry.  Returns ``True`` if it existed."""
        data_path = self._entry_path(namespace, key)
        meta_path = self._meta_path(namespace, key)
        existed = data_path.exists() or meta_path.exists()
        self._remove_pair(data_path, meta_path)
        return existed

    def clear_namespace(self, namespace: str) -> None:
        """Delete all entries in *namespace*."""
        logger.debug("Cleared cache namespace %r", namespace)
        ns_dir = self._base / namespace
        if ns_dir.exists():
            shutil.rmtree(ns_dir)

    def clear_all(self) -> None:
        """Delete every cached entry across all namespaces."""
        logger.info("Cleared all cache")
        if self._base.exists():
            shutil.rmtree(self._base)
            self._base.mkdir(parents=True, exist_ok=True)

    # -- path helpers ------------------------------------------------------

    def _entry_path(self, namespace: str, key: str) -> Path:
        """Return the ``.data`` file path for a given namespace/key pair."""
        return self._base / namespace / (self._hash_key(key) + ".data")

    def _meta_path(self, namespace: str, key: str) -> Path:
        """Return the ``.meta`` file path for a given namespace/key pair."""
        return self._base / namespace / (self._hash_key(key) + ".meta")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _hash_key(key: str) -> str:
        """SHA-256 hex digest of *key*."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _read_meta(meta_path: Path) -> dict | None:
        """Read and parse a ``.meta`` JSON file; return ``None`` on failure."""
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _is_expired(meta: dict) -> bool:
        """Check whether a metadata dict indicates an expired entry."""
        ttl = meta.get("ttl_seconds")
        if ttl is None:
            return False
        return time.time() - meta["created_at"] > ttl

    @staticmethod
    def _remove_pair(data_path: Path, meta_path: Path) -> None:
        """Silently remove the data and metadata files if they exist."""
        data_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
