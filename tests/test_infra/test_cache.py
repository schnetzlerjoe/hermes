"""Tests for the file-based cache.

All tests use real filesystem operations in temporary directories.
TTL expiry tests use short TTLs with time.sleep() to verify actual
time-based behaviour rather than mocking the clock.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes.infra.cache import FileCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache(tmp_path: Path) -> FileCache:
    """Create a FileCache rooted in a temporary directory."""
    return FileCache(base_dir=str(tmp_path / "test_cache"))


# ---------------------------------------------------------------------------
# Tests: put and get
# ---------------------------------------------------------------------------


class TestPutAndGet:
    """Test basic cache storage and retrieval."""

    def test_put_and_get_returns_same_bytes(self, cache: FileCache) -> None:
        """Cached data should be returned unchanged."""
        data = b"Apple Inc reported revenue of $394.3 billion"
        cache.put("test_ns", "key1", data)
        result = cache.get("test_ns", "key1")
        assert result == data

    def test_get_missing_key_returns_none(self, cache: FileCache) -> None:
        """Requesting a non-existent key should return None."""
        result = cache.get("test_ns", "nonexistent")
        assert result is None

    def test_put_overwrites_existing(self, cache: FileCache) -> None:
        """A second put with the same key should overwrite the first."""
        cache.put("test_ns", "key1", b"version1")
        cache.put("test_ns", "key1", b"version2")
        result = cache.get("test_ns", "key1")
        assert result == b"version2"

    def test_binary_data(self, cache: FileCache) -> None:
        """Binary (non-UTF-8) data should round-trip correctly."""
        data = bytes(range(256))
        cache.put("binary_ns", "binkey", data)
        result = cache.get("binary_ns", "binkey")
        assert result == data

    def test_large_data(self, cache: FileCache) -> None:
        """A large payload (1 MB) should store and retrieve correctly."""
        data = b"X" * (1024 * 1024)
        cache.put("large_ns", "bigkey", data)
        result = cache.get("large_ns", "bigkey")
        assert result == data
        assert len(result) == 1024 * 1024

    def test_has_returns_true_for_existing_key(self, cache: FileCache) -> None:
        """has() should return True for a valid, non-expired entry."""
        cache.put("test_ns", "exists", b"data")
        assert cache.has("test_ns", "exists") is True

    def test_has_returns_false_for_missing_key(self, cache: FileCache) -> None:
        """has() should return False for a non-existent key."""
        assert cache.has("test_ns", "missing") is False


# ---------------------------------------------------------------------------
# Tests: TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    """Test time-based cache expiration."""

    def test_entry_available_before_ttl(self, cache: FileCache) -> None:
        """An entry should be retrievable immediately after creation."""
        cache.put("ttl_ns", "fresh", b"data", ttl_seconds=10.0)
        result = cache.get("ttl_ns", "fresh")
        assert result == b"data"

    def test_entry_expires_after_ttl(self, cache: FileCache) -> None:
        """An entry should return None after its TTL has elapsed."""
        cache.put("ttl_ns", "expiring", b"data", ttl_seconds=0.3)
        # Verify it exists right after creation.
        assert cache.get("ttl_ns", "expiring") == b"data"

        # Wait for expiry.
        time.sleep(0.5)
        result = cache.get("ttl_ns", "expiring")
        assert result is None

    def test_has_returns_false_after_expiry(self, cache: FileCache) -> None:
        """has() should return False after TTL expiry."""
        cache.put("ttl_ns", "check_has", b"data", ttl_seconds=0.3)
        assert cache.has("ttl_ns", "check_has") is True
        time.sleep(0.5)
        assert cache.has("ttl_ns", "check_has") is False

    def test_permanent_entry_never_expires(self, cache: FileCache) -> None:
        """An entry with ttl_seconds=None should never expire."""
        cache.put("ttl_ns", "permanent", b"forever", ttl_seconds=None)
        # We cannot truly test "forever" but we can verify it survives
        # a short sleep that would expire a short-TTL entry.
        time.sleep(0.3)
        result = cache.get("ttl_ns", "permanent")
        assert result == b"forever"

    def test_expired_entry_is_cleaned_from_disk(self, cache: FileCache) -> None:
        """After expiry, the data and metadata files should be deleted."""
        cache.put("ttl_ns", "cleanup", b"temp", ttl_seconds=0.2)
        data_path = cache._entry_path("ttl_ns", "cleanup")
        meta_path = cache._meta_path("ttl_ns", "cleanup")
        assert data_path.exists()
        assert meta_path.exists()

        time.sleep(0.4)
        # Trigger cleanup by attempting a get.
        cache.get("ttl_ns", "cleanup")
        assert not data_path.exists()
        assert not meta_path.exists()


# ---------------------------------------------------------------------------
# Tests: namespaces
# ---------------------------------------------------------------------------


class TestNamespaces:
    """Test namespace isolation."""

    def test_same_key_different_namespaces(self, cache: FileCache) -> None:
        """The same key in different namespaces should hold different data."""
        cache.put("ns_a", "shared_key", b"alpha")
        cache.put("ns_b", "shared_key", b"beta")

        assert cache.get("ns_a", "shared_key") == b"alpha"
        assert cache.get("ns_b", "shared_key") == b"beta"

    def test_clear_namespace_only_affects_target(self, cache: FileCache) -> None:
        """Clearing one namespace should not affect others."""
        cache.put("ns_keep", "key1", b"keep_me")
        cache.put("ns_clear", "key1", b"clear_me")

        cache.clear_namespace("ns_clear")

        assert cache.get("ns_keep", "key1") == b"keep_me"
        assert cache.get("ns_clear", "key1") is None

    def test_clear_nonexistent_namespace_is_noop(self, cache: FileCache) -> None:
        """Clearing a namespace that doesn't exist should not raise."""
        cache.clear_namespace("does_not_exist")  # Should not raise.


# ---------------------------------------------------------------------------
# Tests: delete
# ---------------------------------------------------------------------------


class TestDelete:
    """Test single-entry deletion."""

    def test_delete_existing_entry(self, cache: FileCache) -> None:
        """Deleting an existing key should return True and remove the data."""
        cache.put("del_ns", "target", b"data")
        assert cache.delete("del_ns", "target") is True
        assert cache.get("del_ns", "target") is None

    def test_delete_nonexistent_entry(self, cache: FileCache) -> None:
        """Deleting a non-existent key should return False."""
        assert cache.delete("del_ns", "ghost") is False

    def test_delete_does_not_affect_other_keys(self, cache: FileCache) -> None:
        """Deleting one key should not affect other keys in the same namespace."""
        cache.put("del_ns", "keep", b"stay")
        cache.put("del_ns", "remove", b"go")
        cache.delete("del_ns", "remove")

        assert cache.get("del_ns", "keep") == b"stay"
        assert cache.get("del_ns", "remove") is None


# ---------------------------------------------------------------------------
# Tests: clear all
# ---------------------------------------------------------------------------


class TestClearAll:
    """Test clearing the entire cache."""

    def test_clear_all_removes_everything(self, cache: FileCache) -> None:
        """clear_all() should remove all entries across all namespaces."""
        cache.put("ns1", "key1", b"data1")
        cache.put("ns2", "key2", b"data2")
        cache.put("ns3", "key3", b"data3")

        cache.clear_all()

        assert cache.get("ns1", "key1") is None
        assert cache.get("ns2", "key2") is None
        assert cache.get("ns3", "key3") is None

    def test_clear_all_allows_new_entries(self, cache: FileCache) -> None:
        """After clear_all(), new entries should work normally."""
        cache.put("ns1", "old", b"old_data")
        cache.clear_all()

        cache.put("ns1", "new", b"new_data")
        assert cache.get("ns1", "new") == b"new_data"

    def test_clear_all_on_empty_cache(self, cache: FileCache) -> None:
        """clear_all() on an empty cache should not raise."""
        cache.clear_all()  # Should not raise.
