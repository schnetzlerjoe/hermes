"""Tests for the async rate limiter.

Uses pytest-asyncio to test the token-bucket rate limiter with real
asyncio timing.  Tests verify token consumption, blocking when
exhausted, refill behaviour, and context manager usage.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hermes.infra.rate_limiter import RateLimiter, get_limiter

# ---------------------------------------------------------------------------
# Tests: token consumption
# ---------------------------------------------------------------------------


class TestTokenConsumption:
    """Test that tokens are consumed on each acquire."""

    @pytest.mark.asyncio
    async def test_single_acquire_succeeds(self) -> None:
        """A single acquire on a fresh limiter should succeed immediately."""
        limiter = RateLimiter(rate=10, per=1.0)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        # Should be near-instant (well under 100ms).
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_multiple_acquires_within_rate(self) -> None:
        """Acquiring fewer tokens than the rate should not block."""
        limiter = RateLimiter(rate=10, per=1.0)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        # 5 out of 10 tokens should be instant.
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_exhaust_all_tokens(self) -> None:
        """Acquiring exactly rate tokens should succeed without delay."""
        limiter = RateLimiter(rate=5, per=1.0)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# Tests: blocking when tokens exhausted
# ---------------------------------------------------------------------------


class TestBlocking:
    """Test that acquire blocks when no tokens are available."""

    @pytest.mark.asyncio
    async def test_blocks_when_exhausted(self) -> None:
        """Acquiring one more token than the rate should introduce a delay."""
        # Rate of 5 tokens per second.  After consuming all 5, the 6th
        # should block until a token refills (~0.2 seconds for 1 token
        # at rate 5/sec).
        limiter = RateLimiter(rate=5, per=1.0)

        # Exhaust all tokens.
        for _ in range(5):
            await limiter.acquire()

        # The next acquire should block.
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited roughly 0.2 seconds (1 token / 5 per sec).
        # Allow generous bounds for CI timing jitter.
        assert elapsed >= 0.1
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_serialised(self) -> None:
        """Multiple concurrent acquire() calls should be served in order."""
        limiter = RateLimiter(rate=2, per=1.0)

        results: list[float] = []

        async def timed_acquire(idx: int) -> None:
            await limiter.acquire()
            results.append(time.monotonic())

        # Launch 4 acquires concurrently.  With rate=2, the first two
        # should succeed immediately and the next two should wait.
        start = time.monotonic()
        await asyncio.gather(*[timed_acquire(i) for i in range(4)])

        # All 4 should have completed.
        assert len(results) == 4

        # The first two should be near-instant, the latter two delayed.
        # At least one result should be >0.3s after start.
        delays = [t - start for t in results]
        assert max(delays) >= 0.3


# ---------------------------------------------------------------------------
# Tests: refill behaviour
# ---------------------------------------------------------------------------


class TestRefill:
    """Test that tokens refill over time."""

    @pytest.mark.asyncio
    async def test_tokens_refill_after_wait(self) -> None:
        """After exhausting tokens and waiting, new tokens should be available."""
        limiter = RateLimiter(rate=5, per=1.0)

        # Exhaust all tokens.
        for _ in range(5):
            await limiter.acquire()

        # Wait long enough for a full refill (1 second for 5 tokens).
        await asyncio.sleep(1.1)

        # Should be able to acquire 5 more tokens instantly.
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_partial_refill(self) -> None:
        """Waiting for less than the full period should refill proportionally."""
        limiter = RateLimiter(rate=10, per=1.0)

        # Exhaust all tokens.
        for _ in range(10):
            await limiter.acquire()

        # Wait 0.5 seconds => expect ~5 tokens refilled.
        await asyncio.sleep(0.5)

        # Should be able to acquire about 5 tokens quickly.
        start = time.monotonic()
        for _ in range(4):  # Use 4 to be safe with timing.
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_tokens_cap_at_rate(self) -> None:
        """Tokens should never exceed the configured rate, even after long waits."""
        limiter = RateLimiter(rate=3, per=1.0)

        # Wait much longer than needed for a full refill.
        await asyncio.sleep(0.5)

        # Even after oversleeping, we should only have 3 tokens (the cap).
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed_3 = time.monotonic() - start
        assert elapsed_3 < 0.1

        # The 4th acquire should block.
        start2 = time.monotonic()
        await limiter.acquire()
        elapsed_4 = time.monotonic() - start2
        assert elapsed_4 >= 0.2


# ---------------------------------------------------------------------------
# Tests: context manager usage
# ---------------------------------------------------------------------------


class TestContextManager:
    """Test using the rate limiter as an async context manager."""

    @pytest.mark.asyncio
    async def test_async_with_acquires_token(self) -> None:
        """Using 'async with limiter' should consume a token."""
        limiter = RateLimiter(rate=3, per=1.0)

        # Use the context manager 3 times (exhaust tokens).
        for _ in range(3):
            async with limiter:
                pass

        # The 4th should block.
        start = time.monotonic()
        async with limiter:
            pass
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_context_manager_does_not_leak_on_exception(self) -> None:
        """An exception inside the context manager should not break the limiter."""
        limiter = RateLimiter(rate=5, per=1.0)

        with pytest.raises(ValueError, match="test error"):
            async with limiter:
                raise ValueError("test error")

        # The limiter should still work after the exception.
        async with limiter:
            pass  # Should succeed.


# ---------------------------------------------------------------------------
# Tests: get_limiter factory
# ---------------------------------------------------------------------------


class TestGetLimiter:
    """Test the pre-configured limiter factory."""

    def test_returns_known_limiter(self) -> None:
        """get_limiter() for a known API should return a correctly configured instance."""
        limiter = get_limiter("sec_edgar")
        assert limiter.rate == 10
        assert limiter.per == 1.0

    def test_returns_same_instance(self) -> None:
        """get_limiter() should return the same instance for repeated calls."""
        a = get_limiter("fred")
        b = get_limiter("fred")
        assert a is b

    def test_unknown_name_gets_default(self) -> None:
        """An unknown API name should get a default 1 req/sec limiter."""
        limiter = get_limiter("unknown_api_for_test")
        assert limiter.rate == 1
        assert limiter.per == 1.0
