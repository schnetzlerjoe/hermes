"""Async token-bucket rate limiter.

Provides a simple :class:`RateLimiter` that can be ``await``-ed or used as an
``async with`` context manager to throttle outgoing requests.  Pre-configured
limiters for well-known financial APIs are accessible via :func:`get_limiter`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter for async code.

    Parameters
    ----------
    rate:
        Maximum number of tokens (requests) allowed per *per* seconds.
    per:
        Length of the refill window in seconds.  Defaults to ``1.0``.

    Example
    -------
    ::

        limiter = RateLimiter(rate=10, per=1.0)   # 10 req/sec

        async with limiter:
            await httpx.AsyncClient().get(url)
    """

    def __init__(self, rate: float, per: float = 1.0) -> None:
        self.rate = rate
        self.per = per
        self._tokens: float = rate
        self._last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    # -- core algorithm ----------------------------------------------------

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate how long until at least one token is available.
                deficit = 1.0 - self._tokens
                wait = deficit * (self.per / self.rate)

            # Sleep *outside* the lock so other coroutines can check too.
            logger.debug("Rate limiter %s: waiting %.2fs", self.rate, wait)
            await asyncio.sleep(wait)

    # -- context manager ---------------------------------------------------

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Nothing to release; the token was consumed on entry.
        return None

    # -- internals ---------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens based on elapsed time since the last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self.rate, self._tokens + elapsed * (self.rate / self.per))


# ---------------------------------------------------------------------------
# Pre-built limiters for known financial APIs
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    "sec_edgar": (10, 1.0),       # SEC EDGAR: 10 requests/second
    "fred": (2, 1.0),             # FRED: ~120 requests/minute -> 2/sec
    "yahoo_finance": (1, 1.0),    # Yahoo Finance: conservative 1/sec
}

_limiters: dict[str, RateLimiter] = {}


def get_limiter(name: str) -> RateLimiter:
    """Return (or create) a :class:`RateLimiter` for *name*.

    If *name* matches a key in :data:`DEFAULT_LIMITS` the limiter is created
    with the corresponding ``(rate, per)`` tuple.  Unknown names default to
    ``(1, 1.0)`` -- one request per second.
    """
    if name not in _limiters:
        rate, per = DEFAULT_LIMITS.get(name, (1, 1.0))
        _limiters[name] = RateLimiter(rate=rate, per=per)
        logger.debug("Created rate limiter %r (%.0f req/%.1fs)", name, rate, per)
    return _limiters[name]
