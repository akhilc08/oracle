"""Tool caching layer — TTL-based in-memory cache for agent tool calls."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# Default TTLs by tool category (seconds)
DEFAULT_TTLS: dict[str, int] = {
    "market_data": 60,
    "news": 900,
    "graph_query": 300,
    "retrieval": 600,
}


@dataclass
class CacheEntry:
    """A single cached value with expiry."""

    value: Any
    expires_at: float


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ToolCache:
    """TTL-based in-memory cache for tool results.

    Cache key is derived from tool name + sorted kwargs.
    """

    def __init__(self) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    @staticmethod
    def _make_key(tool_name: str, kwargs: dict[str, Any]) -> str:
        """Generate deterministic cache key from tool name and kwargs."""
        raw = json.dumps({"tool": tool_name, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, tool_name: str, kwargs: dict[str, Any]) -> tuple[bool, Any]:
        """Look up a cached value. Returns (hit, value)."""
        key = self._make_key(tool_name, kwargs)
        entry = self._cache.get(key)

        if entry is not None and time.time() < entry.expires_at:
            self._stats.hits += 1
            return True, entry.value

        if entry is not None:
            # Expired — evict
            del self._cache[key]

        self._stats.misses += 1
        return False, None

    def put(self, tool_name: str, kwargs: dict[str, Any], value: Any, ttl: int) -> None:
        """Store a value with TTL."""
        key = self._make_key(tool_name, kwargs)
        self._cache[key] = CacheEntry(value=value, expires_at=time.time() + ttl)

    def invalidate(self, tool_name: str, kwargs: dict[str, Any]) -> None:
        """Remove a specific cache entry."""
        key = self._make_key(tool_name, kwargs)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._cache)


# Global cache instance
_global_cache = ToolCache()


def get_cache() -> ToolCache:
    """Get the global tool cache instance."""
    return _global_cache


def cached_tool(ttl: int = 300) -> Callable:
    """Decorator that caches async tool function results.

    Usage:
        @cached_tool(ttl=60)
        async def get_market_data(market_id: str) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()
            tool_name = func.__qualname__
            cache_kwargs = kwargs.copy()
            # Include positional args (skip self if method)
            for i, arg in enumerate(args):
                if i == 0 and hasattr(arg, "__class__") and not isinstance(arg, (str, int, float)):
                    continue  # Skip self
                cache_kwargs[f"_arg{i}"] = arg

            hit, value = cache.get(tool_name, cache_kwargs)
            if hit:
                logger.debug("cache.hit", tool=tool_name)
                return value

            result = await func(*args, **kwargs)
            if result:  # don't cache empty/falsy results (e.g. failed API calls)
                cache.put(tool_name, cache_kwargs, result, ttl)
            return result

        wrapper._cache_ttl = ttl  # type: ignore[attr-defined]
        return wrapper

    return decorator
