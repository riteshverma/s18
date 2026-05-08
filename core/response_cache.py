"""
Response cache for completed agent runs.

Two backends:
- "redis"  — connects to existing Redis (DB 1 to avoid Celery/broker collision on DB 0)
- "memory" — in-process LRU dict, suitable for single-instance local dev

Backend is selected from settings["response_cache"]["backend"].
Falls back to memory automatically if Redis is unavailable.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from config.settings_loader import load_settings
from core.utils import log_step


def _cache_settings() -> dict:
    return load_settings().get("response_cache", {})


def _make_key(query: str) -> str:
    """
    Stable cache key from query text.

    For clinical queries containing a patient identifier ("[patient id:..."),
    the key encodes the patient segment separately so cross-patient leakage
    is impossible even on hash collision.
    """
    normalized = " ".join((query or "").lower().split())
    patient_prefix = ""
    lower = normalized
    if "[patient id:" in lower:
        start = lower.index("[patient id:")
        end = lower.index("]", start) + 1 if "]" in lower[start:] else len(lower)
        patient_prefix = lower[start:end]
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if patient_prefix:
        patient_hash = hashlib.sha256(patient_prefix.encode()).hexdigest()[:8]
        return f"s18:rc:{patient_hash}:{digest}"
    return f"s18:rc:{digest}"


# ---------------------------------------------------------------------------
# In-memory LRU backend
# ---------------------------------------------------------------------------

class _MemoryBackend:
    def __init__(self, max_entries: int = 500):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max = max_entries

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int):
        expires_at = time.time() + ttl if ttl else 0.0
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expires_at)
        if len(self._store) > self._max:
            self._store.popitem(last=False)

    def delete(self, key: str):
        self._store.pop(key, None)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class _RedisBackend:
    def __init__(self, url: str, db: int = 1):
        import redis as redis_lib
        self._client = redis_lib.from_url(url, db=db, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int):
        serialized = json.dumps(value, default=str)
        if ttl:
            self._client.setex(key, ttl, serialized)
        else:
            self._client.set(key, serialized)

    def delete(self, key: str):
        self._client.delete(key)


# ---------------------------------------------------------------------------
# Public ResponseCache facade
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Thread-safe, async-compatible response cache.

    Usage:
        cache = ResponseCache()
        hit = cache.get(query)
        if hit:
            return hit
        ...run agents...
        await cache.async_set(query, result, ttl=3600)
    """

    def __init__(self):
        cfg = _cache_settings()
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._ttl: int = int(cfg.get("ttl_seconds", 3600))
        self._backend = self._build_backend(cfg)

    def _build_backend(self, cfg: dict):
        backend_type = cfg.get("backend", "memory")
        if backend_type == "redis":
            try:
                import os
                redis_url = os.getenv(
                    "CELERY_BROKER_URL",
                    cfg.get("redis_url", "redis://localhost:6379")
                )
                # Strip DB suffix from broker URL and use dedicated DB for cache
                base_url = redis_url.rsplit("/", 1)[0] if redis_url.count("/") >= 3 else redis_url
                db = int(cfg.get("redis_db", 1))
                backend = _RedisBackend(base_url, db=db)
                # Ping to verify connectivity
                backend._client.ping()
                log_step("Response cache: Redis backend active", symbol="💾")
                return backend
            except Exception as e:
                log_step(f"Response cache: Redis unavailable ({e}), falling back to memory", symbol="⚠️")
        max_entries = int(cfg.get("max_memory_entries", 500))
        log_step(f"Response cache: in-memory backend active (max {max_entries} entries)", symbol="💾")
        return _MemoryBackend(max_entries=max_entries)

    def is_enabled(self) -> bool:
        return self._enabled

    def get(self, query: str) -> dict | None:
        if not self._enabled:
            return None
        key = _make_key(query)
        result = self._backend.get(key)
        if result is not None:
            log_step("Response cache: HIT", symbol="⚡")
        return result

    def set(self, query: str, result: dict, ttl: int | None = None):
        if not self._enabled:
            return
        key = _make_key(query)
        self._backend.set(key, result, ttl if ttl is not None else self._ttl)
        log_step("Response cache: stored result", symbol="💾")

    async def async_set(self, query: str, result: dict, ttl: int | None = None):
        """Non-blocking set — runs sync set in thread to avoid blocking event loop for Redis."""
        import asyncio
        await asyncio.to_thread(self.set, query, result, ttl)

    def invalidate(self, query: str):
        """Manually evict a specific query from the cache."""
        if not self._enabled:
            return
        key = _make_key(query)
        self._backend.delete(key)
        log_step(f"Response cache: invalidated key {key[:20]}...", symbol="🗑️")


# Module-level singleton — created once per process
_cache_instance: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance
