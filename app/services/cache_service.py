"""
Cache Service — Redis-backed multi-tier caching layer.

Provides three caching tiers to minimise LLM costs:

Tier 1 — **Extraction cache**
  Cache extraction results keyed by ``(document_id, query_hash)``.
  Identical queries for the same document are served instantly.

Tier 2 — **Embedding cache**
  Cache embedding vectors keyed by a hash of the input text.
  Avoids redundant OpenAI embedding calls when the same chunk or query
  text is embedded more than once (common during re-indexing or when
  multiple users query with identical text).

Tier 3 — **Semantic deduplication cache**
  A lightweight fingerprint cache that detects near-duplicate documents
  before they enter the processing pipeline.  Uses the content hash to
  short-circuit ingestion for content that has already been indexed.

All tiers use TTL-based expiration so stale data is evicted
automatically.  The embedding cache uses a longer TTL since embeddings
are deterministic for a given model + text pair.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import CACHE_OPS_TOTAL

logger = get_logger(__name__)
settings = get_settings()


class CacheService:
    """Async Redis cache with JSON serialisation and multi-tier support."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        self._redis = aioredis.from_url(
            self._url,
            decode_responses=True,
            max_connections=20,
        )
        logger.info("cache.connected", url=self._url)

    async def disconnect(self) -> None:
        """Close the Redis connection pool."""
        if self._redis:
            await self._redis.close()
            logger.info("cache.disconnected")

    async def ping(self) -> bool:
        """Health check."""
        if not self._redis:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    # ── Generic Cache Operations ─────────────────────────────────────────

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached JSON value by key."""
        if not self._redis:
            return None
        raw = await self._redis.get(key)
        if raw is None:
            CACHE_OPS_TOTAL.labels(operation="get", result="miss").inc()
            return None
        CACHE_OPS_TOTAL.labels(operation="get", result="hit").inc()
        logger.debug("cache.hit", key=key)
        return json.loads(raw)

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Store a JSON-serialisable value with optional TTL."""
        if not self._redis:
            return
        ttl = ttl or settings.REDIS_CACHE_TTL
        await self._redis.set(key, json.dumps(value), ex=ttl)
        CACHE_OPS_TOTAL.labels(operation="set", result="ok").inc()
        logger.debug("cache.set", key=key, ttl=ttl)

    async def delete(self, key: str) -> None:
        """Remove a cached entry."""
        if not self._redis:
            return
        await self._redis.delete(key)
        CACHE_OPS_TOTAL.labels(operation="delete", result="ok").inc()

    # ── Tier 1: Extraction Cache ─────────────────────────────────────────

    @staticmethod
    def extraction_key(document_id: str, query: str) -> str:
        """
        Deterministic cache key for an extraction result.

        Hash the query to keep keys short and avoid special characters.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"extract:{document_id}:{query_hash}"

    # ── Tier 2: Embedding Cache ──────────────────────────────────────────

    @staticmethod
    def embedding_key(text: str, model: str = "") -> str:
        """
        Cache key for an embedding vector.

        Keyed on ``(model, text_hash)`` so that switching embedding
        models does not serve stale vectors.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:20]
        return f"emb:{model}:{text_hash}"

    async def get_embedding(self, text: str, model: str = "") -> Optional[List[float]]:
        """
        Return a cached embedding vector (as list of floats), or None.

        Embedding TTL is 24 hours by default (deterministic for a given
        model + text pair, so long-lived caching is safe).
        """
        key = self.embedding_key(text, model)
        if not self._redis:
            return None
        raw = await self._redis.get(key)
        if raw is None:
            CACHE_OPS_TOTAL.labels(operation="get_embedding", result="miss").inc()
            return None
        CACHE_OPS_TOTAL.labels(operation="get_embedding", result="hit").inc()
        return json.loads(raw)

    async def set_embedding(
        self,
        text: str,
        vector: List[float],
        model: str = "",
        ttl: int | None = None,
    ) -> None:
        """Cache an embedding vector with a long TTL (default 24 h)."""
        if not self._redis:
            return
        key = self.embedding_key(text, model)
        ttl = ttl or settings.EMBEDDING_CACHE_TTL
        await self._redis.set(key, json.dumps(vector), ex=ttl)
        CACHE_OPS_TOTAL.labels(operation="set_embedding", result="ok").inc()

    # ── Tier 3: Semantic Deduplication ───────────────────────────────────

    @staticmethod
    def dedup_key(content_hash: str) -> str:
        """Cache key for deduplication lookup."""
        return f"dedup:{content_hash}"

    async def check_duplicate(self, content_hash: str) -> Optional[str]:
        """
        Check if content with the given hash has already been processed.

        Returns the existing ``document_id`` if found, else None.
        """
        if not self._redis:
            return None
        raw = await self._redis.get(self.dedup_key(content_hash))
        if raw is None:
            return None
        return raw

    async def mark_processed(
        self,
        content_hash: str,
        document_id: str,
        ttl: int | None = None,
    ) -> None:
        """Record that a content hash has been processed."""
        if not self._redis:
            return
        key = self.dedup_key(content_hash)
        ttl = ttl or 86400 * 7  # 7 days
        await self._redis.set(key, document_id, ex=ttl)

    # ── Bulk Invalidation ────────────────────────────────────────────────

    async def invalidate_document(self, document_id: str) -> int:
        """
        Remove all cached entries related to a document.

        Scans for keys matching the document_id pattern and deletes them.
        Returns the number of keys removed.

        This is useful when a document is re-indexed or deleted —
        stale extraction results must not be served.
        """
        if not self._redis:
            return 0

        pattern = f"extract:{document_id}:*"
        deleted = 0
        async for key in self._redis.scan_iter(match=pattern, count=100):
            await self._redis.delete(key)
            deleted += 1

        if deleted:
            CACHE_OPS_TOTAL.labels(operation="invalidate", result="ok").inc()
            logger.info(
                "cache.invalidate_document",
                document_id=document_id,
                keys_deleted=deleted,
            )
        return deleted

    # ── Cache Stats ──────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics for monitoring dashboards.

        Includes memory usage, key count, and hit/miss rates.
        """
        if not self._redis:
            return {"status": "disconnected"}

        try:
            info = await self._redis.info("memory")
            keyspace = await self._redis.info("keyspace")
            stats = await self._redis.info("stats")

            return {
                "status": "connected",
                "used_memory_mb": round(
                    info.get("used_memory", 0) / (1024 * 1024), 2
                ),
                "peak_memory_mb": round(
                    info.get("used_memory_peak", 0) / (1024 * 1024), 2
                ),
                "total_keys": sum(
                    db.get("keys", 0)
                    for db in keyspace.values()
                    if isinstance(db, dict)
                ),
                "keyspace_hits": stats.get("keyspace_hits", 0),
                "keyspace_misses": stats.get("keyspace_misses", 0),
                "hit_rate": round(
                    stats.get("keyspace_hits", 0)
                    / max(
                        stats.get("keyspace_hits", 0)
                        + stats.get("keyspace_misses", 0),
                        1,
                    )
                    * 100,
                    2,
                ),
                "evicted_keys": stats.get("evicted_keys", 0),
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)[:200]}
