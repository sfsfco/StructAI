"""
Unit tests for the CacheService (Redis cache layer).

Tests cover:
  - get / set / delete operations (mocked Redis)
  - JSON serialisation round-trip
  - TTL handling
  - Cache key generation (extraction_key)
  - Graceful behaviour when Redis is disconnected
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.cache_service import CacheService


pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════
#  Key Generation
# ═══════════════════════════════════════════════════════════════════════


class TestCacheKeyGeneration:
    """Test the static key-builder methods."""

    def test_extraction_key_format(self):
        key = CacheService.extraction_key("doc-123", "extract parties")
        assert key.startswith("extract:doc-123:")
        # Hash part should be 16 hex chars
        hash_part = key.split(":")[-1]
        assert len(hash_part) == 16

    def test_same_inputs_same_key(self):
        k1 = CacheService.extraction_key("doc-1", "query A")
        k2 = CacheService.extraction_key("doc-1", "query A")
        assert k1 == k2

    def test_different_queries_different_keys(self):
        k1 = CacheService.extraction_key("doc-1", "query A")
        k2 = CacheService.extraction_key("doc-1", "query B")
        assert k1 != k2

    def test_different_docs_different_keys(self):
        k1 = CacheService.extraction_key("doc-1", "query")
        k2 = CacheService.extraction_key("doc-2", "query")
        assert k1 != k2


# ═══════════════════════════════════════════════════════════════════════
#  Cache Operations (mocked Redis)
# ═══════════════════════════════════════════════════════════════════════


class TestCacheServiceOperations:
    """Test get/set/delete with a mocked async Redis client."""

    @pytest.fixture
    def cache_with_mock_redis(self):
        """CacheService with an injected mock Redis instance."""
        cache = CacheService(redis_url="redis://fake:6379/0")
        mock_redis = AsyncMock()
        cache._redis = mock_redis
        return cache, mock_redis

    async def test_get_returns_parsed_json(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis
        mock_redis.get.return_value = json.dumps({"result": "ok"})

        value = await cache.get("test-key")

        assert value == {"result": "ok"}
        mock_redis.get.assert_awaited_once_with("test-key")

    async def test_get_returns_none_on_miss(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis
        mock_redis.get.return_value = None

        value = await cache.get("missing-key")
        assert value is None

    async def test_set_stores_json(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis
        payload = {"key": "value", "count": 42}

        await cache.set("test-key", payload, ttl=600)

        mock_redis.set.assert_awaited_once_with(
            "test-key", json.dumps(payload), ex=600
        )

    async def test_set_uses_default_ttl(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis

        await cache.set("key", {"data": True})

        # Should use settings.REDIS_CACHE_TTL (3600 by default)
        call_args = mock_redis.set.call_args
        assert call_args.kwargs.get("ex") or call_args[1].get("ex") or call_args[0][2]

    async def test_delete_removes_key(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis

        await cache.delete("old-key")
        mock_redis.delete.assert_awaited_once_with("old-key")

    async def test_ping_returns_true(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis
        mock_redis.ping.return_value = True

        assert await cache.ping() is True

    async def test_ping_returns_false_on_error(self, cache_with_mock_redis):
        cache, mock_redis = cache_with_mock_redis
        mock_redis.ping.side_effect = ConnectionError("down")

        assert await cache.ping() is False


# ═══════════════════════════════════════════════════════════════════════
#  Disconnected State
# ═══════════════════════════════════════════════════════════════════════


class TestCacheServiceDisconnected:
    """Verify graceful degradation when Redis is not connected."""

    async def test_get_returns_none_when_no_redis(self):
        cache = CacheService()
        # _redis is None (never connected)
        assert await cache.get("any-key") is None

    async def test_set_is_noop_when_no_redis(self):
        cache = CacheService()
        # Should not raise
        await cache.set("key", {"data": True})

    async def test_delete_is_noop_when_no_redis(self):
        cache = CacheService()
        await cache.delete("key")

    async def test_ping_returns_false_when_no_redis(self):
        cache = CacheService()
        assert await cache.ping() is False
