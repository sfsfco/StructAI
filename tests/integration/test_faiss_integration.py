"""
Integration tests for FAISS indexing + retrieval combined with
the EmbeddingService.

Tests cover:
  - Embedding text → adding to FAISS → searching with a query
  - Verifying that semantically similar texts rank higher
  - Round-trip persistence (save + reload + search)

These tests use the FakeLLMClient so no real OpenAI calls are made,
but the full FAISS pipeline is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from tests.conftest import FakeLLMClient


pytestmark = pytest.mark.integration


class TestFAISSIntegration:
    """Test the embedding → index → search pipeline end-to-end."""

    async def test_embed_and_search_returns_correct_id(self, tmp_faiss_dir):
        llm = FakeLLMClient()
        embedder = EmbeddingService(llm)
        store = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)

        # Embed and index several chunks
        texts = [
            "Alice agrees to pay Bob $1000.",
            "The agreement is effective from January 1, 2025.",
            "Payment shall be made within 30 days of invoice.",
        ]
        arr = await embedder.embed_batch(texts)

        import faiss as faiss_lib

        faiss_lib.normalize_L2(arr)
        ids = store.add(arr)
        assert len(ids) == 3

        # Search for a query similar to the first chunk
        query_vec = await embedder.embed_text(texts[0])
        query_arr = np.array([query_vec], dtype=np.float32)
        faiss_lib.normalize_L2(query_arr)

        results = store.search(query_arr, k=3)

        # The first chunk should be the top result (exact match)
        assert results[0][0] == 0  # FAISS id for texts[0]
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    async def test_search_after_save_and_reload(self, tmp_faiss_dir):
        llm = FakeLLMClient()
        embedder = EmbeddingService(llm)

        # Build and save
        store1 = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)
        texts = ["Document about contracts.", "Document about payments."]
        arr = await embedder.embed_batch(texts)

        import faiss as faiss_lib

        faiss_lib.normalize_L2(arr)
        store1.add(arr)
        store1.save()

        # Reload and search
        store2 = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)
        assert store2.total_vectors == 2

        query_vec = await embedder.embed_text(texts[1])
        query_arr = np.array([query_vec], dtype=np.float32)
        faiss_lib.normalize_L2(query_arr)

        results = store2.search(query_arr, k=2)
        assert results[0][0] == 1  # second document should be top match

    async def test_batch_embedding_shape_matches_faiss(self, tmp_faiss_dir):
        llm = FakeLLMClient()
        embedder = EmbeddingService(llm)
        store = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)

        texts = [f"chunk number {i}" for i in range(20)]
        arr = await embedder.embed_batch(texts)

        import faiss as faiss_lib

        faiss_lib.normalize_L2(arr)

        assert arr.shape == (20, 1536)
        assert arr.dtype == np.float32

        ids = store.add(arr)
        assert len(ids) == 20
        assert store.total_vectors == 20


class TestRedisCache:
    """Test CacheService with a live Redis instance."""

    async def test_set_and_get_round_trip(self, redis_cache):
        payload = {"parties": ["Alice", "Bob"], "amount": 1000}
        await redis_cache.set("test:roundtrip", payload, ttl=60)

        result = await redis_cache.get("test:roundtrip")
        assert result == payload

    async def test_get_missing_key_returns_none(self, redis_cache):
        result = await redis_cache.get("test:nonexistent")
        assert result is None

    async def test_delete_removes_key(self, redis_cache):
        await redis_cache.set("test:delete", {"data": True}, ttl=60)
        await redis_cache.delete("test:delete")

        result = await redis_cache.get("test:delete")
        assert result is None

    async def test_ping_succeeds(self, redis_cache):
        assert await redis_cache.ping() is True

    async def test_ttl_expiration(self, redis_cache):
        """Set a key with 1-second TTL and verify it expires."""
        import asyncio

        await redis_cache.set("test:ttl", {"temp": True}, ttl=1)

        # Key should exist immediately
        assert await redis_cache.get("test:ttl") is not None

        # Wait for expiration
        await asyncio.sleep(1.5)
        assert await redis_cache.get("test:ttl") is None
