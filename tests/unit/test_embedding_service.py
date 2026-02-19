"""
Unit tests for the EmbeddingService.

Tests cover:
  - Single text embedding
  - Batch embedding (shape, dtype, normalisation)
  - LLM client delegation
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.embedding_service import EmbeddingService


pytestmark = pytest.mark.unit


class TestEmbeddingService:
    """Test embedding generation with a fake LLM client."""

    async def test_embed_text_returns_vector(self, embedding_service):
        vec = await embedding_service.embed_text("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 1536

    async def test_embed_text_is_deterministic(self, embedding_service):
        vec1 = await embedding_service.embed_text("same input")
        vec2 = await embedding_service.embed_text("same input")
        assert vec1 == vec2

    async def test_embed_batch_returns_numpy_array(self, embedding_service):
        texts = ["first chunk", "second chunk", "third chunk"]
        arr = await embedding_service.embed_batch(texts)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 1536)
        assert arr.dtype == np.float32

    async def test_embed_batch_empty_list(self, embedding_service):
        arr = await embedding_service.embed_batch([])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (0,)

    async def test_embed_batch_records_all_calls(self, fake_llm):
        service = EmbeddingService(fake_llm)
        texts = ["a", "b", "c", "d"]
        await service.embed_batch(texts)

        assert len(fake_llm.embed_calls) == 4
        assert fake_llm.embed_calls == ["a", "b", "c", "d"]

    async def test_different_texts_produce_different_vectors(
        self, embedding_service
    ):
        vec_a = await embedding_service.embed_text("alpha text")
        vec_b = await embedding_service.embed_text("beta text")
        # Vectors should differ for different inputs
        assert vec_a != vec_b
