"""
Unit tests for the FAISS VectorStore wrapper.

Tests cover:
  - Index creation and basic properties
  - Adding vectors and retrieving assigned IDs
  - Search (nearest neighbour) with cosine similarity
  - Persistence (save / load)
  - Reset
  - Edge cases (empty index, fewer results than k)
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vector_store import VectorStore


pytestmark = pytest.mark.unit


def _random_vectors(n: int, dim: int = 1536, normalise: bool = True) -> np.ndarray:
    """Helper: generate random L2-normalised float32 vectors."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, dim).astype(np.float32)
    if normalise:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
    return vecs


class TestVectorStore:
    """Test the VectorStore with a temp directory (no shared state)."""

    def test_new_store_has_zero_vectors(self, vector_store):
        assert vector_store.total_vectors == 0

    def test_add_vectors_returns_ids(self, vector_store):
        vecs = _random_vectors(5)
        ids = vector_store.add(vecs)

        assert len(ids) == 5
        assert ids == [0, 1, 2, 3, 4]
        assert vector_store.total_vectors == 5

    def test_add_more_vectors_continues_id_sequence(self, vector_store):
        vecs1 = _random_vectors(3)
        ids1 = vector_store.add(vecs1)

        vecs2 = _random_vectors(2)
        ids2 = vector_store.add(vecs2)

        assert ids1 == [0, 1, 2]
        assert ids2 == [3, 4]
        assert vector_store.total_vectors == 5

    def test_search_returns_nearest_neighbours(self, vector_store):
        # Insert a known vector and search for it
        vecs = _random_vectors(10)
        vector_store.add(vecs)

        # Search with the first vector — should return itself as top result
        query = vecs[0:1]
        results = vector_store.search(query, k=3)

        assert len(results) == 3
        # Top result should be the vector itself (id=0, score≈1.0)
        top_id, top_score = results[0]
        assert top_id == 0
        assert top_score == pytest.approx(1.0, abs=1e-5)

    def test_search_scores_are_descending(self, vector_store):
        vecs = _random_vectors(20)
        vector_store.add(vecs)

        query = vecs[5:6]
        results = vector_store.search(query, k=5)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_index(self, vector_store):
        query = _random_vectors(1)
        results = vector_store.search(query, k=5)
        assert results == []

    def test_search_fewer_vectors_than_k(self, vector_store):
        vecs = _random_vectors(2)
        vector_store.add(vecs)

        query = vecs[0:1]
        results = vector_store.search(query, k=10)

        # Should return only 2 results even though k=10
        assert len(results) == 2

    def test_save_and_reload(self, tmp_faiss_dir):
        # Create, add, save
        store1 = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)
        vecs = _random_vectors(5)
        store1.add(vecs)
        store1.save()

        # Reload from same directory
        store2 = VectorStore(dimension=1536, index_dir=tmp_faiss_dir)
        assert store2.total_vectors == 5

        # Verify search still works after reload
        results = store2.search(vecs[0:1], k=1)
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_reset_clears_all_vectors(self, vector_store):
        vecs = _random_vectors(10)
        vector_store.add(vecs)
        assert vector_store.total_vectors == 10

        vector_store.reset()
        assert vector_store.total_vectors == 0

    def test_add_after_reset(self, vector_store):
        vecs = _random_vectors(3)
        vector_store.add(vecs)
        vector_store.reset()

        new_vecs = _random_vectors(2)
        ids = vector_store.add(new_vecs)

        # IDs start from 0 again after reset
        assert ids == [0, 1]
        assert vector_store.total_vectors == 2

    def test_custom_dimension(self, tmp_faiss_dir):
        store = VectorStore(dimension=384, index_dir=tmp_faiss_dir)
        vecs = _random_vectors(3, dim=384)
        ids = store.add(vecs)
        assert len(ids) == 3

        results = store.search(vecs[0:1], k=1)
        assert results[0][0] == 0
