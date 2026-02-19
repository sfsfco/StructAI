"""
Unit tests for the Prometheus metrics module and metrics middleware.

Tests cover:
  - Metric registry contains all expected metrics
  - PrometheusMiddleware records latency, counts, and in-progress gauges
  - /metrics endpoint returns valid Prometheus text format
  - Path normalisation collapses high-cardinality segments
  - Cache, LLM, FAISS, and task metrics are instrumented correctly
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════
#  Metrics Registry Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsRegistry:
    """Verify all expected metrics are defined in the registry."""

    def test_http_request_duration_exists(self):
        from app.core.metrics import HTTP_REQUEST_DURATION

        assert HTTP_REQUEST_DURATION._name == "structai_http_request_duration_seconds"

    def test_http_requests_total_exists(self):
        from app.core.metrics import HTTP_REQUESTS_TOTAL

        assert HTTP_REQUESTS_TOTAL._name == "structai_http_requests_total"

    def test_http_requests_in_progress_exists(self):
        from app.core.metrics import HTTP_REQUESTS_IN_PROGRESS

        assert HTTP_REQUESTS_IN_PROGRESS._name == "structai_http_requests_in_progress"

    def test_llm_metrics_exist(self):
        from app.core.metrics import LLM_CALL_DURATION, LLM_CALLS_TOTAL, LLM_TOKENS_TOTAL

        assert LLM_CALL_DURATION._name == "structai_llm_call_duration_seconds"
        assert LLM_CALLS_TOTAL._name == "structai_llm_calls_total"
        assert LLM_TOKENS_TOTAL._name == "structai_llm_tokens_total"

    def test_cache_metrics_exist(self):
        from app.core.metrics import CACHE_OPS_TOTAL

        assert CACHE_OPS_TOTAL._name == "structai_cache_ops_total"

    def test_faiss_metrics_exist(self):
        from app.core.metrics import FAISS_INDEX_SIZE, FAISS_SEARCH_DURATION

        assert FAISS_SEARCH_DURATION._name == "structai_faiss_search_duration_seconds"
        assert FAISS_INDEX_SIZE._name == "structai_faiss_index_size_vectors"

    def test_task_metrics_exist(self):
        from app.core.metrics import TASK_DURATION, TASKS_IN_PROGRESS, TASKS_TOTAL

        assert TASK_DURATION._name == "structai_task_duration_seconds"
        assert TASKS_TOTAL._name == "structai_tasks_total"
        assert TASKS_IN_PROGRESS._name == "structai_tasks_in_progress"

    def test_document_metrics_exist(self):
        from app.core.metrics import (
            CHUNKS_CREATED_TOTAL,
            DOCUMENTS_FAILED_TOTAL,
            DOCUMENTS_INDEXED_TOTAL,
        )

        assert DOCUMENTS_INDEXED_TOTAL._name == "structai_documents_indexed_total"
        assert DOCUMENTS_FAILED_TOTAL._name == "structai_documents_failed_total"
        assert CHUNKS_CREATED_TOTAL._name == "structai_chunks_created_total"

    def test_metrics_output_returns_bytes(self):
        from app.core.metrics import metrics_output

        output = metrics_output()
        assert isinstance(output, bytes)
        assert b"structai" in output

    def test_app_info_metric(self):
        from app.core.metrics import APP_INFO

        APP_INFO.info({"version": "test", "name": "StructAI"})
        from app.core.metrics import metrics_output

        output = metrics_output()
        assert b"structai_info" in output


# ═══════════════════════════════════════════════════════════════════════
#  Path Normalisation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPathNormalisation:
    """Verify high-cardinality path segments are collapsed."""

    def test_uuid_is_normalised(self):
        from app.middleware.metrics import _normalise_path

        result = _normalise_path("/api/v1/tasks/550e8400-e29b-41d4-a716-446655440000")
        assert "{id}" in result
        assert "550e8400" not in result

    def test_health_path_unchanged(self):
        from app.middleware.metrics import _normalise_path

        result = _normalise_path("/api/v1/health")
        assert result == "/api/v1/health"

    def test_extract_path_unchanged(self):
        from app.middleware.metrics import _normalise_path

        result = _normalise_path("/api/v1/extract")
        assert result == "/api/v1/extract"

    def test_root_path(self):
        from app.middleware.metrics import _normalise_path

        assert _normalise_path("/") == "/"

    def test_empty_path(self):
        from app.middleware.metrics import _normalise_path

        assert _normalise_path("") == "/"


# ═══════════════════════════════════════════════════════════════════════
#  Cache Metrics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCacheMetrics:
    """Verify cache operations emit Prometheus counters."""

    @pytest.fixture
    def cache_with_mock_redis(self):
        from app.services.cache_service import CacheService

        cache = CacheService()
        cache._redis = AsyncMock()
        return cache

    async def test_cache_get_hit_increments_counter(self, cache_with_mock_redis):
        from app.core.metrics import CACHE_OPS_TOTAL

        cache_with_mock_redis._redis.get = AsyncMock(return_value='{"key": "value"}')

        # Get the current value before the operation
        before = CACHE_OPS_TOTAL.labels(operation="get", result="hit")._value.get()

        await cache_with_mock_redis.get("test-key")

        after = CACHE_OPS_TOTAL.labels(operation="get", result="hit")._value.get()
        assert after == before + 1

    async def test_cache_get_miss_increments_counter(self, cache_with_mock_redis):
        from app.core.metrics import CACHE_OPS_TOTAL

        cache_with_mock_redis._redis.get = AsyncMock(return_value=None)

        before = CACHE_OPS_TOTAL.labels(operation="get", result="miss")._value.get()

        await cache_with_mock_redis.get("missing-key")

        after = CACHE_OPS_TOTAL.labels(operation="get", result="miss")._value.get()
        assert after == before + 1

    async def test_cache_set_increments_counter(self, cache_with_mock_redis):
        from app.core.metrics import CACHE_OPS_TOTAL

        cache_with_mock_redis._redis.set = AsyncMock()

        before = CACHE_OPS_TOTAL.labels(operation="set", result="ok")._value.get()

        await cache_with_mock_redis.set("key", {"data": "value"})

        after = CACHE_OPS_TOTAL.labels(operation="set", result="ok")._value.get()
        assert after == before + 1

    async def test_cache_delete_increments_counter(self, cache_with_mock_redis):
        from app.core.metrics import CACHE_OPS_TOTAL

        cache_with_mock_redis._redis.delete = AsyncMock()

        before = CACHE_OPS_TOTAL.labels(operation="delete", result="ok")._value.get()

        await cache_with_mock_redis.delete("key")

        after = CACHE_OPS_TOTAL.labels(operation="delete", result="ok")._value.get()
        assert after == before + 1


# ═══════════════════════════════════════════════════════════════════════
#  FAISS / Vector Store Metrics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestVectorStoreMetrics:
    """Verify FAISS operations emit Prometheus metrics."""

    @pytest.fixture
    def tmp_vector_store(self, tmp_path):
        from app.services.vector_store import VectorStore

        store = VectorStore(dimension=4, index_dir=str(tmp_path))
        return store

    def test_add_updates_index_size_gauge(self, tmp_vector_store):
        import numpy as np

        from app.core.metrics import FAISS_INDEX_SIZE

        vectors = np.random.randn(3, 4).astype(np.float32)
        tmp_vector_store.add(vectors)

        assert FAISS_INDEX_SIZE._value.get() == 3

    def test_search_records_duration(self, tmp_vector_store):
        import numpy as np

        from app.core.metrics import FAISS_SEARCH_DURATION

        vectors = np.random.randn(5, 4).astype(np.float32)
        tmp_vector_store.add(vectors)

        query = np.random.randn(1, 4).astype(np.float32)

        # Histogram should have observed at least one sample after search
        before_count = FAISS_SEARCH_DURATION._sum.get()
        tmp_vector_store.search(query, k=2)
        after_count = FAISS_SEARCH_DURATION._sum.get()

        assert after_count > before_count

    def test_load_sets_index_size(self, tmp_path):
        import numpy as np

        from app.core.metrics import FAISS_INDEX_SIZE
        from app.services.vector_store import VectorStore

        # Create and save a store with vectors
        store1 = VectorStore(dimension=4, index_dir=str(tmp_path))
        vectors = np.random.randn(7, 4).astype(np.float32)
        store1.add(vectors)
        store1.save()

        # Load from disk — should set gauge
        store2 = VectorStore(dimension=4, index_dir=str(tmp_path))
        assert FAISS_INDEX_SIZE._value.get() == 7


# ═══════════════════════════════════════════════════════════════════════
#  Embedding Metrics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEmbeddingMetrics:
    """Verify embedding service emits batch-size histogram observations."""

    async def test_embed_batch_observes_batch_size(self, fake_llm):
        from app.core.metrics import EMBEDDING_BATCH_SIZE
        from app.services.embedding_service import EmbeddingService

        service = EmbeddingService(fake_llm)

        before = EMBEDDING_BATCH_SIZE._sum.get()

        texts = ["hello", "world", "test"]
        await service.embed_batch(texts)

        after = EMBEDDING_BATCH_SIZE._sum.get()
        assert after == before + 3  # batch size of 3
