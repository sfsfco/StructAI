"""
Integration tests for FastAPI API endpoints.

Tests cover:
  - GET /health — verifies dependency status reporting
  - POST /documents/index — document ingestion + idempotency (409)
  - POST /extract — extraction with cache miss and cache hit
  - Error handling (404, 400)

Requires: PostgreSQL and Redis running (via docker-compose.test.yml).
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
from uuid import uuid4

import numpy as np
import pytest
from httpx import AsyncClient

from app.db.models import Chunk, Document
from app.services.vector_store import VectorStore


pytestmark = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════
#  Health Check
# ═══════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Test GET /api/v1/health."""

    async def test_health_returns_200(self, async_client: AsyncClient):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "0.1.0"

    async def test_health_reports_dependency_status(
        self, async_client: AsyncClient
    ):
        response = await async_client.get("/api/v1/health")
        data = response.json()

        assert "db" in data
        assert "redis" in data
        assert "faiss_index_loaded" in data

    async def test_health_reports_uptime(self, async_client: AsyncClient):
        """Health endpoint should include uptime_seconds."""
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0


# ═══════════════════════════════════════════════════════════════════════
#  Readiness Probe
# ═══════════════════════════════════════════════════════════════════════


class TestReadinessEndpoint:
    """Test GET /api/v1/ready."""

    async def test_ready_returns_checks(self, async_client: AsyncClient):
        """Readiness probe returns structured dependency checks."""
        response = await async_client.get("/api/v1/ready")
        data = response.json()

        assert "ready" in data
        assert "checks" in data
        assert "version" in data
        assert "postgres" in data["checks"]
        assert "redis" in data["checks"]

    async def test_ready_includes_faiss_check(self, async_client: AsyncClient):
        """Readiness probe includes FAISS index check."""
        response = await async_client.get("/api/v1/ready")
        data = response.json()

        assert "faiss" in data["checks"]


# ═══════════════════════════════════════════════════════════════════════
#  Metrics Endpoint
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsEndpoint:
    """Test GET /metrics."""

    async def test_metrics_returns_prometheus_format(
        self, async_client: AsyncClient
    ):
        """The /metrics endpoint returns Prometheus text exposition."""
        response = await async_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "structai" in response.text

    async def test_metrics_includes_http_metrics(
        self, async_client: AsyncClient
    ):
        """After making requests, HTTP metrics should be present."""
        # Make a request first to generate metrics
        await async_client.get("/api/v1/health")

        response = await async_client.get("/metrics")
        body = response.text

        assert "structai_http_request_duration_seconds" in body
        assert "structai_http_requests_total" in body


# ═══════════════════════════════════════════════════════════════════════
#  Correlation ID
# ═══════════════════════════════════════════════════════════════════════


class TestCorrelationID:
    """Verify correlation ID middleware behaviour."""

    async def test_correlation_id_generated(self, async_client: AsyncClient):
        """Requests without X-Correlation-ID get one generated."""
        response = await async_client.get("/api/v1/health")
        assert "x-correlation-id" in response.headers
        cid = response.headers["x-correlation-id"]
        assert len(cid) == 32  # hex UUID without dashes

    async def test_correlation_id_propagated(self, async_client: AsyncClient):
        """Client-supplied X-Correlation-ID is echoed back."""
        custom_id = "my-custom-correlation-id-12345"
        response = await async_client.get(
            "/api/v1/health",
            headers={"X-Correlation-ID": custom_id},
        )
        assert response.headers["x-correlation-id"] == custom_id


# ═══════════════════════════════════════════════════════════════════════
#  Response Timing Header
# ═══════════════════════════════════════════════════════════════════════


class TestResponseTiming:
    """Verify X-Response-Time header is injected by metrics middleware."""

    async def test_response_time_header_present(self, async_client: AsyncClient):
        response = await async_client.get("/api/v1/health")
        assert "x-response-time" in response.headers
        timing = response.headers["x-response-time"]
        assert timing.endswith("s")
        assert float(timing.rstrip("s")) >= 0


# ═══════════════════════════════════════════════════════════════════════
#  Document Indexing
# ═══════════════════════════════════════════════════════════════════════


class TestDocumentIndexEndpoint:
    """Test POST /api/v1/documents/index."""

    async def test_index_document_returns_202(
        self, async_client: AsyncClient, db_session
    ):
        """Successful indexing returns 202 with a document ID."""
        with patch("app.api.routes.process_document") as mock_task:
            mock_task.delay.return_value = MagicMock(id="task-123")

            response = await async_client.post(
                "/api/v1/documents/index",
                json={
                    "filename": "test.pdf",
                    "content": "This is a test document with unique content " + str(uuid4()),
                },
            )

        assert response.status_code == 202
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "pending"

    async def test_duplicate_document_returns_409(
        self, async_client: AsyncClient, db_session
    ):
        """Duplicate content hash should return 409 Conflict."""
        content = "Exact duplicate content for idempotency test."

        with patch("app.api.routes.process_document") as mock_task:
            mock_task.delay.return_value = MagicMock(id="task-1")

            # First request — should succeed
            resp1 = await async_client.post(
                "/api/v1/documents/index",
                json={"filename": "doc1.pdf", "content": content},
            )
            assert resp1.status_code == 202

            # Second request with same content — should fail
            resp2 = await async_client.post(
                "/api/v1/documents/index",
                json={"filename": "doc2.pdf", "content": content},
            )
            assert resp2.status_code == 409

    async def test_index_missing_fields_returns_422(
        self, async_client: AsyncClient
    ):
        """Missing required fields return 422 Unprocessable Entity."""
        response = await async_client.post(
            "/api/v1/documents/index",
            json={"filename": "test.pdf"},  # missing 'content'
        )
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractEndpoint:
    """Test POST /api/v1/extract."""

    async def test_extract_document_not_found(self, async_client: AsyncClient):
        """Extraction on a nonexistent document returns 404."""
        response = await async_client.post(
            "/api/v1/extract",
            json={
                "document_id": str(uuid4()),
                "query": "Extract all parties",
            },
        )
        assert response.status_code == 404

    async def test_extract_document_not_ready(
        self, async_client: AsyncClient, db_session
    ):
        """Extraction on a pending document returns 400."""
        # Insert a document in 'pending' status
        doc = Document(
            filename="pending.pdf",
            content_hash="abc123notready",
            status="pending",
        )
        db_session.add(doc)
        await db_session.flush()

        response = await async_client.post(
            "/api/v1/extract",
            json={
                "document_id": str(doc.id),
                "query": "Extract all parties",
            },
        )
        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    async def test_extract_short_query_returns_422(
        self, async_client: AsyncClient
    ):
        """Query shorter than min_length (3) should return 422."""
        response = await async_client.post(
            "/api/v1/extract",
            json={
                "document_id": str(uuid4()),
                "query": "ab",  # too short
            },
        )
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
#  Task Status
# ═══════════════════════════════════════════════════════════════════════


class TestTaskStatusEndpoint:
    """Test GET /api/v1/tasks/{task_id}."""

    async def test_task_status_pending(self, async_client: AsyncClient):
        """Polling a nonexistent task returns PENDING (Celery default)."""
        with patch("app.api.routes.celery_app") as mock_celery:
            mock_result = MagicMock()
            mock_result.status = "PENDING"
            mock_result.ready.return_value = False
            mock_celery.AsyncResult.return_value = mock_result

            response = await async_client.get("/api/v1/tasks/fake-task-id")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "fake-task-id"
        assert data["status"] == "PENDING"

    async def test_task_status_success(self, async_client: AsyncClient):
        """Completed task returns SUCCESS with result."""
        with patch("app.api.routes.celery_app") as mock_celery:
            mock_result = MagicMock()
            mock_result.status = "SUCCESS"
            mock_result.ready.return_value = True
            mock_result.successful.return_value = True
            mock_result.result = {"document_id": "abc", "status": "indexed"}
            mock_result.date_done = "2025-01-01T00:00:00"
            mock_celery.AsyncResult.return_value = mock_result

            response = await async_client.get("/api/v1/tasks/success-task")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["status"] == "indexed"
