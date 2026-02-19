"""
Centralised Prometheus metrics registry.

All application metrics are defined here in one place so dashboards,
alerts, and exporters have a single source of truth for metric names
and label schemas.

Metric naming follows the Prometheus convention:
  <namespace>_<subsystem>_<name>_<unit>
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    multiprocess,
)

# ── Registry ─────────────────────────────────────────────────────────────
# Using the default global registry so that prometheus_client middlewares
# and instrumentators work out of the box.  For multiprocess deployments
# (gunicorn pre-fork) swap to a custom registry + multiprocess collector.

REGISTRY = CollectorRegistry(auto_describe=True)

# ── App Info ─────────────────────────────────────────────────────────────

APP_INFO = Info(
    "structai",
    "StructAI application metadata",
    registry=REGISTRY,
)

# ── HTTP Metrics ─────────────────────────────────────────────────────────

HTTP_REQUEST_DURATION = Histogram(
    "structai_http_request_duration_seconds",
    "Histogram of HTTP request latencies in seconds",
    labelnames=["method", "path", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

HTTP_REQUESTS_TOTAL = Counter(
    "structai_http_requests_total",
    "Total number of HTTP requests",
    labelnames=["method", "path", "status_code"],
    registry=REGISTRY,
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "structai_http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    labelnames=["method"],
    registry=REGISTRY,
)

# ── LLM Metrics ─────────────────────────────────────────────────────────

LLM_CALL_DURATION = Histogram(
    "structai_llm_call_duration_seconds",
    "Histogram of LLM API call latencies in seconds",
    labelnames=["operation", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

LLM_CALLS_TOTAL = Counter(
    "structai_llm_calls_total",
    "Total number of LLM API calls",
    labelnames=["operation", "model", "status"],
    registry=REGISTRY,
)

LLM_TOKENS_TOTAL = Counter(
    "structai_llm_tokens_total",
    "Total tokens consumed by LLM calls",
    labelnames=["model", "type"],  # type: prompt | completion
    registry=REGISTRY,
)

# ── Embedding Metrics ───────────────────────────────────────────────────

EMBEDDING_BATCH_SIZE = Histogram(
    "structai_embedding_batch_size",
    "Number of texts per embedding batch call",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500),
    registry=REGISTRY,
)

# ── Cache Metrics ────────────────────────────────────────────────────────

CACHE_OPS_TOTAL = Counter(
    "structai_cache_ops_total",
    "Total cache operations",
    labelnames=["operation", "result"],  # operation: get|set|delete, result: hit|miss|ok
    registry=REGISTRY,
)

# ── FAISS / Vector Store Metrics ─────────────────────────────────────────

FAISS_SEARCH_DURATION = Histogram(
    "structai_faiss_search_duration_seconds",
    "Histogram of FAISS search latencies in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    registry=REGISTRY,
)

FAISS_INDEX_SIZE = Gauge(
    "structai_faiss_index_size_vectors",
    "Number of vectors currently in the FAISS index",
    registry=REGISTRY,
)

# ── Celery / Worker Metrics ──────────────────────────────────────────────

TASK_DURATION = Histogram(
    "structai_task_duration_seconds",
    "Histogram of background task durations",
    labelnames=["task_name", "status"],
    buckets=(0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)

TASKS_TOTAL = Counter(
    "structai_tasks_total",
    "Total number of background tasks executed",
    labelnames=["task_name", "status"],  # status: success | failure | retry
    registry=REGISTRY,
)

TASKS_IN_PROGRESS = Gauge(
    "structai_tasks_in_progress",
    "Number of background tasks currently being processed",
    labelnames=["task_name"],
    registry=REGISTRY,
)

# ── Document Processing Metrics ──────────────────────────────────────────

DOCUMENTS_INDEXED_TOTAL = Counter(
    "structai_documents_indexed_total",
    "Total documents successfully indexed",
    registry=REGISTRY,
)

DOCUMENTS_FAILED_TOTAL = Counter(
    "structai_documents_failed_total",
    "Total documents that failed indexing",
    registry=REGISTRY,
)

CHUNKS_CREATED_TOTAL = Counter(
    "structai_chunks_created_total",
    "Total number of chunks created during document processing",
    registry=REGISTRY,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def metrics_output() -> bytes:
    """Generate the Prometheus text exposition format."""
    return generate_latest(REGISTRY)
