# StructAI – Scalable AI Platform for Structured Data Extraction

**StructAI** is a production-grade backend platform that transforms unstructured text (reports, contracts, emails, notes) into structured, machine-readable data. It combines **OpenAI-powered LLMs** with a **RAG pipeline** (Retrieval-Augmented Generation) for reliable information extraction, built on modern, scalable backend architecture.

> Ingestion → Chunking → Embedding → FAISS Retrieval → LLM Extraction → Structured JSON

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start (Docker)](#-quick-start-docker)
- [Configuration Reference](#️-configuration-reference)
- [API Reference](#-api-reference)
- [Processing Pipeline](#-processing-pipeline)
- [Observability](#-observability)
- [Testing](#-testing)
- [Scaling & Production](#-scaling--production)
- [Kubernetes Deployment](#️-kubernetes-deployment)
- [Security](#-security)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🧠 **AI-Powered Extraction** – Structured data extraction via OpenAI LLMs with schema-guided prompts
- 🔎 **Semantic Retrieval** – FAISS vector search finds the most relevant text chunks for each query
- ⚡ **Async Pipeline** – Celery background workers handle chunking, embedding, and indexing
- 🧰 **Multi-Tier Caching** – Redis caches extractions, embeddings, and deduplication lookups
- 🛡️ **Backpressure Protection** – In-flight and queue-depth limits prevent overload
- 📊 **Observability** – Prometheus metrics, Grafana dashboards, structured JSON logging, correlation IDs
- 📈 **Horizontally Scalable** – Stateless API + independently scalable workers
- 🔌 **Pluggable Vector Store** – Abstract interface allows swapping FAISS for Pinecone/Qdrant/Weaviate
- 🐳 **Production Docker Setup** – Nginx load balancer, Gunicorn multi-worker, resource limits
- ☸️ **Kubernetes Ready** – Full K8s manifests with HPA, probes, and PVCs

---

## 🏗️ Architecture

```text
                         ┌──────────────┐
                         │    Client    │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │    Nginx     │  (LB + rate limit)
                         └──────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
          ┌──────▼──────┐┌─────▼──────┐┌──────▼──────┐
          │  API Pod 1  ││  API Pod 2 ││  API Pod N  │
          │  (gunicorn) ││  (gunicorn)││  (gunicorn) │
          └──────┬──────┘└─────┬──────┘└──────┬──────┘
                 └──────────────┼──────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
  ┌──────▼──────┐       ┌──────▼──────┐       ┌───────▼────────┐
  │   Redis     │       │ PostgreSQL  │       │ Celery Workers │
  │ cache+broker│       │  metadata   │       │ indexing: 2-8  │
  └─────────────┘       └─────────────┘       │ default:  1-4  │
                                              └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  FAISS Index   │
                                              │ (shared volume)│
                                              └────────────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  OpenAI API    │
                                              │  (external)    │
                                              └────────────────┘
```

**Design Principles:**

| Principle | How |
|-----------|-----|
| Stateless API | No in-process state → scale horizontally |
| Service abstraction | LLM + vector store behind interfaces → swap providers |
| Async processing | Heavy work offloaded to Celery → API stays responsive |
| Defence-in-depth | Rate limiting at Nginx + app + backpressure middleware |
| Repository pattern | DB access abstracted → testable + clean domain logic |
| Dependency injection | FastAPI `Depends()` → services swappable in tests |

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API Framework | FastAPI 0.115 | Async REST API with OpenAPI docs |
| ASGI Server | Gunicorn + Uvicorn | Multi-worker production server |
| LLM Provider | OpenAI API | Chat completions + embeddings |
| Vector Store | FAISS (faiss-cpu) | In-process cosine similarity search |
| Cache & Broker | Redis 7 | Result cache + Celery message broker |
| Database | PostgreSQL 15 | Document metadata, chunks, extractions |
| Background Jobs | Celery 5.4 | Task queue with retries + dead-letter |
| Monitoring | Prometheus + Grafana | Metrics collection + dashboards |
| Task Dashboard | Flower | Celery monitoring UI |
| Load Balancer | Nginx | Reverse proxy, rate limiting, compression |
| Logging | structlog | Structured JSON logs with correlation IDs |
| Config | pydantic-settings | Type-safe env-based configuration |
| Containerisation | Docker + Compose | Local dev + production deployment |
| Orchestration | Kubernetes (optional) | HPA, rolling updates, probes |
| Testing | pytest + httpx | Unit, integration, E2E with async support |

---

## 📦 Project Structure

```text
StructAI/
├── app/
│   ├── main.py                    # FastAPI application entrypoint
│   ├── core/
│   │   ├── config.py              # Environment configuration (pydantic-settings)
│   │   ├── gunicorn_conf.py       # Gunicorn production config
│   │   ├── logging.py             # Structured logging + correlation IDs
│   │   └── metrics.py             # Prometheus metrics registry
│   ├── api/
│   │   └── routes.py              # REST endpoints
│   ├── middleware/
│   │   ├── backpressure.py        # In-flight + queue-depth load shedding
│   │   ├── correlation.py         # X-Correlation-ID propagation
│   │   ├── metrics.py             # Per-request Prometheus metrics
│   │   └── rate_limit.py          # SlowAPI per-IP rate limiting
│   ├── services/
│   │   ├── llm_client.py          # OpenAI client abstraction (BaseLLMClient)
│   │   ├── langextract_service.py # Structured extraction via LLM
│   │   ├── embedding_service.py   # Text → vector embeddings
│   │   ├── vector_store.py        # BaseVectorStore + FAISS implementation
│   │   ├── cache_service.py       # Multi-tier Redis cache
│   │   └── dependencies.py        # FastAPI dependency injection
│   ├── workers/
│   │   └── tasks.py               # Celery task definitions + pipeline
│   ├── db/
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   ├── repository.py          # Repository layer (data access)
│   │   └── session.py             # Async engine + session factory
│   └── schemas/
│       └── extract.py             # Pydantic request/response schemas
├── worker/
│   └── worker.py                  # Celery worker entrypoint
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── unit/                      # Unit tests (mocked dependencies)
│   ├── integration/               # Integration tests (real DB/Redis)
│   └── e2e/                       # End-to-end pipeline tests
├── k8s/                           # Kubernetes manifests
│   ├── namespace.yml
│   ├── configmap.yml
│   ├── secret.yml
│   ├── api-deployment.yml         # API Deployment + PVC for FAISS
│   ├── api-service.yml            # ClusterIP service
│   ├── api-hpa.yml                # HPA: 2→10 pods on CPU/memory
│   ├── worker-deployment.yml      # Indexing + default workers + Beat
│   ├── worker-hpa.yml             # Worker HPA: 1→8 pods on CPU
│   ├── redis-deployment.yml       # Redis + PVC + service
│   └── postgres-deployment.yml    # PostgreSQL + PVC + service
├── nginx/
│   └── nginx.conf                 # Reverse proxy + load balancer
├── monitoring/
│   ├── prometheus.yml             # Scrape configuration
│   └── grafana/provisioning/      # Grafana datasource provisioning
├── scripts/
│   └── run_tests.sh               # Test runner helper
├── Dockerfile                     # API image (multi-stage, gunicorn)
├── Dockerfile.worker              # Worker image (multi-stage, celery)
├── docker-compose.yml             # Full dev stack
├── docker-compose.override.yml    # Dev overrides (hot reload, bind mount)
├── docker-compose.prod.yml        # Production (Nginx, split workers)
├── docker-compose.test.yml        # Test DB + Redis on separate ports
├── requirements.txt               # Python dependencies
├── pytest.ini                     # pytest configuration
├── .env.example                   # Environment variable template
└── README.md
```

---

## 🚀 Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose v2+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/StructAI.git
cd StructAI

# Create your environment file from the template
cp .env.example .env

# Edit .env and set your OpenAI API key
```

### 2. Start the Stack (Development)

```bash
# Build and start all services
docker compose up --build

# Or run detached
docker compose up -d --build
```

Services available at:

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | FastAPI application |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |
| **Flower** | http://localhost:5555 | Celery task monitoring |
| **Prometheus** | http://localhost:9090 | Metrics query dashboard |
| **Grafana** | http://localhost:3000 | Visualisation (admin/admin) |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache + broker |

### 3. Start the Stack (Production)

```bash
# Production: Nginx load balancer + Gunicorn + split workers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Scale API replicas behind Nginx
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=3

# Scale indexing workers independently
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker-indexing=4
```

### 4. Verify

```bash
# Liveness check
curl http://localhost:8000/api/v1/health | python3 -m json.tool

# Readiness check (verifies all dependencies)
curl http://localhost:8000/api/v1/ready | python3 -m json.tool
```

### 5. Tear Down

```bash
docker compose down        # Stop services
docker compose down -v     # Stop + remove volumes (clean slate)
```

---

## ⚙️ Configuration Reference

All configuration is managed via environment variables (loaded from `.env` via `pydantic-settings`):

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `StructAI` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `DEBUG` | `false` | Enable debug mode (coloured logs, SQL echo) |
| `LOG_LEVEL` | `INFO` | Logging level |

### OpenAI

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_MAX_TOKENS` | `4096` | Max tokens per completion |
| `OPENAI_TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `ai_user` | Database user |
| `POSTGRES_PASSWORD` | `ai_pass` | Database password |
| `POSTGRES_HOST` | `db` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_DB` | `ai_db` | Database name |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database index |
| `REDIS_CACHE_TTL` | `3600` | Default cache TTL (seconds) |
| `EMBEDDING_CACHE_TTL` | `86400` | Embedding cache TTL (24 hours) |

### Vector Store

| Variable | Default | Description |
|----------|---------|-------------|
| `FAISS_INDEX_DIR` | `/app/data/faiss` | FAISS index storage path |
| `FAISS_DIMENSION` | `1536` | Embedding vector dimension |
| `VECTOR_STORE_BACKEND` | `faiss` | Vector store backend (`faiss`, `pinecone`, etc.) |

### Processing & Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (characters) |
| `RATE_LIMIT_DEFAULT` | `60/minute` | Global rate limit per IP |
| `RATE_LIMIT_EXTRACT` | `10/minute` | Extraction endpoint rate limit |
| `BACKPRESSURE_MAX_INFLIGHT` | `100` | Max concurrent API requests |
| `BACKPRESSURE_MAX_QUEUE_DEPTH` | `500` | Max pending queue tasks |

### Production Server

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_CONCURRENCY` | `CPU×2+1` | Gunicorn worker count |
| `MAX_REQUESTS` | `1000` | Worker recycling threshold |
| `MAX_REQUESTS_JITTER` | `50` | Recycling jitter |
| `WORKER_TIMEOUT` | `120` | Request timeout (seconds) |

---

## 🔌 API Reference

**Base path:** `/api/v1`  
**Interactive docs:** http://localhost:8000/docs

### Health & Readiness

#### `GET /api/v1/health` — Liveness Probe

Lightweight check for Kubernetes liveness. Returns status, version, and dependency states.

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3642.15,
  "db": "connected",
  "redis": "connected",
  "faiss_index_loaded": true
}
```

#### `GET /api/v1/ready` — Readiness Probe

Deep check — verifies PostgreSQL, Redis, Celery broker, and FAISS. Returns **HTTP 503** if not ready.

```bash
curl http://localhost:8000/api/v1/ready
```

```json
{
  "ready": true,
  "checks": {
    "postgres": { "status": "ok" },
    "redis": { "status": "ok" },
    "celery_broker": { "status": "ok" },
    "faiss": { "status": "ok" }
  },
  "version": "0.1.0"
}
```

---

### Document Indexing

#### `POST /api/v1/documents/index` — Ingest a Document

Accepts raw text, persists metadata to PostgreSQL, and enqueues background processing.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/documents/index \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "contract.pdf",
    "content": "This Software License Agreement is entered into as of January 15, 2025, between Acme Corp (\"Licensor\") and Beta Inc (\"Licensee\"). The Licensee agrees to pay $50,000 annually. The agreement is effective for 3 years with automatic renewal unless terminated with 90 days notice."
  }'
```

**Response (HTTP 202 Accepted):**

```json
{
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "message": "Document queued for indexing"
}
```

| Error | Reason |
|-------|--------|
| 409 | Duplicate document (identical content hash already indexed) |
| 429 | Rate limit exceeded |
| 503 | Processing queue full (backpressure) |

---

### Task Status

#### `GET /api/v1/tasks/{task_id}` — Poll Background Task

Check the status of a document indexing pipeline.

```bash
curl http://localhost:8000/api/v1/tasks/{task_id}
```

```json
{
  "task_id": "abc123-def456",
  "status": "SUCCESS",
  "result": {
    "document_id": "a1b2c3d4-...",
    "status": "indexed",
    "chunks": 12
  },
  "date_done": "2025-06-15T10:30:00Z"
}
```

**Task states:** `PENDING` → `STARTED` → `SUCCESS` | `FAILURE` | `RETRY`

---

### Structured Extraction

#### `POST /api/v1/extract` — Extract Data from a Document

Runs the RAG pipeline: embed query → FAISS search → retrieve chunks → LLM extraction → cache result.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "query": "Extract all parties, effective date, payment terms, and renewal policy",
    "schema_hint": {
      "parties": ["string"],
      "effective_date": "string",
      "annual_payment": "string",
      "duration": "string",
      "renewal_policy": "string"
    }
  }'
```

**Response:**

```json
{
  "extraction_id": "f7e8d9c0-b1a2-3456-cdef-7890abcdef12",
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "query": "Extract all parties, effective date, payment terms, and renewal policy",
  "result": {
    "parties": ["Acme Corp (Licensor)", "Beta Inc (Licensee)"],
    "effective_date": "January 15, 2025",
    "annual_payment": "$50,000",
    "duration": "3 years",
    "renewal_policy": "Automatic renewal unless terminated with 90 days notice"
  },
  "model_used": "gpt-4o",
  "latency_ms": 2450.32,
  "cached": false
}
```

Subsequent identical queries return `"cached": true` with sub-millisecond latency.

The optional `schema_hint` field guides the LLM to produce output matching the specified shape.

| Error | Reason |
|-------|--------|
| 400 | Document not yet indexed (status ≠ `indexed`) |
| 404 | Document not found or no relevant chunks found |
| 429 | Rate limit exceeded |

---

### Metrics

#### `GET /metrics` — Prometheus Scrape Endpoint

Returns all application metrics in Prometheus text exposition format.

```bash
curl http://localhost:8000/metrics
```

---

## 🔄 Processing Pipeline

When a document is submitted via `POST /documents/index`, the following Celery chain executes:

```text
┌─────────────────┐    ┌────────────────────┐    ┌───────────────┐    ┌──────────────────┐
│ 1. chunk_doc    │───▶│ 2. gen_embeddings  │───▶│ 3. index_vecs │───▶│ 4. finalise_doc  │
│                 │    │                    │    │               │    │                  │
│ Split text into │    │ Call OpenAI embed  │    │ Add to FAISS  │    │ Set status =     │
│ overlapping     │    │ API for each chunk │    │ Persist chunk │    │ "indexed" in DB  │
│ chunks          │    │ L2-normalise vecs  │    │ metadata in DB│    │                  │
└─────────────────┘    └────────────────────┘    └───────────────┘    └──────────────────┘
        │                       │                        │                      │
    queue: indexing         queue: indexing          queue: indexing        queue: default
```

**Reliability features:**

| Feature | Implementation |
|---------|---------------|
| Independent retries | Each stage retries 3× with exponential backoff (30s → 60s → 120s) |
| Idempotency | Re-running deletes old chunks and recreates — safe to retry |
| Error callback | On final failure, document status → `failed` + structured error log |
| Queue routing | Heavy work → `indexing` queue; light work → `default` queue |
| Worker crash safety | `task_acks_late=True` + `task_reject_on_worker_lost=True` |
| Memory leak prevention | `max-tasks-per-child` recycles workers after N tasks |
| Metrics | Per-task duration, success/failure counts, in-progress gauge |

**Periodic maintenance tasks** (via Celery Beat):

| Task | Schedule | Purpose |
|------|----------|---------|
| `cleanup_failed_docs` | Every 24 hours | Remove documents stuck in `failed` status > 7 days |
| `optimise_faiss_index` | Every 1 hour | Re-save FAISS index for disk compaction |

---

## 📊 Observability

### Structured Logging

All logs are JSON-formatted in production and coloured console in debug mode, using `structlog`:

```json
{
  "event": "llm.chat_completion.done",
  "model": "gpt-4o",
  "usage": {"prompt_tokens": 1250, "completion_tokens": 340},
  "correlation_id": "a3f2b1c0d4e5",
  "timestamp": "2025-06-15T10:30:00.000Z",
  "level": "info"
}
```

Every request receives a **correlation ID** (`X-Correlation-ID` header) that:
- Is propagated through all logs during the request
- Is returned in the response headers
- Can be sent by the client to trace a request through the system

### Prometheus Metrics

Key metrics exposed at `/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `structai_http_request_duration_seconds` | Histogram | method, path, status_code | Request latency |
| `structai_http_requests_total` | Counter | method, path, status_code | Total request count |
| `structai_http_requests_in_progress` | Gauge | method | Current in-flight requests |
| `structai_llm_call_duration_seconds` | Histogram | operation, model | LLM API call latency |
| `structai_llm_calls_total` | Counter | operation, model, status | LLM call count |
| `structai_llm_tokens_total` | Counter | model, type | Token consumption |
| `structai_cache_ops_total` | Counter | operation, result | Cache hit/miss/set counts |
| `structai_faiss_search_duration_seconds` | Histogram | — | Vector search latency |
| `structai_faiss_index_size_vectors` | Gauge | — | Current index size |
| `structai_tasks_total` | Counter | task_name, status | Background task count |
| `structai_task_duration_seconds` | Histogram | task_name, status | Task execution time |
| `structai_documents_indexed_total` | Counter | — | Successfully indexed docs |
| `structai_documents_failed_total` | Counter | — | Failed indexing attempts |

### Monitoring Stack

| Service | URL | Purpose |
|---------|-----|---------|
| Prometheus | http://localhost:9090 | Metric collection & querying |
| Grafana | http://localhost:3000 | Dashboards & alerting (admin/admin) |
| Flower | http://localhost:5555 | Celery task monitoring |

Prometheus is pre-configured to scrape the API at 15-second intervals.

---

## 🧪 Testing

### Test Architecture

```text
┌─────────────────────────────────────────────────┐
│                 Test Pyramid                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ▲  E2E Tests          (tests/e2e/)            │
│  │  Full pipeline: index → extract → verify    │
│  │                                             │
│  │  Integration Tests   (tests/integration/)   │
│  │  Real DB + Redis, FastAPI TestClient         │
│  │                                             │
│  │  Unit Tests          (tests/unit/)          │
│  │  Mocked dependencies, fast, isolated         │
│  ▼                                             │
└─────────────────────────────────────────────────┘
```

### Running Tests

```bash
# ── Unit Tests (no external dependencies) ──────────────────────
pytest tests/unit/ -v

# ── Integration Tests ──────────────────────────────────────────
# Start isolated test DB + Redis (separate ports to avoid conflicts)
docker compose -f docker-compose.test.yml up -d

# Run integration tests
TEST_DATABASE_URL=postgresql+asyncpg://ai_user:ai_pass@localhost:5433/ai_db_test \
TEST_REDIS_URL=redis://localhost:6380/1 \
pytest tests/integration/ -v

# Tear down test dependencies
docker compose -f docker-compose.test.yml down -v

# ── E2E Tests ──────────────────────────────────────────────────
pytest tests/e2e/ -v

# ── All Tests with Coverage ───────────────────────────────────
pytest --cov=app --cov-report=term-missing --cov-report=html

# ── Via Helper Script ─────────────────────────────────────────
./scripts/run_tests.sh
```

### Test Examples

**Unit test** — mocked LLM client:

```python
@pytest.mark.asyncio
async def test_extraction_returns_parsed_json(mock_llm_client):
    mock_llm_client.chat_completion.return_value = '{"name": "John", "age": 30}'
    service = LangExtractService(mock_llm_client)

    result = await service.extract(["Some text about John..."], "Extract name and age")

    assert result == {"name": "John", "age": 30}
    mock_llm_client.chat_completion.assert_called_once()
```

**Integration test** — real FastAPI + database:

```python
@pytest.mark.asyncio
async def test_index_document_returns_202(async_client, db_session):
    response = await async_client.post("/api/v1/documents/index", json={
        "filename": "test.txt",
        "content": "Hello world, this is a test document."
    })

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "pending"
    assert "document_id" in data
```

**E2E test** — full pipeline:

```python
@pytest.mark.asyncio
async def test_full_extraction_pipeline(async_client):
    # 1. Index a document
    idx = await async_client.post("/api/v1/documents/index", json={
        "filename": "contract.pdf",
        "content": "Agreement between Acme Corp and Beta Inc..."
    })
    doc_id = idx.json()["document_id"]

    # 2. Wait for background processing to complete
    await wait_for_document_status(doc_id, "indexed", timeout=30)

    # 3. Extract structured data
    ext = await async_client.post("/api/v1/extract", json={
        "document_id": doc_id,
        "query": "Extract all parties mentioned"
    })

    assert ext.status_code == 200
    result = ext.json()
    assert result["cached"] is False
    assert "result" in result

    # 4. Verify cache works on repeat query
    ext2 = await async_client.post("/api/v1/extract", json={
        "document_id": doc_id,
        "query": "Extract all parties mentioned"
    })
    assert ext2.json()["cached"] is True
```

### What Each Test Layer Covers

| Layer | Tests | Dependencies |
|-------|-------|-------------|
| **Unit** | LLM client, embedding service, LangExtract service, cache service, vector store, chunking, metrics | None (all mocked) |
| **Integration** | API endpoints, DB repository, Redis cache, FAISS indexing + retrieval | Test PostgreSQL + Redis |
| **E2E** | Full pipeline: index → process → extract → cache hit | All services running |

---

## 📈 Scaling & Production

### Horizontal Scaling of the API

The API is **stateless** — scale horizontally with zero configuration changes:

```bash
# Docker Compose: scale behind Nginx load balancer
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=3

# Kubernetes: HPA auto-scales 2→10 pods based on CPU/memory
kubectl apply -f k8s/api-hpa.yml
```

**Production server** uses Gunicorn with uvicorn workers:
- `preload_app=True` — shares FAISS index across workers via copy-on-write
- `MAX_REQUESTS=2000` — recycles workers to prevent memory leaks
- `PROMETHEUS_MULTIPROC_DIR` — enables metrics in multi-process mode
- Nginx least-connections load balancing across replicas

### Scaling Workers Independently

Workers are split by queue type for independent scaling:

| Queue | Worker | Workload | Scale Strategy |
|-------|--------|----------|----------------|
| `indexing` | worker-indexing | Embedding generation + FAISS writes | CPU/memory-bound → scale on utilisation |
| `default` | worker-default | Finalisation, pipeline orchestration | I/O-bound → fewer instances needed |
| `maintenance` | worker-default | Cleanup, FAISS compaction | Runs on default workers |

```bash
# Scale only the heavy indexing workers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker-indexing=4
```

### Caching: Three Tiers to Reduce LLM Cost

| Tier | What | Key Pattern | TTL | Benefit |
|------|------|-------------|-----|---------|
| 1 | Extraction results | `extract:{doc_id}:{query_hash}` | 1 hour | Identical queries served instantly |
| 2 | Embedding vectors | `emb:{model}:{text_hash}` | 24 hours | Skip redundant OpenAI embedding calls |
| 3 | Content dedup | `dedup:{content_hash}` | 7 days | Skip re-indexing identical documents |

Additional features: bulk invalidation on re-index, cache stats for monitoring, LRU eviction under memory pressure.

### Rate Limiting & Backpressure

| Layer | Mechanism | Limits |
|-------|-----------|--------|
| **Nginx** | `limit_req_zone` | 10 req/s general, 2 req/s extraction |
| **SlowAPI** | Per-IP rate limiting | 60/min global, 10/min extraction |
| **Backpressure** | In-flight + queue depth | 503 when > 100 inflight or > 500 queued |

All layers return `Retry-After` headers so clients can implement exponential backoff.

### Swapping the Vector Store

The vector store uses an abstract interface (`BaseVectorStore`):

```python
class BaseVectorStore(ABC):
    def add(self, vectors, metadata=None) -> List[int]: ...
    def search(self, query_vector, k=5) -> List[Tuple[int, float]]: ...
    def delete(self, ids) -> int: ...
    def save(self) -> None: ...
    def reset(self) -> None: ...
```

To migrate: implement a new subclass, set `VECTOR_STORE_BACKEND=pinecone` (or `qdrant`, `weaviate`), done.

**When to migrate from FAISS:** index exceeds available RAM, need metadata filtering, need multi-node writes, or hosting in ephemeral/serverless environments.

---

## ☸️ Kubernetes Deployment

Full manifests in `k8s/`:

```bash
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/secret.yml          # Populate with real values first
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/postgres-deployment.yml
kubectl apply -f k8s/redis-deployment.yml
kubectl apply -f k8s/api-deployment.yml
kubectl apply -f k8s/api-service.yml
kubectl apply -f k8s/api-hpa.yml
kubectl apply -f k8s/worker-deployment.yml
kubectl apply -f k8s/worker-hpa.yml
```

| Feature | Details |
|---------|---------|
| **HPA** | API: 2→10 pods on CPU/memory. Workers: 1→8 pods on CPU |
| **Rolling updates** | Zero-downtime with `maxUnavailable: 0`, `maxSurge: 1` |
| **Three probe types** | Startup, liveness (`/health`), readiness (`/ready`) |
| **Resource limits** | CPU/memory requests and limits on every container |
| **PVC** | FAISS index shared via `ReadWriteMany` PersistentVolumeClaim |
| **Secrets** | Sensitive values in K8s Secrets |
| **Beat singleton** | `replicas: 1` + `Recreate` strategy for the scheduler |

For queue-depth-based autoscaling, add [KEDA](https://keda.sh) with a Redis scaler.

---

## 🔐 Security

| Measure | Implementation |
|---------|---------------|
| Secret management | Environment variables / K8s Secrets — never committed to git |
| Non-root containers | Docker images create and run as `appuser` (UID 1000) |
| Rate limiting | Three layers: SlowAPI, Nginx, backpressure middleware |
| Input validation | Pydantic models validate all request bodies |
| Content deduplication | SHA-256 content hash prevents duplicate processing |
| CORS | Configurable middleware (restrict origins in production) |
| Network isolation | Docker Compose internal bridge network |
| Log sanitisation | Sensitive data excluded from structured logs |
| Health endpoints | No sensitive data exposed in `/health` or `/ready` |

---

## 🛣️ Roadmap

- [x] Core RAG pipeline (chunk → embed → index → extract)
- [x] Background processing with Celery (retries, dead-letter, queue routing)
- [x] Multi-tier Redis caching (extraction, embedding, dedup)
- [x] Prometheus + Grafana observability stack
- [x] Structured logging with correlation IDs
- [x] Rate limiting + backpressure middleware
- [x] Abstract vector store interface (FAISS swappable)
- [x] Kubernetes manifests with HPA
- [x] Nginx load balancer + Gunicorn production config
- [x] Full testing pyramid (unit, integration, E2E)
- [x] Comprehensive documentation
- [ ] Multi-LLM provider support (Azure OpenAI, Anthropic, local models)
- [ ] Streaming ingestion & real-time extraction
- [ ] KEDA-based autoscaling (scale workers on queue depth)
- [ ] Helm chart for simplified K8s deployment
- [ ] OpenTelemetry distributed tracing
- [ ] Alembic database migrations

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

---

## 📄 License

This project is for educational and portfolio purposes. See the repository for license details.

---

> Built with ☕ and a passion for production-grade AI systems.

