# Changelog

All notable changes to StructAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.9.0] – 2025-06-15

### Added – Phase 9: Documentation

- Comprehensive README with architecture diagrams, full API reference, configuration table, and deployment guides
- CONTRIBUTING.md with development setup, coding standards, and PR guidelines
- CHANGELOG.md documenting all phases of development
- Updated `.env.example` with all Phase 8 configuration variables

---

## [0.8.0] – 2025-06-15

### Added – Phase 8: Scaling & Production

- Gunicorn production server configuration (`app/core/gunicorn_conf.py`) with worker recycling, preload, and Prometheus multiprocess support
- Backpressure middleware (`app/middleware/backpressure.py`) — in-flight request limits and Celery queue-depth checks with HTTP 503 + Retry-After
- Production Docker Compose (`docker-compose.prod.yml`) with Nginx load balancer and split worker services
- Nginx reverse proxy (`nginx/nginx.conf`) with least-conn balancing, rate limiting, gzip, and structured logging
- Kubernetes manifests (`k8s/`) — Deployments, Services, HPA, ConfigMap, Secrets, PVCs for API, workers, Redis, and PostgreSQL
- Abstract vector store interface (`BaseVectorStore` ABC) with `create_vector_store()` factory
- Multi-tier caching: embedding cache (Tier 2) and content deduplication (Tier 3)
- Cache stats endpoint and bulk invalidation support
- Configuration: `EMBEDDING_CACHE_TTL`, `BACKPRESSURE_MAX_INFLIGHT`, `BACKPRESSURE_MAX_QUEUE_DEPTH`, `VECTOR_STORE_BACKEND`, `API_WORKERS`, `WORKER_MAX_TASKS_PER_CHILD`, `WORKER_PREFETCH_MULTIPLIER`

### Changed

- Dockerfile CMD switched from `uvicorn` to `gunicorn` with `gunicorn_conf.py`
- `VectorStore` renamed to `FAISSVectorStore`, now implements `BaseVectorStore`
- Added `gunicorn==22.0.0` to requirements

---

## [0.7.0] – 2025-06-14

### Added – Phase 7: CI/CD & DevOps

- Multi-stage Docker builds for API and worker images
- Docker Compose orchestration (dev, test, production overrides)
- `.env.example` with all configuration variables
- `docker-compose.test.yml` with isolated test database and Redis
- Non-root container user (`appuser`) for security
- Health check commands in Docker Compose services

---

## [0.6.0] – 2025-06-14

### Added – Phase 6: Testing

- Unit tests for all services: LLM client, embedding service, LangExtract service, cache service, vector store, chunking logic, metrics
- Integration tests: API routes, database repository, FAISS vector operations
- End-to-end pipeline test: index → process → extract → cache verification
- Test fixtures in `conftest.py` with async support
- pytest configuration (`pytest.ini`) with async mode and coverage
- Test runner script (`scripts/run_tests.sh`)

---

## [0.5.0] – 2025-06-13

### Added – Phase 5: Observability & Monitoring

- Prometheus metrics: HTTP request duration/count, LLM call latency/tokens, FAISS search duration/index size, task execution metrics
- Grafana dashboards with pre-provisioned Prometheus datasource
- Structured JSON logging via `structlog` with correlation ID propagation
- `X-Correlation-ID` header middleware — generated or passed through from client
- Prometheus metrics middleware for per-request instrumentation
- Monitoring stack in Docker Compose: Prometheus (`:9090`), Grafana (`:3000`)

---

## [0.4.0] – 2025-06-13

### Added – Phase 4: Resilience & Error Handling

- Celery task retries with exponential backoff (3 attempts, 30s → 60s → 120s)
- Dead-letter handling: failed documents marked with error status
- Error callbacks on pipeline chain failure
- Rate limiting via SlowAPI: 60/min global, 10/min extraction endpoint
- Input validation via Pydantic schemas (`ExtractionRequest`, `ExtractionResponse`)
- Content deduplication via SHA-256 hash (HTTP 409 on duplicates)
- Document status lifecycle: `pending` → `processing` → `indexed` | `failed`

---

## [0.3.0] – 2025-06-12

### Added – Phase 3: Background Processing

- Celery worker with Redis broker and result backend
- Processing pipeline as chained tasks: `chunk_document` → `generate_embeddings` → `index_vectors` → `finalise_document`
- Queue routing: heavy work → `indexing`, light work → `default`, scheduled → `maintenance`
- Celery Beat periodic tasks: failed document cleanup (24h), FAISS index optimisation (1h)
- Flower monitoring UI at `:5555`
- Task status polling endpoint: `GET /api/v1/tasks/{task_id}`
- Worker entrypoint (`worker/worker.py`) with auto-discovery and queue configuration

---

## [0.2.0] – 2025-06-11

### Added – Phase 2: RAG Pipeline & Services

- OpenAI LLM client (`BaseLLMClient` + `OpenAIClient`) with chat completion and embedding methods
- Embedding service with batch processing and L2 normalisation
- FAISS vector store with add, search, delete, save, and reset operations
- LangExtract service: schema-guided structured extraction via LLM
- Redis cache service for extraction result caching
- Document chunking with configurable size and overlap
- Extraction endpoint: `POST /api/v1/extract`
- FastAPI dependency injection (`services/dependencies.py`)

---

## [0.1.0] – 2025-06-10

### Added – Phase 1: Foundation

- FastAPI application scaffold with async support
- PostgreSQL database models: `Document`, `DocumentChunk`, `Extraction` (SQLAlchemy async)
- Repository pattern for data access (`DocumentRepository`)
- Async database session management with connection pooling
- Configuration via pydantic-settings (`.env` support)
- Document indexing endpoint: `POST /api/v1/documents/index`
- Health check endpoint: `GET /api/v1/health`
- Readiness probe endpoint: `GET /api/v1/ready`
- Project structure with clear separation of concerns
