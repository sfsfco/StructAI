"""
API routes for the AI extraction platform.

Endpoints:
  POST /documents/index  — ingest and queue a document for indexing
  POST /extract          — run structured extraction on an indexed document
  GET  /health           — liveness probe
  GET  /ready            — readiness probe (checks all dependencies)
"""

import hashlib
import json
import os
import time
from uuid import UUID

import faiss
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repository import ChunkRepository, DocumentRepository, ExtractionRepository
from app.db.session import async_session_factory, get_db
from app.middleware.rate_limit import limiter
from app.schemas.extract import (
    DocumentIndexRequest,
    DocumentIndexResponse,
    ErrorResponse,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    ReadinessResponse,
    TaskStatusResponse,
)
from app.services.cache_service import CacheService
from app.services.dependencies import (
    get_cache,
    get_chunk_repo,
    get_document_repo,
    get_embedding_service,
    get_extract_service,
    get_extraction_repo,
    get_vector_store,
)
from app.services.embedding_service import EmbeddingService
from app.services.langextract_service import LangExtractService
from app.services.vector_store import VectorStore

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Track startup time for uptime reporting
_STARTUP_TIME: float = time.time()


# ── Health ───────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness probe",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Lightweight liveness probe.

    Returns basic service status and version.  Kubernetes uses this to
    determine whether the container should be restarted (liveness).
    Dependency checks are intentionally shallow — use ``/ready`` for
    deeper readiness checks.
    """

    # ── Check Redis ──────────────────────────────────────────────────
    redis_status = "unavailable"
    try:
        cache: CacheService | None = getattr(request.app.state, "cache", None)
        if cache and await cache.ping():
            redis_status = "connected"
    except Exception:
        redis_status = "error"

    # ── Check PostgreSQL ─────────────────────────────────────────────
    db_status = "unavailable"
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception:
        db_status = "error"

    # ── Check FAISS index ────────────────────────────────────────────
    faiss_loaded = os.path.exists(
        os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    )

    overall = (
        "ok"
        if db_status == "connected" and redis_status == "connected"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _STARTUP_TIME, 2),
        db=db_status,
        redis=redis_status,
        faiss_index_loaded=faiss_loaded,
    )


# ── Readiness ────────────────────────────────────────────────────────────


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["ops"],
    summary="Readiness probe",
)
async def readiness_check(request: Request) -> ReadinessResponse:
    """
    Deep readiness probe.

    Verifies that every critical dependency (PostgreSQL, Redis, Celery
    broker) is reachable and operational.  Kubernetes uses this to decide
    whether to route traffic to the pod.

    Returns HTTP 200 when ready, HTTP 503 when not.
    """
    checks: dict = {}

    # ── PostgreSQL ───────────────────────────────────────────────────
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = {"status": "ok"}
    except Exception as exc:
        checks["postgres"] = {"status": "error", "detail": str(exc)[:200]}

    # ── Redis ────────────────────────────────────────────────────────
    try:
        cache: CacheService | None = getattr(request.app.state, "cache", None)
        if cache and await cache.ping():
            checks["redis"] = {"status": "ok"}
        else:
            checks["redis"] = {"status": "unavailable"}
    except Exception as exc:
        checks["redis"] = {"status": "error", "detail": str(exc)[:200]}

    # ── Celery Broker ────────────────────────────────────────────────
    try:
        from app.workers.tasks import celery_app

        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.close()
        checks["celery_broker"] = {"status": "ok"}
    except Exception as exc:
        checks["celery_broker"] = {"status": "error", "detail": str(exc)[:200]}

    # ── FAISS index ──────────────────────────────────────────────────
    faiss_path = os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(faiss_path):
        checks["faiss"] = {"status": "ok"}
    else:
        checks["faiss"] = {"status": "not_found", "detail": "Index file missing"}

    # ── Overall readiness ────────────────────────────────────────────
    critical = ["postgres", "redis"]
    ready = all(checks.get(k, {}).get("status") == "ok" for k in critical)

    response = ReadinessResponse(
        ready=ready,
        checks=checks,
        version=settings.APP_VERSION,
    )

    if not ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )

    return response


# ── Documents ────────────────────────────────────────────────────────────


@router.post(
    "/documents/index",
    response_model=DocumentIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["documents"],
    summary="Index a new document",
    responses={
        409: {"model": ErrorResponse, "description": "Duplicate document"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def index_document(
    request: Request,
    body: DocumentIndexRequest,
    doc_repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentIndexResponse:
    """
    Accept a document, persist metadata, and enqueue background
    processing (chunking → embedding → FAISS indexing).
    """
    content_hash = hashlib.sha256(body.content.encode()).hexdigest()

    # Idempotency: reject if document with same hash already exists
    if await doc_repo.get_by_content_hash(content_hash):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document with identical content already indexed.",
        )

    doc = await doc_repo.create(
        filename=body.filename,
        content_hash=content_hash,
    )

    # Enqueue background task (import here to avoid circular deps at module level)
    from app.workers.tasks import process_document

    process_document.delay(str(doc.id), body.content)

    logger.info("document.index.queued", document_id=str(doc.id))
    return DocumentIndexResponse(
        document_id=doc.id,
        status="pending",
    )


# ── Task Status ──────────────────────────────────────────────────────────


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["tasks"],
    summary="Check background task status",
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Poll the status of a background processing task.

    Returns the Celery task state (PENDING / STARTED / SUCCESS / FAILURE)
    along with the result or error message when available.
    """
    from app.workers.tasks import celery_app

    result = celery_app.AsyncResult(task_id)

    response = TaskStatusResponse(
        task_id=task_id,
        status=result.status,
    )

    if result.ready():
        if result.successful():
            response.result = result.result if isinstance(result.result, dict) else {"value": result.result}
        else:
            response.error = str(result.result)
        response.date_done = str(result.date_done) if result.date_done else None

    return response


# ── Extraction ───────────────────────────────────────────────────────────


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    tags=["extraction"],
    summary="Extract structured data from an indexed document",
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        400: {"model": ErrorResponse, "description": "Document not ready"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit(settings.RATE_LIMIT_EXTRACT)
async def extract(
    request: Request,
    body: ExtractionRequest,
    doc_repo: DocumentRepository = Depends(get_document_repo),
    chunk_repo: ChunkRepository = Depends(get_chunk_repo),
    extraction_repo: ExtractionRepository = Depends(get_extraction_repo),
    cache: CacheService = Depends(get_cache),
    embedder: EmbeddingService = Depends(get_embedding_service),
    extractor: LangExtractService = Depends(get_extract_service),
    store: VectorStore = Depends(get_vector_store),
) -> ExtractionResponse:
    """
    Run RAG-based extraction:
      1. Check Redis cache
      2. Retrieve relevant chunks via FAISS
      3. Call LangExtract + LLM
      4. Cache & persist the result
    """
    # ── Verify document exists and is indexed ────────────────────────
    doc = await doc_repo.get_by_id(body.document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    if doc.status != "indexed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is not ready (status={doc.status}).",
        )

    # ── Cache check ──────────────────────────────────────────────────
    cache_key = CacheService.extraction_key(str(body.document_id), body.query)
    cached = await cache.get(cache_key)
    if cached:
        logger.info("extract.cache_hit", document_id=str(body.document_id))
        return ExtractionResponse(
            extraction_id=cached["extraction_id"],
            document_id=body.document_id,
            query=body.query,
            result=cached["result"],
            model_used=cached.get("model_used"),
            latency_ms=cached.get("latency_ms"),
            cached=True,
        )

    # ── Vector search ────────────────────────────────────────────────
    query_vec = await embedder.embed_text(body.query)
    query_arr = np.array([query_vec], dtype=np.float32)
    faiss.normalize_L2(query_arr)

    neighbours = store.search(query_arr, k=5)
    faiss_ids = [n[0] for n in neighbours]

    # Fetch chunk texts from DB via repository
    chunk_rows = await chunk_repo.get_by_faiss_ids(faiss_ids)
    chunks = [c.text for c in chunk_rows]

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found for this query.",
        )

    # ── LLM extraction ───────────────────────────────────────────────
    t0 = time.perf_counter()
    extracted = await extractor.extract(chunks, body.query, body.schema_hint)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    # ── Persist & cache ──────────────────────────────────────────────
    extraction = await extraction_repo.create(
        document_id=body.document_id,
        query=body.query,
        result_json=json.dumps(extracted),
        model_used=settings.OPENAI_MODEL,
        latency_ms=latency,
    )

    cache_payload = {
        "extraction_id": str(extraction.id),
        "result": extracted,
        "model_used": settings.OPENAI_MODEL,
        "latency_ms": latency,
    }
    await cache.set(cache_key, cache_payload)

    return ExtractionResponse(
        extraction_id=extraction.id,
        document_id=body.document_id,
        query=body.query,
        result=extracted,
        model_used=settings.OPENAI_MODEL,
        latency_ms=latency,
        cached=False,
    )
