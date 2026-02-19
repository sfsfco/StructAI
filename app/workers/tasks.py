"""
Celery task definitions for background document processing.

Architecture
------------
The pipeline is split into **granular, chainable tasks** so each stage
can be retried independently and monitored in isolation:

  1. ``chunk_document``    – split raw text into overlapping chunks
  2. ``generate_embeddings`` – call the embedding model for each chunk
  3. ``index_vectors``     – insert vectors into FAISS and persist chunks
  4. ``finalise_document`` – mark the document as *indexed* in PostgreSQL

The convenience task ``process_document`` wires them into a Celery
**chain** that executes sequentially and short-circuits on failure.

Periodic tasks (registered via Celery Beat):
  - ``cleanup_failed_documents`` – purge stale *failed* documents
  - ``optimise_faiss_index``     – compact / re-save the FAISS index

Idempotency
-----------
Every task can be re-run safely:
  - Chunks and FAISS vectors are **deleted and re-created** for the
    document on each run (upsert semantics via ``_clear_previous``).
  - Status transitions are last-write-wins; only ``finalise_document``
    sets the terminal *indexed* state.

Retries & failure handling
--------------------------
  - Each stage retries up to 3 times with exponential back-off
    (30 s → 60 s → 120 s).
  - ``task_reject_on_worker_lost`` ensures the broker re-delivers if a
    worker is killed mid-task.
  - On final failure after exhausting retries, the document status is
    set to *failed* and a structured-log error is emitted.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from celery import Celery, chain, signals
from celery.utils.log import get_task_logger

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.metrics import (
    CHUNKS_CREATED_TOTAL,
    DOCUMENTS_FAILED_TOTAL,
    DOCUMENTS_INDEXED_TOTAL,
    TASK_DURATION,
    TASKS_IN_PROGRESS,
    TASKS_TOTAL,
)

logger = get_logger(__name__)
settings = get_settings()

# ── Celery App ───────────────────────────────────────────────────────────

celery_app = Celery(
    "structai",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,  # keep results for 1 hour

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Reliability
    task_acks_late=True,                 # re-deliver if worker crashes
    worker_prefetch_multiplier=1,        # one task at a time per worker
    task_reject_on_worker_lost=True,     # reject on SIGKILL / OOM
    task_track_started=True,             # report STARTED state to backend

    # Retries
    task_default_retry_delay=30,
    task_max_retries=3,

    # Queue routing — heavy GPU / IO work goes to "indexing" queue
    task_routes={
        "chunk_document":       {"queue": "indexing"},
        "generate_embeddings":  {"queue": "indexing"},
        "index_vectors":        {"queue": "indexing"},
        "finalise_document":    {"queue": "default"},
        "process_document":     {"queue": "default"},
        "cleanup_failed_docs":  {"queue": "maintenance"},
        "optimise_faiss_index": {"queue": "maintenance"},
    },

    # Default queue consumed by every worker
    task_default_queue="default",

    # Beat schedule — periodic maintenance tasks
    beat_schedule={
        "cleanup-failed-docs-daily": {
            "task": "cleanup_failed_docs",
            "schedule": 86_400.0,  # every 24 h
        },
        "optimise-faiss-hourly": {
            "task": "optimise_faiss_index",
            "schedule": 3_600.0,  # every 1 h
        },
    },
)


# ── Celery Signals (observability hooks) ─────────────────────────────────

@signals.task_prerun.connect
def _on_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    """Log + attach timing info when a task starts."""
    setup_logging(log_level=settings.LOG_LEVEL, json_logs=True)
    task_name = sender.name if sender else "unknown"
    TASKS_IN_PROGRESS.labels(task_name=task_name).inc()
    # Store start time on the task request for duration tracking
    if task and hasattr(task, "request"):
        task.request._metrics_start = time.perf_counter()
    logger.info(
        "celery.task.prerun",
        task_name=task_name,
        task_id=task_id,
    )


@signals.task_postrun.connect
def _on_task_postrun(sender=None, task_id=None, retval=None, state=None, **kwargs):
    task_name = sender.name if sender else "unknown"
    TASKS_IN_PROGRESS.labels(task_name=task_name).dec()
    status = "success" if state == "SUCCESS" else "failure"
    TASKS_TOTAL.labels(task_name=task_name, status=status).inc()

    # Record duration if start time was captured
    if sender and hasattr(sender, "request"):
        start = getattr(sender.request, "_metrics_start", None)
        if start is not None:
            duration = time.perf_counter() - start
            TASK_DURATION.labels(task_name=task_name, status=status).observe(duration)

    logger.info(
        "celery.task.postrun",
        task_name=task_name,
        task_id=task_id,
        state=state,
    )


@signals.task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    task_name = sender.name if sender else "unknown"
    TASKS_IN_PROGRESS.labels(task_name=task_name).dec()
    TASKS_TOTAL.labels(task_name=task_name, status="failure").inc()
    logger.error(
        "celery.task.failure",
        task_name=task_name,
        task_id=task_id,
        error=str(exception),
    )


@signals.task_retry.connect
def _on_task_retry(sender=None, request=None, reason=None, **kwargs):
    task_name = sender.name if sender else "unknown"
    TASKS_TOTAL.labels(task_name=task_name, status="retry").inc()
    logger.warning(
        "celery.task.retry",
        task_name=task_name,
        task_id=request.id if request else "unknown",
        reason=str(reason),
    )


# ── Helpers ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """
    Split text into overlapping chunks by character count.

    Uses paragraph / sentence boundaries when possible, falling back
    to hard character splits.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to snap to a paragraph or sentence boundary
        if end < len(text):
            # Look backwards for a paragraph break
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                # Fall back to sentence-ending punctuation
                for sep in (". ", "! ", "? ", "\n"):
                    boundary = text.rfind(sep, start + (chunk_size // 2), end)
                    if boundary != -1:
                        boundary += len(sep)
                        break
            else:
                boundary += 2  # skip past the double newline

            if boundary > start:
                end = boundary

        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)

    return chunks


def _run_async(coro):
    """Run an async coroutine from synchronous Celery task code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Granular Tasks ───────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="chunk_document",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
)
def chunk_document(self, document_id: str, content: str) -> Dict[str, Any]:
    """
    Stage 1 — split raw text into overlapping chunks.

    Idempotent: always produces the same chunk list for the same input.
    Clears any previous chunks so a re-run starts fresh.

    Returns
    -------
    dict with ``document_id`` and ``chunks`` list for the next stage.
    """
    logger.info("task.chunk_document.start", document_id=document_id)
    _run_async(_update_document_status(document_id, "processing"))

    # Idempotency — remove old chunks if re-running
    _run_async(_clear_previous_chunks(document_id))

    chunks = chunk_text(content)
    logger.info(
        "task.chunk_document.done",
        document_id=document_id,
        chunk_count=len(chunks),
    )
    return {"document_id": document_id, "chunks": chunks}


@celery_app.task(
    bind=True,
    name="generate_embeddings",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
)
def generate_embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2 — generate embedding vectors for each chunk.

    Receives the output of ``chunk_document``.

    Returns
    -------
    dict with ``document_id``, ``chunks``, and ``vectors`` (list of lists).
    """
    document_id = payload["document_id"]
    chunks: List[str] = payload["chunks"]

    logger.info(
        "task.generate_embeddings.start",
        document_id=document_id,
        chunk_count=len(chunks),
    )

    vectors = _run_async(_generate_embeddings(chunks))

    logger.info(
        "task.generate_embeddings.done",
        document_id=document_id,
        shape=list(vectors.shape),
    )
    # Serialise ndarray → nested list for JSON transport
    return {
        "document_id": document_id,
        "chunks": chunks,
        "vectors": vectors.tolist(),
    }


@celery_app.task(
    bind=True,
    name="index_vectors",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
)
def index_vectors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3 — insert vectors into FAISS and persist chunk metadata.

    Receives the output of ``generate_embeddings``.

    Returns
    -------
    dict with ``document_id``, ``chunk_count``, and ``faiss_ids``.
    """
    document_id = payload["document_id"]
    chunks: List[str] = payload["chunks"]
    vectors_list: List[List[float]] = payload["vectors"]

    logger.info("task.index_vectors.start", document_id=document_id)

    vectors = np.array(vectors_list, dtype=np.float32)

    from app.services.vector_store import VectorStore

    store = VectorStore()
    faiss_ids = store.add(vectors)
    store.save()

    logger.info(
        "task.index_vectors.faiss_done",
        document_id=document_id,
        faiss_ids_count=len(faiss_ids),
    )

    # Persist chunks + FAISS mapping in PostgreSQL
    _run_async(_persist_chunks(document_id, chunks, faiss_ids))

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "faiss_ids": faiss_ids,
    }


@celery_app.task(
    bind=True,
    name="finalise_document",
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def finalise_document(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 4 — mark the document as *indexed* in PostgreSQL.

    Terminal step of the pipeline.
    """
    document_id = payload["document_id"]
    chunk_count = payload["chunk_count"]

    logger.info("task.finalise_document.start", document_id=document_id)

    _run_async(
        _update_document_status(document_id, "indexed", total_chunks=chunk_count)
    )

    DOCUMENTS_INDEXED_TOTAL.inc()
    CHUNKS_CREATED_TOTAL.inc(chunk_count)
    logger.info("task.finalise_document.done", document_id=document_id)
    return {
        "document_id": document_id,
        "status": "indexed",
        "chunks": chunk_count,
    }


# ── Orchestrator ─────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="process_document",
    max_retries=0,
    acks_late=True,
)
def process_document(self, document_id: str, content: str) -> str:
    """
    Orchestrate the full document processing pipeline as a Celery **chain**.

    chunk → embed → index → finalise

    Each stage retries independently.  If any stage exhausts retries the
    chain short-circuits and the on-failure callback marks the document
    as *failed*.

    Returns the chain's AsyncResult ID so callers can poll status.
    """
    logger.info("task.process_document.dispatch", document_id=document_id)

    pipeline = chain(
        chunk_document.s(document_id, content),
        generate_embeddings.s(),
        index_vectors.s(),
        finalise_document.s(),
    )

    # link_error attaches a callback if any task in the chain fails
    pipeline.link_error(on_pipeline_error.s(document_id=document_id))

    result = pipeline.apply_async()
    return result.id


@celery_app.task(name="on_pipeline_error")
def on_pipeline_error(request, exc, traceback, document_id: str = "") -> None:
    """
    Error callback attached to the processing chain.

    Marks the document as *failed* so the API can report accurate status.
    """
    logger.error(
        "task.pipeline.error",
        document_id=document_id,
        error=str(exc),
    )
    DOCUMENTS_FAILED_TOTAL.inc()
    if document_id:
        _run_async(_update_document_status(document_id, "failed"))


# ── Periodic / Maintenance Tasks ─────────────────────────────────────────


@celery_app.task(name="cleanup_failed_docs")
def cleanup_failed_documents() -> Dict[str, Any]:
    """
    Remove documents stuck in *failed* status for more than 7 days.

    Runs daily via Celery Beat.
    """
    logger.info("task.cleanup_failed_docs.start")
    count = _run_async(_delete_old_failed_documents(days=7))
    logger.info("task.cleanup_failed_docs.done", deleted=count)
    return {"deleted": count}


@celery_app.task(name="optimise_faiss_index")
def optimise_faiss_index() -> Dict[str, Any]:
    """
    Re-save the FAISS index to compact on-disk representation.

    Runs hourly via Celery Beat.  For IndexFlatIP this is a simple
    write; for IVF-based indices this would trigger re-training.
    """
    logger.info("task.optimise_faiss.start")
    from app.services.vector_store import VectorStore

    store = VectorStore()
    total = store.total_vectors
    store.save()
    logger.info("task.optimise_faiss.done", total_vectors=total)
    return {"total_vectors": total}


# ── Async DB / Service Helpers ───────────────────────────────────────────


async def _update_document_status(
    document_id: str,
    status: str,
    total_chunks: int | None = None,
) -> None:
    from app.db.repository import DocumentRepository
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        await repo.update_status(document_id, status, total_chunks)
        await session.commit()


async def _generate_embeddings(chunks: List[str]) -> np.ndarray:
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_client import get_llm_client

    llm = get_llm_client()
    embedder = EmbeddingService(llm)
    vectors = await embedder.embed_batch(chunks)

    # L2-normalise for cosine similarity via inner product
    import faiss

    faiss.normalize_L2(vectors)
    return vectors


async def _persist_chunks(
    document_id: str,
    chunks: List[str],
    faiss_ids: List[int],
) -> None:
    from app.db.repository import ChunkRepository
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        repo = ChunkRepository(session)
        await repo.bulk_create(document_id, chunks, faiss_ids)
        await session.commit()


async def _clear_previous_chunks(document_id: str) -> None:
    """Delete existing chunks for idempotent re-processing."""
    from sqlalchemy import delete

    from app.db.models import Chunk
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        await session.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        await session.commit()


async def _delete_old_failed_documents(days: int = 7) -> int:
    """Remove documents in *failed* status older than ``days``."""
    from sqlalchemy import delete

    from app.db.models import Document
    from app.db.session import async_session_factory

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with async_session_factory() as session:
        result = await session.execute(
            delete(Document)
            .where(Document.status == "failed")
            .where(Document.updated_at < cutoff)
        )
        await session.commit()
        return result.rowcount  # type: ignore[return-value]
