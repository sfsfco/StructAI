"""
FastAPI dependency injection for services.

Provides ``Depends(...)``-compatible callables that wire up service
instances so routes never have to instantiate them manually.  This keeps
the route layer thin and makes services trivially swappable in tests.
"""

from typing import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import ChunkRepository, DocumentRepository, ExtractionRepository
from app.db.session import get_db
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService
from app.services.langextract_service import LangExtractService
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.vector_store import VectorStore


# ── LLM Client ──────────────────────────────────────────────────────────


def get_llm() -> BaseLLMClient:
    """Return the configured LLM client (singleton-friendly)."""
    return get_llm_client()


# ── Embedding Service ───────────────────────────────────────────────────


def get_embedding_service(
    llm: BaseLLMClient = Depends(get_llm),
) -> EmbeddingService:
    return EmbeddingService(llm)


# ── LangExtract Service ─────────────────────────────────────────────────


def get_extract_service(
    llm: BaseLLMClient = Depends(get_llm),
) -> LangExtractService:
    return LangExtractService(llm)


# ── Vector Store ─────────────────────────────────────────────────────────


def get_vector_store() -> VectorStore:
    """Return a VectorStore instance (loads/creates the FAISS index)."""
    return VectorStore()


# ── Cache Service (from app.state) ──────────────────────────────────────


async def get_cache(request: Request) -> CacheService:
    """
    Return the cache service attached to the application state at startup.
    Falls back to creating a new transient connection if state is unavailable.
    """
    cache: CacheService | None = getattr(request.app.state, "cache", None)
    if cache is not None:
        return cache

    # Fallback: create a fresh connection (should not happen in production)
    cache = CacheService()
    await cache.connect()
    return cache


# ── Repositories ─────────────────────────────────────────────────────────


def get_document_repo(
    db: AsyncSession = Depends(get_db),
) -> DocumentRepository:
    return DocumentRepository(db)


def get_chunk_repo(
    db: AsyncSession = Depends(get_db),
) -> ChunkRepository:
    return ChunkRepository(db)


def get_extraction_repo(
    db: AsyncSession = Depends(get_db),
) -> ExtractionRepository:
    return ExtractionRepository(db)
