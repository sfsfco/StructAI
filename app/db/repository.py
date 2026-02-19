"""
Repository layer — data-access abstraction over SQLAlchemy models.

Each repository encapsulates all DB queries for a single aggregate root.
Routes and workers call repositories instead of constructing queries
directly, keeping the domain logic clean and testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, Extraction


# ── Document Repository ─────────────────────────────────────────────────


class DocumentRepository:
    """CRUD operations on the ``documents`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        filename: str,
        content_hash: str,
        status: str = "pending",
    ) -> Document:
        """Insert a new document and return it with its generated ID."""
        doc = Document(
            filename=filename,
            content_hash=content_hash,
            status=status,
        )
        self._session.add(doc)
        await self._session.flush()  # assigns the ID
        return doc

    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Fetch a single document by primary key."""
        result = await self._session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_content_hash(self, content_hash: str) -> Optional[Document]:
        """Look up a document by its SHA-256 content hash (idempotency)."""
        result = await self._session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        document_id: UUID | str,
        status: str,
        total_chunks: Optional[int] = None,
    ) -> None:
        """Update a document's processing status (and optional chunk count)."""
        values: Dict[str, Any] = {"status": status}
        if total_chunks is not None:
            values["total_chunks"] = total_chunks
        await self._session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(**values)
        )

    async def list_all(self, limit: int = 50, offset: int = 0) -> List[Document]:
        """Return a paginated list of documents ordered by creation time."""
        result = await self._session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


# ── Chunk Repository ────────────────────────────────────────────────────


class ChunkRepository:
    """CRUD operations on the ``chunks`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(
        self,
        document_id: UUID | str,
        chunks: List[str],
        faiss_ids: List[int],
    ) -> List[Chunk]:
        """Insert chunks with their FAISS vector IDs in a single flush."""
        objects: List[Chunk] = []
        for idx, (text, fid) in enumerate(zip(chunks, faiss_ids)):
            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                text=text,
                token_count=len(text.split()),
                faiss_vector_id=fid,
            )
            self._session.add(chunk)
            objects.append(chunk)
        await self._session.flush()
        return objects

    async def get_by_faiss_ids(self, faiss_ids: List[int]) -> List[Chunk]:
        """Retrieve chunks matching a list of FAISS vector IDs."""
        result = await self._session.execute(
            select(Chunk).where(Chunk.faiss_vector_id.in_(faiss_ids))
        )
        return list(result.scalars().all())

    async def get_by_document(self, document_id: UUID | str) -> List[Chunk]:
        """Return all chunks belonging to a document, ordered by index."""
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())


# ── Extraction Repository ───────────────────────────────────────────────


class ExtractionRepository:
    """CRUD operations on the ``extractions`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        document_id: UUID | str,
        query: str,
        result_json: str,
        model_used: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Extraction:
        """Persist a new extraction result."""
        extraction = Extraction(
            document_id=document_id,
            query=query,
            result=result_json,
            model_used=model_used,
            latency_ms=latency_ms,
        )
        self._session.add(extraction)
        await self._session.flush()
        return extraction

    async def get_by_id(self, extraction_id: UUID) -> Optional[Extraction]:
        """Fetch a single extraction by ID."""
        result = await self._session.execute(
            select(Extraction).where(Extraction.id == extraction_id)
        )
        return result.scalar_one_or_none()

    async def get_by_document(
        self,
        document_id: UUID | str,
        limit: int = 20,
    ) -> List[Extraction]:
        """Return all extractions for a document, newest first."""
        result = await self._session.execute(
            select(Extraction)
            .where(Extraction.document_id == document_id)
            .order_by(Extraction.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
