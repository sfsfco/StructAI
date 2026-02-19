"""
Integration tests for the database repository layer.

Tests cover:
  - DocumentRepository: create, get_by_id, get_by_content_hash, update_status
  - ChunkRepository: bulk_create, get_by_faiss_ids, get_by_document
  - ExtractionRepository: create, get_by_id, get_by_document

Requires: PostgreSQL running (via docker-compose.test.yml).
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.db.repository import ChunkRepository, DocumentRepository, ExtractionRepository


pytestmark = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════
#  Document Repository
# ═══════════════════════════════════════════════════════════════════════


class TestDocumentRepository:
    """Integration tests for DocumentRepository with real Postgres."""

    async def test_create_document(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        doc = await repo.create(
            filename="contract.pdf",
            content_hash="hash_create_test",
        )

        assert doc.id is not None
        assert doc.filename == "contract.pdf"
        assert doc.status == "pending"

    async def test_get_by_id(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        doc = await repo.create(filename="test.txt", content_hash="hash_get_by_id")

        fetched = await repo.get_by_id(doc.id)
        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.filename == "test.txt"

    async def test_get_by_id_not_found(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        result = await repo.get_by_id(uuid4())
        assert result is None

    async def test_get_by_content_hash(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        await repo.create(filename="doc.pdf", content_hash="unique_hash_lookup")

        found = await repo.get_by_content_hash("unique_hash_lookup")
        assert found is not None
        assert found.filename == "doc.pdf"

    async def test_get_by_content_hash_not_found(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        result = await repo.get_by_content_hash("nonexistent_hash")
        assert result is None

    async def test_update_status(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        doc = await repo.create(filename="status.pdf", content_hash="hash_status_test")

        await repo.update_status(doc.id, "indexed", total_chunks=10)

        updated = await repo.get_by_id(doc.id)
        assert updated.status == "indexed"
        assert updated.total_chunks == 10

    async def test_list_all(self, db_session: AsyncSession):
        repo = DocumentRepository(db_session)
        await repo.create(filename="a.pdf", content_hash="hash_list_a")
        await repo.create(filename="b.pdf", content_hash="hash_list_b")

        docs = await repo.list_all(limit=50)
        assert len(docs) >= 2


# ═══════════════════════════════════════════════════════════════════════
#  Chunk Repository
# ═══════════════════════════════════════════════════════════════════════


class TestChunkRepository:
    """Integration tests for ChunkRepository."""

    async def test_bulk_create_chunks(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="chunked.pdf", content_hash="hash_chunk_bulk"
        )

        chunk_repo = ChunkRepository(db_session)
        chunks = await chunk_repo.bulk_create(
            document_id=doc.id,
            chunks=["chunk one", "chunk two", "chunk three"],
            faiss_ids=[0, 1, 2],
        )

        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[2].text == "chunk three"

    async def test_get_by_faiss_ids(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="faiss_lookup.pdf", content_hash="hash_faiss_lookup"
        )

        chunk_repo = ChunkRepository(db_session)
        await chunk_repo.bulk_create(
            document_id=doc.id,
            chunks=["alpha", "beta", "gamma"],
            faiss_ids=[100, 101, 102],
        )

        found = await chunk_repo.get_by_faiss_ids([100, 102])
        texts = {c.text for c in found}
        assert "alpha" in texts
        assert "gamma" in texts
        assert "beta" not in texts

    async def test_get_by_document(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="by_doc.pdf", content_hash="hash_get_by_doc"
        )

        chunk_repo = ChunkRepository(db_session)
        await chunk_repo.bulk_create(
            document_id=doc.id,
            chunks=["first", "second"],
            faiss_ids=[200, 201],
        )

        chunks = await chunk_repo.get_by_document(doc.id)
        assert len(chunks) == 2
        assert chunks[0].chunk_index < chunks[1].chunk_index


# ═══════════════════════════════════════════════════════════════════════
#  Extraction Repository
# ═══════════════════════════════════════════════════════════════════════


class TestExtractionRepository:
    """Integration tests for ExtractionRepository."""

    async def test_create_extraction(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="extract.pdf", content_hash="hash_extraction_create"
        )

        ext_repo = ExtractionRepository(db_session)
        extraction = await ext_repo.create(
            document_id=doc.id,
            query="Extract parties",
            result_json=json.dumps({"parties": ["Alice"]}),
            model_used="gpt-4o",
            latency_ms=123.45,
        )

        assert extraction.id is not None
        assert extraction.query == "Extract parties"
        assert extraction.model_used == "gpt-4o"

    async def test_get_by_id(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="ext_get.pdf", content_hash="hash_ext_get_by_id"
        )

        ext_repo = ExtractionRepository(db_session)
        created = await ext_repo.create(
            document_id=doc.id,
            query="test query",
            result_json="{}",
        )

        fetched = await ext_repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_by_document(self, db_session: AsyncSession):
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="ext_list.pdf", content_hash="hash_ext_list"
        )

        ext_repo = ExtractionRepository(db_session)
        await ext_repo.create(
            document_id=doc.id,
            query="query 1",
            result_json="{}",
        )
        await ext_repo.create(
            document_id=doc.id,
            query="query 2",
            result_json="{}",
        )

        extractions = await ext_repo.get_by_document(doc.id)
        assert len(extractions) == 2
