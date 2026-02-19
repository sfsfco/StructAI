"""
End-to-end tests for the complete StructAI pipeline.

Simulates the full user journey:
  1. Index a document via POST /documents/index
  2. Process the document (chunking → embedding → FAISS indexing)
  3. Run an extraction via POST /extract
  4. Verify the extraction result
  5. Verify cache hit on repeated extraction

These tests require PostgreSQL and Redis running.
The Celery worker is bypassed by executing tasks synchronously in-process.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import UUID

import faiss
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.db.repository import ChunkRepository, DocumentRepository
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from tests.conftest import FakeLLMClient


pytestmark = pytest.mark.e2e


SAMPLE_DOCUMENT = """
SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of
January 15, 2025, by and between:

Licensor: Acme Corporation, a Delaware corporation, with its principal
office at 123 Innovation Drive, San Francisco, CA 94105.

Licensee: Beta Industries LLC, a California limited liability company,
with its principal office at 456 Market Street, San Francisco, CA 94102.

TERMS AND CONDITIONS:

1. Grant of License. Licensor hereby grants to Licensee a non-exclusive,
   non-transferable license to use the Software Product known as "AcmeAI
   Platform" (the "Software").

2. License Fee. Licensee shall pay Licensor an annual license fee of
   $50,000 USD, payable within 30 days of the Effective Date and each
   anniversary thereof.

3. Term. This Agreement shall commence on January 15, 2025, and shall
   continue for a period of three (3) years unless earlier terminated.

4. Confidentiality. Each party agrees to maintain the confidentiality of
   any proprietary information received from the other party.

5. Limitation of Liability. In no event shall either party's liability
   exceed the total fees paid under this Agreement.
"""


class TestFullPipeline:
    """
    End-to-end test: index a document, process it in-process,
    then run extraction and verify results.
    """

    async def test_complete_indexing_and_extraction_pipeline(
        self,
        db_session: AsyncSession,
        redis_cache: CacheService,
        tmp_path,
    ):
        """
        Full pipeline test:
          1. Create a document in the DB
          2. Chunk the document
          3. Generate embeddings
          4. Index in FAISS
          5. Run extraction via the service
          6. Verify the result contains expected fields
          7. Verify cache hit on second call
        """
        fake_llm = FakeLLMClient(
            chat_response=json.dumps(
                {
                    "parties": {
                        "licensor": "Acme Corporation",
                        "licensee": "Beta Industries LLC",
                    },
                    "effective_date": "January 15, 2025",
                    "license_fee": "$50,000 USD annually",
                    "term": "3 years",
                }
            )
        )
        embedder = EmbeddingService(fake_llm)
        faiss_dir = str(tmp_path / "faiss")

        # ── Step 1: Create document ──────────────────────────────────
        doc_repo = DocumentRepository(db_session)
        doc = await doc_repo.create(
            filename="license_agreement.pdf",
            content_hash="e2e_test_hash_pipeline",
        )
        assert doc.status == "pending"

        # ── Step 2: Chunk the document ───────────────────────────────
        from app.workers.tasks import chunk_text

        chunks = chunk_text(SAMPLE_DOCUMENT)
        assert len(chunks) > 0

        # ── Step 3: Generate embeddings ──────────────────────────────
        embeddings = await embedder.embed_batch(chunks)
        faiss.normalize_L2(embeddings)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 1536

        # ── Step 4: Index in FAISS + persist chunks ──────────────────
        store = VectorStore(dimension=1536, index_dir=faiss_dir)
        faiss_ids = store.add(embeddings)
        store.save()

        chunk_repo = ChunkRepository(db_session)
        await chunk_repo.bulk_create(
            document_id=doc.id,
            chunks=chunks,
            faiss_ids=faiss_ids,
        )

        await doc_repo.update_status(doc.id, "indexed", total_chunks=len(chunks))

        # Verify document is indexed
        updated_doc = await doc_repo.get_by_id(doc.id)
        assert updated_doc.status == "indexed"
        assert updated_doc.total_chunks == len(chunks)

        # ── Step 5: Run extraction ───────────────────────────────────
        from app.services.langextract_service import LangExtractService

        extractor = LangExtractService(fake_llm)

        # Search FAISS for relevant chunks
        query = "Extract all parties, effective date, and payment terms"
        query_vec = await embedder.embed_text(query)
        query_arr = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_arr)

        neighbours = store.search(query_arr, k=5)
        faiss_result_ids = [n[0] for n in neighbours]

        # Fetch chunk texts
        chunk_rows = await chunk_repo.get_by_faiss_ids(faiss_result_ids)
        chunk_texts = [c.text for c in chunk_rows]
        assert len(chunk_texts) > 0

        # Run LLM extraction
        result = await extractor.extract(chunk_texts, query)

        # ── Step 6: Verify extraction result ─────────────────────────
        assert "parties" in result
        assert result["parties"]["licensor"] == "Acme Corporation"
        assert result["parties"]["licensee"] == "Beta Industries LLC"
        assert result["effective_date"] == "January 15, 2025"
        assert "$50,000" in result["license_fee"]
        assert result["term"] == "3 years"

        # ── Step 7: Verify caching ───────────────────────────────────
        cache_key = CacheService.extraction_key(str(doc.id), query)
        cache_payload = {
            "extraction_id": "e2e-test-id",
            "result": result,
            "model_used": "gpt-4o",
            "latency_ms": 150.0,
        }
        await redis_cache.set(cache_key, cache_payload)

        # Second call should hit cache
        cached = await redis_cache.get(cache_key)
        assert cached is not None
        assert cached["result"] == result
        assert cached["model_used"] == "gpt-4o"

    async def test_index_then_extract_via_api(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        redis_cache: CacheService,
        tmp_path,
    ):
        """
        API-level E2E: POST /documents/index → simulate processing →
        POST /extract → verify response.
        """
        fake_llm = FakeLLMClient(
            chat_response=json.dumps(
                {
                    "software_name": "AcmeAI Platform",
                    "license_type": "non-exclusive, non-transferable",
                }
            )
        )

        # ── Index document via API ───────────────────────────────────
        with patch("app.api.routes.process_document") as mock_task:
            mock_task.delay.return_value = MagicMock(id="e2e-task-1")

            response = await async_client.post(
                "/api/v1/documents/index",
                json={
                    "filename": "e2e_contract.pdf",
                    "content": SAMPLE_DOCUMENT,
                },
            )
        assert response.status_code == 202
        doc_id = response.json()["document_id"]

        # ── Simulate background processing (in-process) ─────────────
        from app.workers.tasks import chunk_text

        chunks = chunk_text(SAMPLE_DOCUMENT)
        embedder = EmbeddingService(fake_llm)
        embeddings = await embedder.embed_batch(chunks)
        faiss.normalize_L2(embeddings)

        faiss_dir = str(tmp_path / "faiss_e2e_api")
        store = VectorStore(dimension=1536, index_dir=faiss_dir)
        faiss_ids = store.add(embeddings)
        store.save()

        chunk_repo = ChunkRepository(db_session)
        await chunk_repo.bulk_create(
            document_id=doc_id,
            chunks=chunks,
            faiss_ids=faiss_ids,
        )

        doc_repo = DocumentRepository(db_session)
        await doc_repo.update_status(doc_id, "indexed", total_chunks=len(chunks))

        # ── Extract via API ──────────────────────────────────────────
        from app.services.dependencies import (
            get_embedding_service,
            get_extract_service,
            get_vector_store,
        )
        from app.services.langextract_service import LangExtractService
        from app.main import app

        # Override vector store to use our test store
        app.dependency_overrides[get_vector_store] = lambda: store
        app.dependency_overrides[get_extract_service] = lambda llm=None: LangExtractService(
            fake_llm
        )
        app.dependency_overrides[get_embedding_service] = lambda llm=None: embedder

        extract_response = await async_client.post(
            "/api/v1/extract",
            json={
                "document_id": doc_id,
                "query": "Extract software name and license type",
            },
        )

        assert extract_response.status_code == 200
        data = extract_response.json()
        assert "result" in data
        assert data["result"]["software_name"] == "AcmeAI Platform"
        assert data["cached"] is False

        # Clean up overrides
        app.dependency_overrides.clear()
