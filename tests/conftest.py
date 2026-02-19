"""
Root-level test fixtures shared across all test layers.

Provides:
  - Fake / mock LLM client (no real OpenAI calls)
  - In-memory FAISS vector store
  - Async DB session with SQLite (unit) or Postgres (integration)
  - Fake Redis cache (unit) or real Redis (integration)
  - FastAPI TestClient configured with dependency overrides
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.services.llm_client import BaseLLMClient

# ── Ensure test env vars are set before anything imports config ───────
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ═══════════════════════════════════════════════════════════════════════
#  Fake LLM Client
# ═══════════════════════════════════════════════════════════════════════


class FakeLLMClient(BaseLLMClient):
    """
    Deterministic LLM client for testing.

    - chat_completion returns a configurable JSON string.
    - generate_embedding returns a reproducible vector seeded by input hash.
    """

    def __init__(
        self,
        chat_response: str | None = None,
        embedding_dim: int = 1536,
    ) -> None:
        self._chat_response = chat_response or json.dumps(
            {"parties": ["Alice", "Bob"], "effective_date": "2025-01-01"}
        )
        self._embedding_dim = embedding_dim
        self.chat_calls: List[Dict[str, Any]] = []
        self.embed_calls: List[str] = []

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.chat_calls.append(
            {"messages": messages, "model": model, "temperature": temperature}
        )
        return self._chat_response

    async def generate_embedding(
        self,
        text: str,
        *,
        model: Optional[str] = None,
    ) -> List[float]:
        self.embed_calls.append(text)
        # Deterministic vector: hash text → seed → random vector → normalise
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._embedding_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """Return a fresh FakeLLMClient instance."""
    return FakeLLMClient()


@pytest.fixture
def fake_llm_custom():
    """Factory fixture to build a FakeLLMClient with custom chat response."""

    def _factory(chat_response: str, **kwargs) -> FakeLLMClient:
        return FakeLLMClient(chat_response=chat_response, **kwargs)

    return _factory


@pytest.fixture
def tmp_faiss_dir(tmp_path):
    """Return a temporary directory for FAISS index files."""
    d = tmp_path / "faiss"
    d.mkdir()
    return str(d)


@pytest.fixture
def vector_store(tmp_faiss_dir):
    """Return a VectorStore backed by a temporary directory."""
    from app.services.vector_store import VectorStore

    return VectorStore(dimension=1536, index_dir=tmp_faiss_dir)


@pytest.fixture
def embedding_service(fake_llm) -> "EmbeddingService":
    """Return an EmbeddingService wired to the fake LLM."""
    from app.services.embedding_service import EmbeddingService

    return EmbeddingService(fake_llm)


@pytest.fixture
def langextract_service(fake_llm) -> "LangExtractService":
    """Return a LangExtractService wired to the fake LLM."""
    from app.services.langextract_service import LangExtractService

    return LangExtractService(fake_llm)
