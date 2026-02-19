"""
Integration-test fixtures.

Provides:
  - Async PostgreSQL test database (via async SQLAlchemy + real Postgres)
  - Real Redis client for cache integration tests
  - FastAPI TestClient with dependency overrides
  - Auto-cleanup between tests
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tests.conftest import FakeLLMClient

# ── Test database URL — uses a separate DB to avoid clobbering dev ──────
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://ai_user:ai_pass@localhost:5432/ai_db_test",
)
TEST_REDIS_URL = os.environ.get(
    "TEST_REDIS_URL",
    "redis://localhost:6379/1",  # Use DB 1 to isolate from dev
)


# ═══════════════════════════════════════════════════════════════════════
#  Database Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create a test engine that lives for the entire test session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create all tables
    from app.db.session import Base

    # Ensure models are imported so metadata is populated
    import app.db.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Teardown: drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional DB session that rolls back after each test.

    This ensures test isolation without needing to truncate tables.
    """
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        async with session.begin():
            yield session
            # Rollback to undo any changes made during the test
            await session.rollback()


# ═══════════════════════════════════════════════════════════════════════
#  Redis Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def redis_cache():
    """Provide a real Redis CacheService connected to the test DB."""
    from app.services.cache_service import CacheService

    cache = CacheService(redis_url=TEST_REDIS_URL)
    await cache.connect()

    yield cache

    # Clean up test keys
    if cache._redis:
        await cache._redis.flushdb()
    await cache.disconnect()


# ═══════════════════════════════════════════════════════════════════════
#  FastAPI Test Client
# ═══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def async_client(
    db_session: AsyncSession,
    redis_cache,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an httpx AsyncClient pointing at the FastAPI app with
    dependency overrides for DB and cache.
    """
    from app.db.repository import ChunkRepository, DocumentRepository, ExtractionRepository
    from app.db.session import get_db
    from app.main import app
    from app.services.dependencies import (
        get_cache,
        get_embedding_service,
        get_extract_service,
        get_llm,
        get_vector_store,
    )
    from app.services.embedding_service import EmbeddingService
    from app.services.langextract_service import LangExtractService
    from app.services.vector_store import VectorStore

    fake_llm = FakeLLMClient()

    # Override dependencies
    async def override_get_db():
        yield db_session

    async def override_get_cache(request=None):
        return redis_cache

    def override_get_llm(request=None):
        return fake_llm

    def override_get_embedding_service(llm=None):
        return EmbeddingService(fake_llm)

    def override_get_extract_service(llm=None):
        return LangExtractService(fake_llm)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_cache] = override_get_cache
    app.dependency_overrides[get_llm] = override_get_llm
    app.dependency_overrides[get_embedding_service] = override_get_embedding_service
    app.dependency_overrides[get_extract_service] = override_get_extract_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()
