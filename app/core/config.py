"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe config with .env support.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — every value comes from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "StructAI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── OpenAI ───────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.0

    # ── PostgreSQL ───────────────────────────────────────────────────────
    POSTGRES_USER: str = "ai_user"
    POSTGRES_PASSWORD: str = "ai_pass"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_db"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 3600  # seconds

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── Celery ───────────────────────────────────────────────────────────
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.redis_url

    @property
    def celery_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.redis_url

    # ── FAISS ────────────────────────────────────────────────────────────
    FAISS_INDEX_DIR: str = "/app/data/faiss"
    FAISS_DIMENSION: int = 1536  # text-embedding-3-small dimension

    # ── Rate Limiting ─────────────────────────────────────────────────
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_EXTRACT: str = "10/minute"

    # ── Document Processing ──────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Embedding Cache ──────────────────────────────────────────────────
    EMBEDDING_CACHE_TTL: int = 86400  # 24 hours (deterministic per model+text)

    # ── Backpressure ─────────────────────────────────────────────────────
    BACKPRESSURE_MAX_INFLIGHT: int = 100
    BACKPRESSURE_MAX_QUEUE_DEPTH: int = 500

    # ── Vector Store ─────────────────────────────────────────────────────
    VECTOR_STORE_BACKEND: str = "faiss"  # "faiss" | "pinecone" | "qdrant" ...

    # ── Scaling ──────────────────────────────────────────────────────────
    API_WORKERS: int = 4          # gunicorn worker count (WEB_CONCURRENCY)
    WORKER_MAX_TASKS_PER_CHILD: int = 100  # recycle after N tasks
    WORKER_PREFETCH_MULTIPLIER: int = 1    # one task at a time per worker


@lru_cache()
def get_settings() -> Settings:
    """Singleton accessor — cached so env is read only once."""
    return Settings()
