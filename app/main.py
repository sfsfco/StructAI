"""
FastAPI application entrypoint.

Responsibilities:
  - Mount API routes
  - Set up lifespan events (startup / shutdown)
  - Configure CORS, error handlers, middleware
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.metrics import APP_INFO
from app.db.session import init_db
from app.middleware.backpressure import BackpressureMiddleware
from app.middleware.correlation import CorrelationIDMiddleware
from app.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from app.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from app.services.cache_service import CacheService
from app.services.vector_store import VectorStore

settings = get_settings()


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup and shutdown hooks."""
    # ── Startup ──────────────────────────────────────────────────────
    setup_logging(
        log_level=settings.LOG_LEVEL,
        json_logs=not settings.DEBUG,
    )
    logger = get_logger("startup")
    logger.info("app.starting", version=settings.APP_VERSION)

    # Publish app metadata to Prometheus info metric
    APP_INFO.info({
        "version": settings.APP_VERSION,
        "name": settings.APP_NAME,
        "openai_model": settings.OPENAI_MODEL,
    })

    # Ensure database tables exist (dev convenience; use Alembic in prod)
    # Import models so Base.metadata is populated before create_all
    import app.db.models  # noqa: F401
    await init_db()
    logger.info("app.db_initialised")

    # Warm up Redis connection
    cache = CacheService()
    await cache.connect()
    application.state.cache = cache

    # Pre-load or create the FAISS index
    store = VectorStore()
    application.state.vector_store = store
    logger.info("app.faiss_loaded", total_vectors=store.total_vectors)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    await cache.disconnect()
    logger.info("app.shutdown")


# ── Application ──────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered structured data extraction service",
    lifespan=lifespan,
)

# ── Middleware (order matters: outermost first) ──────────────────────────

# Prometheus metrics — must be outermost to capture full request duration
app.add_middleware(PrometheusMiddleware)

# Backpressure — shed load when system is overloaded
app.add_middleware(
    BackpressureMiddleware,
    max_inflight=settings.BACKPRESSURE_MAX_INFLIGHT,
    max_queue_depth=settings.BACKPRESSURE_MAX_QUEUE_DEPTH,
)

# Correlation-ID — attaches a unique ID to every request / response
app.add_middleware(CorrelationIDMiddleware)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Mount all routes
app.include_router(router, prefix="/api/v1")

# Prometheus metrics scrape endpoint (outside /api/v1 prefix)
app.add_route("/metrics", metrics_endpoint)


# ── Global Error Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    from app.core.logging import correlation_id_ctx
    logger = get_logger("error_handler")
    cid = correlation_id_ctx.get()
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path,
        correlation_id=cid,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "correlation_id": cid},
    )
