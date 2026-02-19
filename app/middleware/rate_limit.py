"""
Rate-limiting middleware using SlowAPI.

Limits are applied per-client IP address and can be tuned via the
``RATE_LIMIT_DEFAULT`` setting (e.g. "20/minute").

Heavy endpoints like ``/extract`` use a stricter limit than the global
default so that LLM costs stay under control.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings

settings = get_settings()

# ── Limiter instance (import in routes to decorate endpoints) ────────────

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri=settings.redis_url,
)


async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a structured 429 response when the client exceeds the limit."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": exc.detail,
        },
        headers={"Retry-After": str(exc.detail)},
    )
