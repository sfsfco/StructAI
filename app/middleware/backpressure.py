"""
Backpressure middleware — protects the system under heavy load.

Monitors two pressure signals:

1. **Celery queue depth** — if the indexing queue exceeds a threshold,
   new document indexing requests receive HTTP 503 (Service Unavailable)
   rather than piling up unbounded work.

2. **In-flight request count** — if the API server already has too many
   concurrent requests, additional requests are shed with HTTP 503.

Both thresholds are configurable via environment variables
(``BACKPRESSURE_MAX_QUEUE_DEPTH`` and ``BACKPRESSURE_MAX_INFLIGHT``).

Why this matters
----------------
Without backpressure:
  - Queue depth grows unbounded → worker lag increases → timeouts
  - Memory consumption grows → OOM kills
  - LLM API costs spike uncontrollably

With backpressure:
  - Clients receive a fast 503 and can retry with exponential back-off
  - System stays within its capacity envelope
  - Dashboards/alerts fire on 503 rate instead of cascading failures
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import REGISTRY

logger = get_logger(__name__)
settings = get_settings()

# Paths that trigger queue-depth checks (write operations that enqueue work)
_QUEUE_GUARDED_PATHS = {"/api/v1/documents/index"}

# Paths exempt from in-flight limiting (ops endpoints should always respond)
_EXEMPT_PATHS = {"/api/v1/health", "/api/v1/ready", "/metrics"}


class BackpressureMiddleware(BaseHTTPMiddleware):
    """
    Shed load when the system is under pressure.

    Checks:
      1. Global in-flight request count against ``max_inflight``.
      2. Celery queue depth against ``max_queue_depth`` (only for
         write endpoints that create background work).
    """

    def __init__(
        self,
        app,
        *,
        max_inflight: int = 0,
        max_queue_depth: int = 0,
        queue_check_interval: float = 5.0,
    ) -> None:
        super().__init__(app)
        self._max_inflight = max_inflight or int(
            settings.__dict__.get("BACKPRESSURE_MAX_INFLIGHT", 100)
        )
        self._max_queue_depth = max_queue_depth or int(
            settings.__dict__.get("BACKPRESSURE_MAX_QUEUE_DEPTH", 500)
        )
        self._queue_check_interval = queue_check_interval

        self._inflight = 0
        self._lock = asyncio.Lock()

        # Cached queue depth (refreshed at most every N seconds)
        self._cached_queue_depth: int = 0
        self._last_queue_check: float = 0.0

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Always allow ops endpoints through
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        # ── In-flight check ──────────────────────────────────────────
        async with self._lock:
            self._inflight += 1
            current = self._inflight

        if current > self._max_inflight:
            async with self._lock:
                self._inflight -= 1
            logger.warning(
                "backpressure.inflight_exceeded",
                inflight=current,
                max=self._max_inflight,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Server is at capacity. Please retry later.",
                    "reason": "max_inflight_exceeded",
                },
                headers={"Retry-After": "5"},
            )

        # ── Queue-depth check (only for write paths) ─────────────────
        if path in _QUEUE_GUARDED_PATHS:
            depth = await self._get_queue_depth()
            if depth > self._max_queue_depth:
                async with self._lock:
                    self._inflight -= 1
                logger.warning(
                    "backpressure.queue_depth_exceeded",
                    queue_depth=depth,
                    max=self._max_queue_depth,
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "Processing queue is full. Please retry later.",
                        "reason": "queue_depth_exceeded",
                        "queue_depth": depth,
                    },
                    headers={"Retry-After": "30"},
                )

        try:
            return await call_next(request)
        finally:
            async with self._lock:
                self._inflight -= 1

    async def _get_queue_depth(self) -> int:
        """
        Return the approximate Celery queue depth.

        Cached for ``_queue_check_interval`` seconds to avoid hammering
        Redis on every request.
        """
        now = time.monotonic()
        if now - self._last_queue_check < self._queue_check_interval:
            return self._cached_queue_depth

        try:
            from app.workers.tasks import celery_app

            with celery_app.connection_or_acquire() as conn:
                depth = 0
                for queue_name in ("default", "indexing"):
                    channel = conn.channel()
                    _, msg_count, _ = channel.queue_declare(
                        queue=queue_name, passive=True
                    )
                    depth += msg_count
            self._cached_queue_depth = depth
        except Exception as exc:
            # If we can't reach the broker, don't block — assume OK
            logger.debug("backpressure.queue_check_failed", error=str(exc))
            self._cached_queue_depth = 0

        self._last_queue_check = now
        return self._cached_queue_depth
