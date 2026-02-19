"""
Prometheus metrics middleware for FastAPI.

Tracks per-request latency, counts, and in-flight requests.
Exposes a ``/metrics`` endpoint for Prometheus scraping.

Usage::

    from app.middleware.metrics import PrometheusMiddleware, metrics_endpoint

    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", metrics_endpoint)
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    metrics_output,
)


def _normalise_path(path: str) -> str:
    """
    Collapse high-cardinality path segments to avoid metric explosion.

    ``/api/v1/tasks/abc-123``  →  ``/api/v1/tasks/{id}``
    ``/api/v1/documents/abc``  →  ``/api/v1/documents/{id}``
    """
    parts = path.rstrip("/").split("/")
    normalised: list[str] = []
    for part in parts:
        # UUIDs (hex with dashes) and other opaque IDs
        if len(part) >= 8 and not part.startswith("v"):
            try:
                int(part, 16)
                normalised.append("{id}")
                continue
            except ValueError:
                pass
            # Celery task IDs (uuid4 with dashes)
            if "-" in part and len(part) == 36:
                normalised.append("{id}")
                continue
        normalised.append(part)
    return "/".join(normalised) or "/"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record HTTP request metrics for Prometheus."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip the metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Record 500 for unhandled exceptions that escape middleware
            elapsed = time.perf_counter() - start
            path = _normalise_path(request.url.path)
            HTTP_REQUEST_DURATION.labels(
                method=method, path=path, status_code="500"
            ).observe(elapsed)
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status_code="500"
            ).inc()
            raise
        finally:
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method).dec()

        elapsed = time.perf_counter() - start
        path = _normalise_path(request.url.path)
        status_code = str(response.status_code)

        HTTP_REQUEST_DURATION.labels(
            method=method, path=path, status_code=status_code
        ).observe(elapsed)
        HTTP_REQUESTS_TOTAL.labels(
            method=method, path=path, status_code=status_code
        ).inc()

        # Inject timing header for easy debugging
        response.headers["X-Response-Time"] = f"{elapsed:.4f}s"

        return response


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape target — returns metrics in text exposition format."""
    return Response(
        content=metrics_output(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
