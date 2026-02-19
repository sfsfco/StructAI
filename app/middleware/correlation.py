"""
Correlation-ID middleware.

Attaches a unique correlation ID to every request, making it available via
the context variable in `app.core.logging`. If the client sends a header
``X-Correlation-ID`` it is reused; otherwise a new UUID is generated.

The ID is returned in the response header ``X-Correlation-ID`` and is
injected into every structured log emitted during the request lifecycle.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import correlation_id_ctx

_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject / propagate a correlation ID on every HTTP request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Prefer client-supplied ID, else generate one
        cid = request.headers.get(_HEADER) or uuid.uuid4().hex
        correlation_id_ctx.set(cid)

        response = await call_next(request)
        response.headers[_HEADER] = cid
        return response
