"""
Pydantic request / response schemas for the extraction platform API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Document Schemas ─────────────────────────────────────────────────────

class DocumentIndexRequest(BaseModel):
    """Request body for POST /documents/index."""
    filename: str = Field(..., description="Original filename of the document")
    content: str = Field(..., description="Raw text content of the document")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "filename": "contract.pdf",
                "content": "This Software License Agreement is entered into ..."
            }
        ]
    }}


class DocumentIndexResponse(BaseModel):
    """Response body for POST /documents/index."""
    document_id: UUID
    status: DocumentStatus
    message: str = "Document queued for indexing"


class DocumentDetail(BaseModel):
    """Detailed document info returned by GET endpoints."""
    id: UUID
    filename: str
    status: DocumentStatus
    total_chunks: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Extraction Schemas ───────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    """Request body for POST /extract."""
    document_id: UUID = Field(..., description="ID of the indexed document")
    query: str = Field(
        ...,
        description="Natural-language query describing what to extract",
        min_length=3,
    )
    schema_hint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON schema hint for the expected output structure",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "Extract all parties, effective date, and payment terms",
                "schema_hint": {
                    "parties": ["string"],
                    "effective_date": "string",
                    "payment_terms": "string",
                },
            }
        ]
    }}


class ExtractionResponse(BaseModel):
    """Response body for POST /extract."""
    extraction_id: UUID
    document_id: UUID
    query: str
    result: Dict[str, Any]
    model_used: Optional[str] = None
    latency_ms: Optional[float] = None
    cached: bool = False


# ── Task Status ──────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskStatusResponse(BaseModel):
    """Response body for GET /tasks/{task_id}."""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    date_done: Optional[str] = None


# ── Health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health (liveness probe)."""
    status: str = "ok"
    version: str
    uptime_seconds: Optional[float] = None
    db: str = "unknown"
    redis: str = "unknown"
    faiss_index_loaded: bool = False


class ReadinessResponse(BaseModel):
    """Response body for GET /ready (readiness probe)."""
    ready: bool = False
    checks: Dict[str, Any] = Field(default_factory=dict)
    version: str = ""


# ── Error ────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str
    correlation_id: Optional[str] = None
