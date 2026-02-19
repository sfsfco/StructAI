"""
SQLAlchemy ORM models for the AI extraction platform.

Tables:
  - documents:   ingested document metadata
  - chunks:      text chunks produced by document splitting
  - extractions: structured extraction results
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


class Document(Base):
    """Represents an ingested document."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    status = Column(
        SAEnum("pending", "processing", "indexed", "failed", name="doc_status"),
        default="pending",
        nullable=False,
    )
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    extractions = relationship(
        "Extraction", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document {self.filename} [{self.status}]>"


class Chunk(Base):
    """A text chunk belonging to a document, with an embedding vector ID."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    faiss_vector_id = Column(Integer, nullable=True)  # FAISS internal ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk doc={self.document_id} idx={self.chunk_index}>"


class Extraction(Base):
    """Stores a structured extraction result for a document."""

    __tablename__ = "extractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    query = Column(Text, nullable=False)
    result = Column(Text, nullable=False)  # JSON string of extracted data
    model_used = Column(String(128), nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="extractions")

    def __repr__(self) -> str:
        return f"<Extraction doc={self.document_id} query={self.query[:30]}>"
