"""
Vector Store — abstraction for similarity search backends.

Architecture
------------
``BaseVectorStore`` defines the contract that any vector backend must
implement.  The default ``FAISSVectorStore`` wraps Meta's FAISS library
for local, in-process similarity search.

To migrate to a managed service (Pinecone, Weaviate, Qdrant, Milvus,
pgvector, …), implement a new subclass of ``BaseVectorStore`` and
update the factory function ``create_vector_store()``.

The ``VectorStore`` alias is kept for backward compatibility — existing
code that imports ``VectorStore`` continues to work unchanged.

Migration path
--------------
1. Implement ``PineconeVectorStore(BaseVectorStore)`` (or any other).
2. Add a ``VECTOR_STORE_BACKEND`` env var (default: ``faiss``).
3. Update ``create_vector_store()`` to dispatch on the env var.
4. No other code changes needed — all consumers use the base interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import FAISS_INDEX_SIZE, FAISS_SEARCH_DURATION

logger = get_logger(__name__)
settings = get_settings()

INDEX_FILENAME = "index.faiss"


# ══════════════════════════════════════════════════════════════════════════
#  Abstract Base
# ══════════════════════════════════════════════════════════════════════════


class BaseVectorStore(ABC):
    """
    Contract for any vector-search backend.

    All methods that interact with external services should be
    synchronous (FAISS) or async (managed services).  The base keeps
    the interface sync-friendly; managed backends can run I/O in
    thread-pool executors if needed.
    """

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        """Return the number of vectors currently stored."""
        ...

    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add vectors to the store.

        Args:
            vectors: (N, D) float32 array — **must be L2-normalised**.
            metadata: Optional per-vector metadata (used by managed backends).

        Returns:
            List of assigned vector IDs.
        """
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Return the top-k nearest neighbours.

        Args:
            query_vector: (1, D) float32 array — **must be L2-normalised**.
            k: Number of results.

        Returns:
            List of ``(vector_id, score)`` tuples sorted by descending
            similarity.
        """
        ...

    @abstractmethod
    def delete(self, ids: List[int]) -> int:
        """
        Remove vectors by ID.

        Returns the number of vectors actually deleted.
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the index / flush writes."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all vectors and start fresh."""
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return backend-specific health info."""
        return {
            "backend": self.__class__.__name__,
            "total_vectors": self.total_vectors,
        }


# ══════════════════════════════════════════════════════════════════════════
#  FAISS Implementation
# ══════════════════════════════════════════════════════════════════════════


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store with disk persistence.

    Uses ``IndexFlatIP`` (inner-product) which equals cosine similarity
    when vectors are L2-normalised before insertion.

    Limitations
    -----------
    - Single-process write: only one writer at a time (file lock).
    - In-memory: index must fit in RAM.
    - No built-in filtering / metadata queries.

    When these become bottlenecks, migrate to a managed vector DB using
    the ``BaseVectorStore`` interface.
    """

    def __init__(self, dimension: int | None = None, index_dir: str | None = None) -> None:
        self._dimension = dimension or settings.FAISS_DIMENSION
        self._index_dir = Path(index_dir or settings.FAISS_INDEX_DIR)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / INDEX_FILENAME
        self._index: faiss.Index = self._load_or_create()

    # ── BaseVectorStore interface ────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add vectors to the index. Returns the FAISS IDs assigned.

        ``metadata`` is ignored by FAISS (stored externally in PostgreSQL).
        """
        start_id = self._index.ntotal
        self._index.add(vectors)
        ids = list(range(start_id, self._index.ntotal))
        FAISS_INDEX_SIZE.set(self._index.ntotal)
        logger.info("faiss.add", count=len(ids), total=self._index.ntotal)
        return ids

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        with FAISS_SEARCH_DURATION.time():
            distances, indices = self._index.search(query_vector, k)

        results: List[Tuple[int, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # padding when fewer than k vectors exist
            results.append((int(idx), float(dist)))
        return results

    def delete(self, ids: List[int]) -> int:
        """
        FAISS ``IndexFlatIP`` does not support deletion.

        For production, consider ``IndexIDMap`` wrapping or rebuilding
        the index periodically.  Managed vector DBs handle this natively.
        """
        logger.warning(
            "faiss.delete_not_supported",
            ids_count=len(ids),
            hint="Rebuild index or migrate to a managed vector DB",
        )
        return 0

    def save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        logger.info("faiss.save", path=str(self._index_path), total=self._index.ntotal)

    def reset(self) -> None:
        self._index = faiss.IndexFlatIP(self._dimension)
        FAISS_INDEX_SIZE.set(0)
        logger.info("faiss.reset")

    def health_check(self) -> Dict[str, Any]:
        return {
            "backend": "faiss",
            "dimension": self._dimension,
            "total_vectors": self.total_vectors,
            "index_path": str(self._index_path),
            "index_exists_on_disk": self._index_path.exists(),
        }

    # ── Private ──────────────────────────────────────────────────────────

    def _load_or_create(self) -> faiss.Index:
        if self._index_path.exists():
            logger.info("faiss.load", path=str(self._index_path))
            idx = faiss.read_index(str(self._index_path))
            FAISS_INDEX_SIZE.set(idx.ntotal)
            return idx
        logger.info("faiss.create_new", dimension=self._dimension)
        FAISS_INDEX_SIZE.set(0)
        return faiss.IndexFlatIP(self._dimension)


# ══════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════

# Backend registry — add new backends here.
_BACKENDS: Dict[str, type] = {
    "faiss": FAISSVectorStore,
}


def create_vector_store(backend: str | None = None, **kwargs) -> BaseVectorStore:
    """
    Factory function to create the configured vector store backend.

    Args:
        backend: ``"faiss"`` (default), ``"pinecone"``, etc.
        **kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        A ``BaseVectorStore`` implementation.

    Example migration::

        # .env
        VECTOR_STORE_BACKEND=pinecone
        PINECONE_API_KEY=pk-...
        PINECONE_INDEX_NAME=structai

        # In code — nothing changes:
        store = create_vector_store()
        store.add(vectors)
    """
    backend = backend or os.getenv("VECTOR_STORE_BACKEND", "faiss")
    cls = _BACKENDS.get(backend)
    if cls is None:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown vector store backend '{backend}'. "
            f"Available: {available}"
        )
    return cls(**kwargs)


# ── Backward-compatible alias ────────────────────────────────────────────
# Existing code imports ``VectorStore`` — keep it working.
VectorStore = FAISSVectorStore
