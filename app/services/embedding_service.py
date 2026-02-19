"""
Embedding Service — generates embeddings for text using the LLM client.

Handles chunking-aware batch embedding and normalisation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import EMBEDDING_BATCH_SIZE
from app.services.llm_client import BaseLLMClient

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """Produce embedding vectors for text chunks via the LLM client."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    async def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single text string."""
        return await self._llm.generate_embedding(text)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts and return an (N, D) numpy array.

        Processes sequentially for now; can be upgraded to batch API calls.
        """
        logger.info("embedding.batch.start", count=len(texts))
        EMBEDDING_BATCH_SIZE.observe(len(texts))
        vectors: List[List[float]] = []
        for idx, text in enumerate(texts):
            vec = await self._llm.generate_embedding(text)
            vectors.append(vec)
            if (idx + 1) % 50 == 0:
                logger.info("embedding.batch.progress", done=idx + 1, total=len(texts))

        arr = np.array(vectors, dtype=np.float32)
        logger.info("embedding.batch.done", shape=arr.shape)
        return arr
