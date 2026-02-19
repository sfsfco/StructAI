"""
LLM Client — abstraction over the OpenAI API.

Provides a clean interface so the rest of the application never imports
`openai` directly.  Swapping providers (Azure OpenAI, Anthropic, local
models) only requires a new implementation of `BaseLLMClient`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import LLM_CALL_DURATION, LLM_CALLS_TOTAL, LLM_TOKENS_TOTAL

logger = get_logger(__name__)
settings = get_settings()


class BaseLLMClient(ABC):
    """Interface that any LLM provider must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the assistant message content from a chat completion."""
        ...

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        *,
        model: Optional[str] = None,
    ) -> List[float]:
        """Return an embedding vector for the given text."""
        ...


class OpenAIClient(BaseLLMClient):
    """
    Concrete OpenAI implementation.

    Uses the official `openai` async client under the hood.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        model = model or settings.OPENAI_MODEL
        temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS

        logger.info(
            "llm.chat_completion.start",
            model=model,
            message_count=len(messages),
        )

        with LLM_CALL_DURATION.labels(operation="chat_completion", model=model).time():
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                LLM_CALLS_TOTAL.labels(
                    operation="chat_completion", model=model, status="success"
                ).inc()
            except Exception:
                LLM_CALLS_TOTAL.labels(
                    operation="chat_completion", model=model, status="error"
                ).inc()
                raise

        content = response.choices[0].message.content or ""

        # Track token usage
        if response.usage:
            LLM_TOKENS_TOTAL.labels(model=model, type="prompt").inc(
                response.usage.prompt_tokens
            )
            LLM_TOKENS_TOTAL.labels(model=model, type="completion").inc(
                response.usage.completion_tokens
            )

        logger.info(
            "llm.chat_completion.done",
            model=model,
            usage=response.usage.model_dump() if response.usage else {},
        )
        return content

    async def generate_embedding(
        self,
        text: str,
        *,
        model: Optional[str] = None,
    ) -> List[float]:
        model = model or settings.OPENAI_EMBEDDING_MODEL

        with LLM_CALL_DURATION.labels(operation="embedding", model=model).time():
            try:
                response = await self._client.embeddings.create(
                    model=model,
                    input=text,
                )
                LLM_CALLS_TOTAL.labels(
                    operation="embedding", model=model, status="success"
                ).inc()
            except Exception:
                LLM_CALLS_TOTAL.labels(
                    operation="embedding", model=model, status="error"
                ).inc()
                raise

        if response.usage:
            LLM_TOKENS_TOTAL.labels(model=model, type="prompt").inc(
                response.usage.total_tokens
            )

        return response.data[0].embedding


def get_llm_client() -> BaseLLMClient:
    """Factory — returns the configured LLM client (currently OpenAI)."""
    return OpenAIClient()
