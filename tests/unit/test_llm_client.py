"""
Unit tests for the LLM client abstraction layer.

Tests cover:
  - OpenAIClient delegates to the openai SDK correctly
  - Error handling / retries at the client boundary
  - BaseLLMClient interface contract
  - FakeLLMClient determinism (used elsewhere in the test suite)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_client import BaseLLMClient, OpenAIClient


# ═══════════════════════════════════════════════════════════════════════
#  FakeLLMClient contract
# ═══════════════════════════════════════════════════════════════════════


pytestmark = pytest.mark.unit


class TestFakeLLMClient:
    """Verify the test double itself is reliable."""

    async def test_chat_returns_configured_response(self, fake_llm):
        result = await fake_llm.chat_completion(
            [{"role": "user", "content": "hello"}]
        )
        parsed = json.loads(result)
        assert "parties" in parsed
        assert isinstance(parsed["parties"], list)

    async def test_chat_records_calls(self, fake_llm):
        await fake_llm.chat_completion(
            [{"role": "user", "content": "test"}], model="gpt-4o"
        )
        assert len(fake_llm.chat_calls) == 1
        assert fake_llm.chat_calls[0]["model"] == "gpt-4o"

    async def test_embedding_is_deterministic(self, fake_llm):
        vec1 = await fake_llm.generate_embedding("hello world")
        vec2 = await fake_llm.generate_embedding("hello world")
        assert vec1 == vec2

    async def test_embedding_dimension(self, fake_llm):
        vec = await fake_llm.generate_embedding("test")
        assert len(vec) == 1536

    async def test_different_texts_give_different_vectors(self, fake_llm):
        vec_a = await fake_llm.generate_embedding("alpha")
        vec_b = await fake_llm.generate_embedding("beta")
        assert vec_a != vec_b

    async def test_embedding_is_normalised(self, fake_llm):
        import numpy as np

        vec = await fake_llm.generate_embedding("normalised?")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5


# ═══════════════════════════════════════════════════════════════════════
#  OpenAIClient (mocked SDK)
# ═══════════════════════════════════════════════════════════════════════


class TestOpenAIClient:
    """Test OpenAIClient with a mocked AsyncOpenAI SDK."""

    @pytest.fixture
    def mock_openai_client(self):
        """Patch AsyncOpenAI and return the mock instance."""
        with patch("app.services.llm_client.AsyncOpenAI") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance
            yield mock_instance

    async def test_chat_completion_calls_sdk(self, mock_openai_client):
        # Arrange
        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        client = OpenAIClient()
        result = await client.chat_completion(
            [{"role": "user", "content": "test"}]
        )

        assert result == '{"key": "value"}'
        mock_openai_client.chat.completions.create.assert_awaited_once()

    async def test_chat_completion_empty_content_returns_empty(
        self, mock_openai_client
    ):
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        client = OpenAIClient()
        result = await client.chat_completion(
            [{"role": "user", "content": "test"}]
        )
        assert result == ""

    async def test_generate_embedding_calls_sdk(self, mock_openai_client):
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_openai_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        client = OpenAIClient()
        vec = await client.generate_embedding("test text")

        assert len(vec) == 1536
        assert vec[0] == 0.1
        mock_openai_client.embeddings.create.assert_awaited_once()

    async def test_chat_completion_propagates_exception(self, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        client = OpenAIClient()
        with pytest.raises(Exception, match="API Error"):
            await client.chat_completion(
                [{"role": "user", "content": "fail"}]
            )


# ═══════════════════════════════════════════════════════════════════════
#  Interface contract
# ═══════════════════════════════════════════════════════════════════════


class TestBaseLLMClientInterface:
    """Ensure the ABC cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseLLMClient()
