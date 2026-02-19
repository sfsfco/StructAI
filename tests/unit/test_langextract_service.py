"""
Unit tests for the LangExtract service.

Tests cover:
  - Prompt construction (system prompt, context, query, schema hint)
  - JSON parsing — clean, markdown-fenced, and malformed responses
  - End-to-end extract flow with mocked LLM
"""

from __future__ import annotations

import json

import pytest

from app.services.langextract_service import LangExtractService, SYSTEM_PROMPT


pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════
#  JSON Parsing
# ═══════════════════════════════════════════════════════════════════════


class TestParseJson:
    """Test _parse_json edge cases."""

    def test_clean_json(self):
        raw = '{"name": "Alice", "age": 30}'
        result = LangExtractService._parse_json(raw)
        assert result == {"name": "Alice", "age": 30}

    def test_json_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = LangExtractService._parse_json(raw)
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = LangExtractService._parse_json(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_fence(self):
        raw = '```\n{"items": [1, 2, 3]}\n```'
        result = LangExtractService._parse_json(raw)
        assert result == {"items": [1, 2, 3]}

    def test_invalid_json_returns_raw(self):
        raw = "This is not JSON at all"
        result = LangExtractService._parse_json(raw)
        assert "_raw" in result
        assert "This is not JSON at all" in result["_raw"]

    def test_nested_json(self):
        raw = json.dumps(
            {"parties": [{"name": "Alice"}, {"name": "Bob"}], "valid": True}
        )
        result = LangExtractService._parse_json(raw)
        assert len(result["parties"]) == 2
        assert result["valid"] is True


# ═══════════════════════════════════════════════════════════════════════
#  Extraction Flow
# ═══════════════════════════════════════════════════════════════════════


class TestLangExtractService:
    """Test the extract method with the fake LLM."""

    async def test_extract_basic(self, langextract_service, fake_llm):
        chunks = ["Alice agrees to pay Bob $1000.", "Effective date: Jan 1, 2025."]
        query = "Extract parties and effective date"

        result = await langextract_service.extract(chunks, query)

        # FakeLLMClient returns {"parties": ["Alice", "Bob"], "effective_date": "2025-01-01"}
        assert "parties" in result
        assert "effective_date" in result
        assert len(fake_llm.chat_calls) == 1

    async def test_extract_passes_system_prompt(self, langextract_service, fake_llm):
        await langextract_service.extract(["text"], "query")

        messages = fake_llm.chat_calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    async def test_extract_includes_context_in_user_message(
        self, langextract_service, fake_llm
    ):
        chunks = ["Chunk A", "Chunk B"]
        await langextract_service.extract(chunks, "my query")

        user_msg = fake_llm.chat_calls[0]["messages"][1]["content"]
        assert "Chunk A" in user_msg
        assert "Chunk B" in user_msg
        assert "my query" in user_msg

    async def test_extract_includes_schema_hint(
        self, langextract_service, fake_llm
    ):
        schema = {"name": "string", "amount": "number"}
        await langextract_service.extract(["text"], "query", schema_hint=schema)

        user_msg = fake_llm.chat_calls[0]["messages"][1]["content"]
        assert "Expected output schema" in user_msg
        assert '"name"' in user_msg

    async def test_extract_without_schema_hint(
        self, langextract_service, fake_llm
    ):
        await langextract_service.extract(["text"], "query", schema_hint=None)

        user_msg = fake_llm.chat_calls[0]["messages"][1]["content"]
        assert "Expected output schema" not in user_msg

    async def test_extract_handles_malformed_llm_response(self, fake_llm_custom):
        bad_llm = fake_llm_custom(chat_response="Not valid JSON!")
        service = LangExtractService(bad_llm)

        result = await service.extract(["text"], "query")
        assert "_raw" in result

    async def test_extract_handles_markdown_wrapped_response(
        self, fake_llm_custom
    ):
        response = '```json\n{"status": "ok"}\n```'
        llm = fake_llm_custom(chat_response=response)
        service = LangExtractService(llm)

        result = await service.extract(["text"], "query")
        assert result == {"status": "ok"}
