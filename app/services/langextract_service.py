"""
LangExtract Service — structured data extraction from text using LLM.

Wraps the LangExtract library to:
  1. Accept raw text + a user query (+ optional schema hint)
  2. Build a prompt that instructs the LLM to return structured JSON
  3. Parse and validate the LLM response
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.services.llm_client import BaseLLMClient

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Given a set of text chunks and a user query, extract the requested "
    "information and return it as valid JSON. "
    "Only include fields that are explicitly requested or strongly implied. "
    "If a value cannot be determined, set it to null."
)


class LangExtractService:
    """
    Orchestrates structured extraction:
      - receives relevant text chunks (from FAISS retrieval)
      - builds a prompt with the user query + optional schema hint
      - calls the LLM and parses the JSON response
    """

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    async def extract(
        self,
        chunks: List[str],
        query: str,
        schema_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run extraction against the provided text chunks.

        Args:
            chunks: Relevant text passages retrieved from the vector store.
            query: Natural-language extraction instruction.
            schema_hint: Optional dict describing desired output shape.

        Returns:
            Parsed JSON dict with the extracted fields.
        """
        context = "\n---\n".join(chunks)

        user_message = f"### Context\n{context}\n\n### Query\n{query}"
        if schema_hint:
            user_message += (
                f"\n\n### Expected output schema\n{json.dumps(schema_hint, indent=2)}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "langextract.extract.start",
            chunk_count=len(chunks),
            query=query[:80],
        )

        raw = await self._llm.chat_completion(messages)

        # Attempt to parse JSON from the response
        result = self._parse_json(raw)

        logger.info("langextract.extract.done", keys=list(result.keys()))
        return result

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Best-effort JSON extraction from LLM output.
        Handles cases where the model wraps JSON in markdown fences.
        """
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("langextract.parse_json.failed", raw_preview=cleaned[:200])
            return {"_raw": cleaned}
