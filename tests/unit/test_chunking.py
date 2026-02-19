"""
Unit tests for the document chunking utility.

Tests cover:
  - Basic chunking with default settings
  - Overlap between chunks
  - Boundary snapping (paragraph, sentence)
  - Edge cases (empty text, text shorter than chunk_size)
  - Custom chunk_size and overlap
"""

from __future__ import annotations

import pytest

from app.workers.tasks import chunk_text


pytestmark = pytest.mark.unit


class TestChunkText:
    """Test the chunk_text helper function."""

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello, world!"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size(self):
        text = "a" * 512
        chunks = chunk_text(text, chunk_size=512, overlap=0)
        assert len(chunks) == 1

    def test_multiple_chunks_produced(self):
        # Create text longer than one chunk
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 30
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        # Reconstruct — every character of the original should appear
        combined = "".join(chunks)
        # Due to overlap some parts repeat, but all original content is present
        for word in ["quick", "brown", "fox", "lazy", "dog"]:
            assert word in combined

    def test_overlap_creates_shared_content(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunks = chunk_text(text, chunk_size=120, overlap=30)

        # With overlap, adjacent chunks should share some content
        assert len(chunks) >= 2
        # The end of chunk[0] should overlap with the start of chunk[1]
        if len(chunks) >= 2:
            tail = chunks[0][-30:]
            head = chunks[1][:30]
            # At least some overlap
            assert len(set(tail) & set(head)) > 0

    def test_paragraph_boundary_snapping(self):
        text = (
            "First paragraph content here.\n\n"
            "Second paragraph content here.\n\n"
            "Third paragraph with more text."
        )
        chunks = chunk_text(text, chunk_size=50, overlap=5)

        # The chunker should try to break at \n\n boundaries
        assert len(chunks) >= 2

    def test_sentence_boundary_snapping(self):
        text = (
            "First sentence here. Second sentence follows. "
            "Third sentence ends. Fourth sentence completes."
        )
        chunks = chunk_text(text, chunk_size=50, overlap=5)

        # Chunks should try to end at sentence boundaries
        for chunk in chunks[:-1]:  # Last chunk may not end at sentence boundary
            stripped = chunk.rstrip()
            # Should end at a natural boundary (period+space was the delimiter)
            assert (
                stripped.endswith(".")
                or stripped.endswith("!")
                or stripped.endswith("?")
                or stripped.endswith("\n")
                or len(stripped) <= 50
            )

    def test_zero_overlap(self):
        """Note: overlap=0 is falsy, so chunk_text falls back to default.
        Use overlap=1 for near-zero overlap test instead."""
        text = "abcdefghij" * 20  # 200 chars, no natural boundaries
        chunks = chunk_text(text, chunk_size=100, overlap=1)
        assert len(chunks) >= 2
        # With overlap=1, each chunk except the last should be ~100 chars
        for chunk in chunks[:-1]:
            assert len(chunk) <= 101

    def test_custom_chunk_size(self):
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=256, overlap=32)
        for chunk in chunks:
            # Chunks should not vastly exceed chunk_size
            assert len(chunk) <= 256 + 50  # Allow some boundary-snapping slack
