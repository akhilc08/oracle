"""Tests for semantic chunking."""

from oracle.ingestion.chunker import (
    Chunk,
    estimate_tokens,
    hierarchical_chunk,
    semantic_chunk,
    split_into_paragraphs,
)


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 2  # 11 chars / 4
    assert estimate_tokens("") == 0


def test_split_into_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = split_into_paragraphs(text)
    assert len(paragraphs) == 3
    assert paragraphs[0] == "First paragraph."


def test_split_into_paragraphs_filters_empty():
    text = "First.\n\n\n\nSecond."
    paragraphs = split_into_paragraphs(text)
    assert len(paragraphs) == 2


def test_semantic_chunk_single_paragraph():
    """Single paragraph should return one chunk."""

    def mock_embed(texts):
        return [[0.1] * 1024 for _ in texts]

    text = "This is a single paragraph with enough content to pass."
    chunks = semantic_chunk(text, embed_fn=mock_embed)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_semantic_chunk_multiple_paragraphs():
    """Multiple similar paragraphs should be merged."""

    def mock_embed(texts):
        # Return same embedding for all — high similarity
        return [[1.0] * 1024 for _ in texts]

    text = "First paragraph about politics.\n\nSecond paragraph about politics.\n\nThird paragraph about politics."
    chunks = semantic_chunk(text, embed_fn=mock_embed, min_tokens=5, max_tokens=500)
    # All similar, should be merged into one chunk
    assert len(chunks) == 1


def test_semantic_chunk_topic_shift():
    """Dissimilar paragraphs should be split."""

    def mock_embed(texts):
        import numpy as np
        # Return different embeddings for each paragraph
        embeddings = []
        for i, _ in enumerate(texts):
            vec = [0.0] * 1024
            vec[i % 1024] = 1.0
            embeddings.append(vec)
        return embeddings

    para = "x " * 200  # ~200 chars = ~50 tokens each
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = semantic_chunk(text, embed_fn=mock_embed, min_tokens=10, max_tokens=100)
    # Should split due to dissimilarity
    assert len(chunks) >= 2


def test_hierarchical_chunk():
    text = """# Section One
This is content in section one. It has enough text to be meaningful.
More content here about important topics.

# Section Two
This is content in section two. Different topic entirely.
More analysis and details here.

# Section Three
Final section with conclusions and recommendations.
"""
    chunks = hierarchical_chunk(text, min_tokens=10, max_tokens=500)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
