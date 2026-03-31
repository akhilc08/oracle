"""Semantic chunking for news articles and documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, filtering empty ones."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def semantic_chunk(
    text: str,
    embed_fn: callable,
    min_tokens: int = 200,
    max_tokens: int = 500,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """Chunk text by detecting semantic topic shifts.

    1. Split into paragraphs
    2. Embed each paragraph
    3. Merge adjacent paragraphs if cosine similarity > threshold
    4. Enforce min/max token limits
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    if len(paragraphs) == 1:
        return [Chunk(text=paragraphs[0], index=0, token_count=estimate_tokens(paragraphs[0]))]

    # Embed all paragraphs
    embeddings = embed_fn(paragraphs)

    # Calculate cosine similarities between adjacent paragraphs
    similarities = []
    for i in range(len(embeddings) - 1):
        a = np.array(embeddings[i])
        b = np.array(embeddings[i + 1])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarities.append(float(sim))

    # Group paragraphs into chunks based on similarity drops
    chunks: list[Chunk] = []
    current_paragraphs: list[str] = [paragraphs[0]]

    for i, sim in enumerate(similarities):
        current_text = "\n\n".join(current_paragraphs)
        next_text = "\n\n".join(current_paragraphs + [paragraphs[i + 1]])

        if sim < similarity_threshold and estimate_tokens(current_text) >= min_tokens:
            # Topic shift detected and current chunk is big enough
            chunks.append(Chunk(
                text=current_text,
                index=len(chunks),
                token_count=estimate_tokens(current_text),
            ))
            current_paragraphs = [paragraphs[i + 1]]
        elif estimate_tokens(next_text) > max_tokens:
            # Would exceed max tokens — split here
            chunks.append(Chunk(
                text=current_text,
                index=len(chunks),
                token_count=estimate_tokens(current_text),
            ))
            current_paragraphs = [paragraphs[i + 1]]
        else:
            current_paragraphs.append(paragraphs[i + 1])

    # Don't forget the last chunk
    if current_paragraphs:
        final_text = "\n\n".join(current_paragraphs)
        chunks.append(Chunk(
            text=final_text,
            index=len(chunks),
            token_count=estimate_tokens(final_text),
        ))

    return chunks


def hierarchical_chunk(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
) -> list[Chunk]:
    """Hierarchical chunking for structured documents (legal, SEC filings).

    Preserves section headers and splits on section boundaries.
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_header: str = ""

    for line in lines:
        stripped = line.strip()

        # Detect section headers (lines that are ALL CAPS or start with #/Section/Article)
        is_header = (
            (stripped.isupper() and len(stripped) > 3)
            or stripped.startswith("#")
            or stripped.lower().startswith(("section", "article", "chapter", "part"))
        )

        if is_header and current_lines:
            current_text = "\n".join(current_lines)
            tokens = estimate_tokens(current_text)

            if tokens >= min_tokens:
                chunks.append(Chunk(
                    text=current_text,
                    index=len(chunks),
                    token_count=tokens,
                    metadata={"section_header": current_header},
                ))
                current_lines = [line]
                current_header = stripped
            elif tokens > max_tokens:
                # Split oversized chunk
                chunks.append(Chunk(
                    text=current_text,
                    index=len(chunks),
                    token_count=tokens,
                    metadata={"section_header": current_header},
                ))
                current_lines = [line]
                current_header = stripped
            else:
                current_lines.append(line)
        else:
            if is_header:
                current_header = stripped
            current_lines.append(line)

    # Final chunk
    if current_lines:
        final_text = "\n".join(current_lines)
        chunks.append(Chunk(
            text=final_text,
            index=len(chunks),
            token_count=estimate_tokens(final_text),
            metadata={"section_header": current_header},
        ))

    return chunks
