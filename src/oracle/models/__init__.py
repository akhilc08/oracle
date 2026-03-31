"""Oracle data models for retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RetrievalResult:
    """A single result from any retrieval strategy."""

    chunk_id: str
    text: str
    score: float
    source: str  # "vector", "bm25", "graph", "recency"
    metadata: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.chunk_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return self.chunk_id == other.chunk_id


@dataclass
class FusedResult:
    """A result after RRF fusion across multiple strategies."""

    chunk_id: str
    text: str
    rrf_score: float
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)
    strategy_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    expanded_context: ExpandedContext | None = None


@dataclass
class ExpandedContext:
    """Expanded context for a retrieval result."""

    surrounding_chunks: list[str] = field(default_factory=list)
    graph_properties: dict = field(default_factory=dict)
    graph_neighbors: list[dict] = field(default_factory=list)


@dataclass
class RetrievalQuery:
    """Structured query for the retrieval engine."""

    text: str
    collection: str = "news_articles"
    top_k: int = 20
    final_k: int = 10
    entity_ids: list[str] | None = None
    market_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_authority_score: float | None = None
    enable_vector: bool = True
    enable_bm25: bool = True
    enable_graph: bool = True
    enable_recency: bool = True
    recency_decay_days: float = 7.0
