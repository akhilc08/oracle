"""Vector similarity search over Qdrant with payload filtering."""

from __future__ import annotations

from datetime import datetime

from oracle.knowledge.embeddings import EmbeddingService
from oracle.knowledge.qdrant_client import QdrantManager
from oracle.models import RetrievalQuery, RetrievalResult


class VectorSearchStrategy:
    """Semantic search using BGE-large embeddings + Qdrant HNSW."""

    def __init__(self, qdrant: QdrantManager, embedder: EmbeddingService) -> None:
        self.qdrant = qdrant
        self.embedder = embedder

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Run vector similarity search with optional payload filters."""
        query_vector = self.embedder.embed_query(query.text)
        filters = self._build_filters(query)

        raw_results = await self.qdrant.search(
            collection=query.collection,
            query_vector=query_vector,
            limit=query.top_k,
            filters=filters,
        )

        return [
            RetrievalResult(
                chunk_id=str(r["id"]),
                text=r["payload"].get("text", ""),
                score=r["score"],
                source="vector",
                metadata=r["payload"],
            )
            for r in raw_results
        ]

    def _build_filters(self, query: RetrievalQuery) -> dict | None:
        """Build Qdrant payload filter from query parameters."""
        conditions = []

        if query.entity_ids:
            conditions.append(
                {"key": "entity_ids", "match": {"any": query.entity_ids}}
            )

        if query.market_ids:
            conditions.append(
                {"key": "market_ids", "match": {"any": query.market_ids}}
            )

        if query.date_from:
            conditions.append(
                {
                    "key": "publication_date",
                    "range": {"gte": query.date_from.isoformat()},
                }
            )

        if query.date_to:
            conditions.append(
                {
                    "key": "publication_date",
                    "range": {"lte": query.date_to.isoformat()},
                }
            )

        if query.min_authority_score is not None:
            conditions.append(
                {
                    "key": "source_authority_score",
                    "range": {"gte": query.min_authority_score},
                }
            )

        if not conditions:
            return None

        return {"must": conditions}
