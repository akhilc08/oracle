"""Vector similarity search over Qdrant with payload filtering."""

from __future__ import annotations

from datetime import datetime

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    Range,
)

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
        qdrant_filter = self._build_filter(query)

        hits = await self.qdrant.client.search(
            collection_name=query.collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=query.top_k,
        )

        return [
            RetrievalResult(
                chunk_id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=hit.score,
                source="vector",
                metadata=hit.payload,
            )
            for hit in hits
        ]

    @staticmethod
    def _build_filter(query: RetrievalQuery) -> Filter | None:
        """Build Qdrant Filter object from query parameters."""
        conditions: list[FieldCondition] = []

        if query.entity_ids:
            conditions.append(
                FieldCondition(key="entity_ids", match=MatchAny(any=query.entity_ids))
            )

        if query.market_ids:
            conditions.append(
                FieldCondition(key="market_ids", match=MatchAny(any=query.market_ids))
            )

        if query.date_from:
            conditions.append(
                FieldCondition(
                    key="publication_date",
                    range=DatetimeRange(gte=query.date_from),
                )
            )

        if query.date_to:
            conditions.append(
                FieldCondition(
                    key="publication_date",
                    range=DatetimeRange(lte=query.date_to),
                )
            )

        if query.min_authority_score is not None:
            conditions.append(
                FieldCondition(
                    key="source_authority_score",
                    range=Range(gte=query.min_authority_score),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)
