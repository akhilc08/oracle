"""Qdrant vector database manager with collection setup."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    VectorParams,
)

logger = structlog.get_logger()

# Collection configurations per PRD
COLLECTIONS = {
    "news_articles": {
        "description": "News, blog posts, press releases",
        "chunking": "Semantic (200-500 tokens)",
    },
    "official_documents": {
        "description": "Court filings, SEC, legislation",
        "chunking": "Hierarchical (300-800 tokens)",
    },
    "social_media": {
        "description": "Tweets, Reddit, forums",
        "chunking": "Per-post; short posts batched",
    },
    "transcripts": {
        "description": "Earnings calls, press conferences, hearings",
        "chunking": "Speaker-aware chunking",
    },
}

# Shared payload indexes for all collections
PAYLOAD_INDEXES = {
    "source_url": PayloadSchemaType.KEYWORD,
    "publication_date": PayloadSchemaType.DATETIME,
    "author": PayloadSchemaType.KEYWORD,
    "source_authority_score": PayloadSchemaType.FLOAT,
    "entity_ids": PayloadSchemaType.KEYWORD,
    "market_ids": PayloadSchemaType.KEYWORD,
}

EMBEDDING_DIM = 1024  # BGE-large-en-v1.5


class QdrantManager:
    """Async Qdrant manager for Oracle vector store."""

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = AsyncQdrantClient(host=host, port=port)
        self._host = host
        self._port = port

    async def setup_collections(self) -> None:
        """Create all 4 collections with HNSW config and payload indexes."""
        existing = {c.name for c in (await self._client.get_collections()).collections}

        for name in COLLECTIONS:
            if name in existing:
                logger.info("qdrant.collection_exists", collection=name)
                continue

            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                    ),
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )
            logger.info("qdrant.collection_created", collection=name)

            # Create payload indexes for filtering
            for field_name, field_type in PAYLOAD_INDEXES.items():
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            logger.info("qdrant.indexes_created", collection=name, count=len(PAYLOAD_INDEXES))

        logger.info("qdrant.setup_complete", collections=len(COLLECTIONS))

    async def verify_connectivity(self) -> bool:
        """Check Qdrant is reachable."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        for name in COLLECTIONS:
            try:
                info = await self._client.get_collection(name)
                stats[name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.value,
                }
            except Exception:
                stats[name] = {"status": "not_found"}
        return stats

    async def upsert_chunks(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert embedded chunks into a collection."""
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=uid, vector=vec, payload=payload)
            for uid, vec, payload in zip(ids, vectors, payloads)
        ]

        await self._client.upsert(
            collection_name=collection,
            points=points,
        )
        logger.info("qdrant.upserted", collection=collection, count=len(points))

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search a collection by vector similarity."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        results = await self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]
