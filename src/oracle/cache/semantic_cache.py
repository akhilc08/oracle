"""Semantic cache backed by Qdrant — caches LLM responses by query similarity."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from oracle.config import settings
from oracle.knowledge.embeddings import EmbeddingService

logger = structlog.get_logger()

COLLECTION_NAME = "semantic_cache"
VECTOR_DIM = 1024


@dataclass
class CachedResult:
    """A cached query result."""

    query: str
    result: str
    entity_ids: list[str] = field(default_factory=list)
    market_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    score: float = 0.0


@dataclass
class CacheStats:
    """In-memory cache statistics."""

    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "total_queries": self.total_queries,
        }


class SemanticCache:
    """Qdrant-backed semantic cache for LLM query results.

    Stores query embeddings alongside results. On lookup, finds the most similar
    past query and returns the cached result if similarity exceeds the threshold.
    """

    def __init__(
        self,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        self._host = qdrant_host or settings.qdrant_host
        self._port = qdrant_port or settings.qdrant_port
        self._ttl_seconds = ttl_seconds
        self._client: QdrantClient | None = None
        self._embedder: EmbeddingService | None = None
        self.stats = CacheStats()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    def _get_embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = EmbeddingService.get_instance()
        return self._embedder

    def setup(self) -> None:
        """Ensure the semantic_cache collection exists in Qdrant."""
        client = self._get_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("semantic_cache.collection_created")

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string using the shared EmbeddingService."""
        return self._get_embedder().embed_query(text)

    def lookup(self, query: str, threshold: float = 0.95) -> CachedResult | None:
        """Search for a similar cached query.

        Args:
            query: The query to look up.
            threshold: Minimum cosine similarity to consider a cache hit.

        Returns:
            CachedResult if a sufficiently similar query is found, else None.
        """
        client = self._get_client()
        vector = self.embed_query(query)
        cutoff = time.time() - self._ttl_seconds

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=1,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=cutoff),
                    ),
                ]
            ),
            score_threshold=threshold,
        )

        if results:
            point = results[0]
            self.stats.hits += 1
            logger.info("semantic_cache.hit", score=round(point.score, 4))
            payload = point.payload or {}
            return CachedResult(
                query=payload.get("query", ""),
                result=payload.get("result", ""),
                entity_ids=payload.get("entity_ids", []),
                market_ids=payload.get("market_ids", []),
                timestamp=payload.get("timestamp", 0.0),
                score=point.score,
            )

        self.stats.misses += 1
        return None

    def store(
        self,
        query: str,
        result: str,
        entity_ids: list[str] | None = None,
        market_ids: list[str] | None = None,
    ) -> str:
        """Store a query result in the cache.

        Args:
            query: The original query.
            result: The LLM response to cache.
            entity_ids: Associated entity IDs for invalidation.
            market_ids: Associated market IDs for invalidation.

        Returns:
            The point ID of the stored entry.
        """
        client = self._get_client()
        vector = self.embed_query(query)
        point_id = uuid.uuid4().hex

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "query": query,
                        "result": result,
                        "entity_ids": entity_ids or [],
                        "market_ids": market_ids or [],
                        "timestamp": time.time(),
                    },
                )
            ],
        )

        logger.info(
            "semantic_cache.stored",
            entity_ids=entity_ids or [],
            market_ids=market_ids or [],
        )
        return point_id

    def invalidate_by_entity(self, entity_id: str) -> None:
        """Delete all cache entries related to a specific entity."""
        client = self._get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="entity_ids",
                        match=MatchValue(value=entity_id),
                    )
                ]
            ),
        )
        logger.info("semantic_cache.invalidated_entity", entity_id=entity_id)

    def invalidate_by_market(self, market_id: str) -> None:
        """Delete all cache entries related to a specific market."""
        client = self._get_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="market_ids",
                        match=MatchValue(value=market_id),
                    )
                ]
            ),
        )
        logger.info("semantic_cache.invalidated_market", market_id=market_id)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics including total entries in Qdrant."""
        client = self._get_client()
        try:
            info = client.get_collection(COLLECTION_NAME)
            total_entries = info.points_count or 0
        except Exception:
            total_entries = 0

        stats = self.stats.to_dict()
        stats["total_entries"] = total_entries
        return stats
