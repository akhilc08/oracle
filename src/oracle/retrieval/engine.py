"""Hybrid retrieval engine — the main orchestrator for all retrieval strategies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.embeddings import EmbeddingService
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager
from oracle.models import FusedResult, RetrievalQuery, RetrievalResult
from oracle.retrieval.bm25_search import BM25SearchStrategy
from oracle.retrieval.expansion import ContextualExpander
from oracle.retrieval.fusion import reciprocal_rank_fusion
from oracle.retrieval.graph_search import GraphSearchStrategy
from oracle.retrieval.recency import apply_recency_weight
from oracle.retrieval.reranker import CrossEncoderReranker
from oracle.retrieval.vector_search import VectorSearchStrategy

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Timing and count metrics for a single retrieval call."""

    vector_time_ms: float = 0.0
    bm25_time_ms: float = 0.0
    graph_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    expansion_time_ms: float = 0.0
    total_time_ms: float = 0.0
    vector_count: int = 0
    bm25_count: int = 0
    graph_count: int = 0
    fused_count: int = 0
    final_count: int = 0
    strategies_used: list[str] = field(default_factory=list)


class HybridRetrievalEngine:
    """Orchestrates multi-strategy retrieval with RRF fusion and re-ranking.

    Pipeline:
    1. Run enabled strategies in parallel (vector, BM25, graph)
    2. Apply recency weighting to vector results
    3. Merge via Reciprocal Rank Fusion (k=60)
    4. Re-rank top results with BGE-reranker-v2-m3
    5. Expand top results with surrounding chunks + graph context
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        qdrant: QdrantManager,
        embedder: EmbeddingService | None = None,
        entity_resolver: EntityResolver | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.embedder = embedder or EmbeddingService.get_instance()
        self.entity_resolver = entity_resolver or EntityResolver()

        # Initialize strategies
        self.vector_strategy = VectorSearchStrategy(self.qdrant, self.embedder)
        self.bm25_strategy = BM25SearchStrategy()
        self.graph_strategy = GraphSearchStrategy(self.neo4j, self.entity_resolver)
        self.expander = ContextualExpander(self.qdrant, self.neo4j)
        self.reranker = reranker or CrossEncoderReranker.get_instance()

    async def retrieve(
        self, query: RetrievalQuery
    ) -> tuple[list[FusedResult], RetrievalMetrics]:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: Structured retrieval query with filters and strategy flags.

        Returns:
            Tuple of (final results, timing metrics).
        """
        metrics = RetrievalMetrics()
        total_start = time.perf_counter()
        result_lists: dict[str, list[RetrievalResult]] = {}

        # --- Strategy 1: Vector search ---
        if query.enable_vector:
            t0 = time.perf_counter()
            try:
                vector_results = await self.vector_strategy.search(query)

                # Apply recency weighting
                if query.enable_recency and vector_results:
                    vector_results = apply_recency_weight(
                        vector_results, decay_days=query.recency_decay_days
                    )
                    # Re-sort after recency weighting
                    vector_results.sort(key=lambda r: r.score, reverse=True)

                result_lists["vector"] = vector_results
                metrics.vector_count = len(vector_results)
                metrics.strategies_used.append("vector")
                logger.info("Vector search: %d results", len(vector_results))
            except Exception as e:
                logger.error("Vector search failed: %s", e)
                result_lists["vector"] = []
            metrics.vector_time_ms = (time.perf_counter() - t0) * 1000

        # --- Strategy 2: BM25 keyword search ---
        if query.enable_bm25:
            t0 = time.perf_counter()
            try:
                bm25_results = await self.bm25_strategy.search_with_qdrant_docs(
                    query, self.qdrant
                )
                result_lists["bm25"] = bm25_results
                metrics.bm25_count = len(bm25_results)
                metrics.strategies_used.append("bm25")
                logger.info("BM25 search: %d results", len(bm25_results))
            except Exception as e:
                logger.error("BM25 search failed: %s", e)
                result_lists["bm25"] = []
            metrics.bm25_time_ms = (time.perf_counter() - t0) * 1000

        # --- Strategy 3: Graph traversal ---
        if query.enable_graph:
            t0 = time.perf_counter()
            try:
                graph_results = await self.graph_strategy.search(query)
                result_lists["graph"] = graph_results
                metrics.graph_count = len(graph_results)
                metrics.strategies_used.append("graph")
                logger.info("Graph search: %d results", len(graph_results))
            except Exception as e:
                logger.error("Graph search failed: %s", e)
                result_lists["graph"] = []
            metrics.graph_time_ms = (time.perf_counter() - t0) * 1000

        # --- Fusion ---
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            result_lists, k=60, top_k=query.top_k
        )
        metrics.fusion_time_ms = (time.perf_counter() - t0) * 1000
        metrics.fused_count = len(fused)
        logger.info("RRF fusion: %d results from %d strategies", len(fused), len(result_lists))

        # --- Re-ranking ---
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query.text, fused, top_k=query.final_k)
        metrics.rerank_time_ms = (time.perf_counter() - t0) * 1000
        logger.info("Re-ranking: %d → %d results", len(fused), len(reranked))

        # --- Contextual expansion ---
        t0 = time.perf_counter()
        expanded = await self.expander.expand(reranked, collection=query.collection)
        metrics.expansion_time_ms = (time.perf_counter() - t0) * 1000

        metrics.final_count = len(expanded)
        metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Retrieval complete: %d final results in %.1fms "
            "(vector=%.1fms, bm25=%.1fms, graph=%.1fms, "
            "fusion=%.1fms, rerank=%.1fms, expansion=%.1fms)",
            metrics.final_count,
            metrics.total_time_ms,
            metrics.vector_time_ms,
            metrics.bm25_time_ms,
            metrics.graph_time_ms,
            metrics.fusion_time_ms,
            metrics.rerank_time_ms,
            metrics.expansion_time_ms,
        )

        return expanded, metrics

    async def retrieve_simple(
        self,
        query_text: str,
        collection: str = "news_articles",
        top_k: int = 10,
    ) -> list[FusedResult]:
        """Convenience method for simple text queries."""
        query = RetrievalQuery(
            text=query_text,
            collection=collection,
            top_k=20,
            final_k=top_k,
        )
        results, _ = await self.retrieve(query)
        return results
