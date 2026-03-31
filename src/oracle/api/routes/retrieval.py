"""Retrieval API endpoints for hybrid search."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from oracle.models import RetrievalQuery
from oracle.retrieval.engine import HybridRetrievalEngine

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    """Request body for the search endpoint."""

    query: str
    collection: str = "news_articles"
    top_k: int = 10
    entity_ids: list[str] | None = None
    market_ids: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    min_authority_score: float | None = None
    enable_vector: bool = True
    enable_bm25: bool = True
    enable_graph: bool = True
    enable_recency: bool = True
    recency_decay_days: float = 7.0


@router.post("/search")
async def search(request: Request, body: SearchRequest):
    """Run hybrid retrieval search.

    Executes vector, BM25, and graph strategies, fuses with RRF,
    re-ranks with cross-encoder, and expands context.
    """
    engine = HybridRetrievalEngine(
        neo4j=request.app.state.neo4j,
        qdrant=request.app.state.qdrant,
    )

    # Parse dates if provided
    date_from = None
    date_to = None
    if body.date_from:
        try:
            date_from = datetime.fromisoformat(body.date_from)
        except ValueError:
            pass
    if body.date_to:
        try:
            date_to = datetime.fromisoformat(body.date_to)
        except ValueError:
            pass

    query = RetrievalQuery(
        text=body.query,
        collection=body.collection,
        top_k=20,
        final_k=body.top_k,
        entity_ids=body.entity_ids,
        market_ids=body.market_ids,
        date_from=date_from,
        date_to=date_to,
        min_authority_score=body.min_authority_score,
        enable_vector=body.enable_vector,
        enable_bm25=body.enable_bm25,
        enable_graph=body.enable_graph,
        enable_recency=body.enable_recency,
        recency_decay_days=body.recency_decay_days,
    )

    results, metrics = await engine.retrieve(query)

    return {
        "query": body.query,
        "count": len(results),
        "metrics": {
            "total_time_ms": round(metrics.total_time_ms, 1),
            "vector_time_ms": round(metrics.vector_time_ms, 1),
            "bm25_time_ms": round(metrics.bm25_time_ms, 1),
            "graph_time_ms": round(metrics.graph_time_ms, 1),
            "fusion_time_ms": round(metrics.fusion_time_ms, 1),
            "rerank_time_ms": round(metrics.rerank_time_ms, 1),
            "expansion_time_ms": round(metrics.expansion_time_ms, 1),
            "strategies_used": metrics.strategies_used,
            "vector_count": metrics.vector_count,
            "bm25_count": metrics.bm25_count,
            "graph_count": metrics.graph_count,
            "fused_count": metrics.fused_count,
        },
        "results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:500],  # Truncate for API response
                "rrf_score": round(r.rrf_score, 6),
                "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                "sources": r.sources,
                "strategy_scores": {
                    k: round(v, 4) for k, v in r.strategy_scores.items()
                },
                "metadata": {
                    k: v
                    for k, v in r.metadata.items()
                    if k in ("source_url", "publication_date", "author",
                             "source_name", "article_title", "entity_ids",
                             "market_ids", "source_authority_score")
                },
                "expanded_context": {
                    "surrounding_chunks": [
                        c[:200] for c in (r.expanded_context.surrounding_chunks if r.expanded_context else [])
                    ],
                    "graph_neighbors_count": len(
                        r.expanded_context.graph_neighbors if r.expanded_context else []
                    ),
                }
                if r.expanded_context
                else None,
            }
            for r in results
        ],
    }


@router.get("/search")
async def search_get(
    request: Request,
    q: str = Query(..., description="Search query text"),
    collection: str = Query("news_articles", description="Qdrant collection to search"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
):
    """Simple GET endpoint for quick searches."""
    body = SearchRequest(query=q, collection=collection, top_k=top_k)
    return await search(request, body)
