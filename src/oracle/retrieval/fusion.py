"""Reciprocal Rank Fusion (RRF) to merge results from multiple retrieval strategies."""

from __future__ import annotations

from collections import defaultdict

from oracle.models import FusedResult, RetrievalResult


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievalResult]],
    k: int = 60,
    top_k: int = 20,
) -> list[FusedResult]:
    """Merge ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over all lists L of: 1 / (k + rank_L(d))

    where k=60 is a constant that prevents high-ranked documents from
    dominating. RRF is robust to score scale differences across strategies.

    Args:
        result_lists: Dict mapping strategy name to ranked results.
        k: RRF constant (default 60, per original paper).
        top_k: Number of results to return after fusion.

    Returns:
        Fused and sorted list of FusedResult objects.
    """
    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    strategy_scores: dict[str, dict[str, float]] = defaultdict(dict)
    chunk_data: dict[str, RetrievalResult] = {}
    chunk_sources: dict[str, list[str]] = defaultdict(list)

    for strategy_name, results in result_lists.items():
        for rank, result in enumerate(results):
            rrf_scores[result.chunk_id] += 1.0 / (k + rank + 1)
            strategy_scores[result.chunk_id][strategy_name] = result.score
            chunk_sources[result.chunk_id].append(strategy_name)

            # Keep the result with the highest original score as canonical
            if (
                result.chunk_id not in chunk_data
                or result.score > chunk_data[result.chunk_id].score
            ):
                chunk_data[result.chunk_id] = result

    # Sort by RRF score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results = []
    for chunk_id, rrf_score in ranked:
        result = chunk_data[chunk_id]
        fused_results.append(
            FusedResult(
                chunk_id=chunk_id,
                text=result.text,
                rrf_score=rrf_score,
                sources=chunk_sources[chunk_id],
                strategy_scores=dict(strategy_scores[chunk_id]),
                metadata=result.metadata,
            )
        )

    return fused_results
