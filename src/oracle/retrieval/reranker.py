"""Cross-encoder re-ranker using BGE-reranker-v2-m3."""

from __future__ import annotations

import logging

from oracle.models import FusedResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank fused results using BGE-reranker-v2-m3 cross-encoder.

    The cross-encoder scores (query, passage) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity alone.
    """

    _instance: CrossEncoderReranker | None = None
    _model = None

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name

    @classmethod
    def get_instance(
        cls, model_name: str = "BAAI/bge-reranker-v2-m3"
    ) -> CrossEncoderReranker:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if CrossEncoderReranker._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder model: %s", self.model_name)
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                logger.info("Cross-encoder model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers CrossEncoder not available, "
                    "falling back to no re-ranking"
                )
                return None
            except Exception as e:
                logger.warning("Failed to load cross-encoder: %s", e)
                return None
        return CrossEncoderReranker._model

    def rerank(
        self,
        query: str,
        results: list[FusedResult],
        top_k: int = 10,
    ) -> list[FusedResult]:
        """Re-rank fused results using the cross-encoder.

        Args:
            query: The original query text.
            results: Fused results to re-rank.
            top_k: Number of results to return after re-ranking.

        Returns:
            Re-ranked list of FusedResult objects with rerank_score set.
        """
        if not results:
            return []

        model = self._load_model()
        if model is None:
            # Fallback: just return top_k by RRF score
            logger.warning("Cross-encoder unavailable, returning RRF-ranked results")
            return results[:top_k]

        # Prepare (query, passage) pairs
        pairs = [(query, result.text) for result in results]

        # Score all pairs
        scores = model.predict(pairs)

        # Assign rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score descending
        results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        return results[:top_k]
