"""Tests for Phase 2 retrieval components."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from oracle.models import FusedResult, RetrievalQuery, RetrievalResult
from oracle.retrieval.bm25_search import BM25Index, BM25SearchStrategy
from oracle.retrieval.fusion import reciprocal_rank_fusion
from oracle.retrieval.recency import apply_recency_weight


# --- BM25 Index Tests ---


class TestBM25Index:
    def test_tokenize(self):
        tokens = BM25Index.tokenize("The Federal Reserve raised interest rates!")
        assert tokens == ["the", "federal", "reserve", "raised", "interest", "rates"]

    def test_add_and_search(self):
        idx = BM25Index()
        idx.add_documents(
            doc_ids=["d1", "d2", "d3"],
            texts=[
                "Federal Reserve interest rate decision monetary policy",
                "Bitcoin cryptocurrency price prediction market",
                "Federal Reserve inflation target rate hike",
            ],
        )
        results = idx.search("Federal Reserve interest rate", top_k=3)
        assert len(results) >= 2
        # d1 and d3 should rank above d2 (more relevant)
        ids = [r[0] for r in results]
        assert "d1" in ids[:2]
        assert "d3" in ids[:2]

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_empty_query(self):
        idx = BM25Index()
        idx.add_documents(["d1"], ["some text"])
        results = idx.search("")
        assert results == []

    def test_no_matching_terms(self):
        idx = BM25Index()
        idx.add_documents(["d1"], ["apple banana cherry"])
        results = idx.search("quantum physics")
        assert results == []

    def test_allowed_ids_filter(self):
        idx = BM25Index()
        idx.add_documents(
            doc_ids=["d1", "d2", "d3"],
            texts=[
                "Federal Reserve policy",
                "Federal Reserve rates",
                "Federal Reserve inflation",
            ],
        )
        results = idx.search("Federal Reserve", top_k=10, allowed_ids={"d1", "d3"})
        ids = [r[0] for r in results]
        assert "d2" not in ids

    def test_metadata_passthrough(self):
        idx = BM25Index()
        idx.add_documents(
            doc_ids=["d1"],
            texts=["test document"],
            metadata=[{"source": "reuters"}],
        )
        results = idx.search("test document")
        assert results[0][3] == {"source": "reuters"}


# --- Recency Weighting Tests ---


class TestRecencyWeighting:
    def test_recent_gets_higher_score(self):
        now = datetime.now(timezone.utc)
        results = [
            RetrievalResult(
                chunk_id="new",
                text="new article",
                score=1.0,
                source="vector",
                metadata={"publication_date": (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ),
            RetrievalResult(
                chunk_id="old",
                text="old article",
                score=1.0,
                source="vector",
                metadata={"publication_date": (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ),
        ]
        weighted = apply_recency_weight(results, decay_days=7.0, reference_time=now)
        assert weighted[0].score > weighted[1].score

    def test_decay_math(self):
        now = datetime.now(timezone.utc)
        results = [
            RetrievalResult(
                chunk_id="7d",
                text="week old",
                score=1.0,
                source="vector",
                metadata={"publication_date": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ),
        ]
        weighted = apply_recency_weight(results, decay_days=7.0, reference_time=now)
        expected = math.exp(-1.0)  # -7/7 = -1
        assert abs(weighted[0].score - expected) < 0.01

    def test_missing_date_gets_penalty(self):
        results = [
            RetrievalResult(
                chunk_id="no_date",
                text="no date",
                score=1.0,
                source="vector",
                metadata={},
            ),
        ]
        weighted = apply_recency_weight(results, decay_days=7.0)
        assert weighted[0].score == pytest.approx(0.15, abs=0.001)

    def test_future_date_gets_full_weight(self):
        now = datetime.now(timezone.utc)
        results = [
            RetrievalResult(
                chunk_id="future",
                text="future",
                score=1.0,
                source="vector",
                metadata={"publication_date": (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ),
        ]
        weighted = apply_recency_weight(results, decay_days=7.0, reference_time=now)
        assert weighted[0].score == pytest.approx(1.0, abs=0.001)


# --- RRF Fusion Tests ---


class TestRRFFusion:
    def test_single_strategy(self):
        results = {
            "vector": [
                RetrievalResult("a", "text a", 0.9, "vector"),
                RetrievalResult("b", "text b", 0.7, "vector"),
            ]
        }
        fused = reciprocal_rank_fusion(results, k=60, top_k=10)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_multi_strategy_boost(self):
        """Results appearing in multiple strategies should rank higher."""
        results = {
            "vector": [
                RetrievalResult("shared", "shared text", 0.9, "vector"),
                RetrievalResult("vector_only", "v text", 0.8, "vector"),
            ],
            "bm25": [
                RetrievalResult("shared", "shared text", 5.0, "bm25"),
                RetrievalResult("bm25_only", "b text", 4.0, "bm25"),
            ],
        }
        fused = reciprocal_rank_fusion(results, k=60, top_k=10)
        assert fused[0].chunk_id == "shared"
        assert "vector" in fused[0].sources
        assert "bm25" in fused[0].sources

    def test_k_parameter(self):
        """Higher k should reduce score differences between ranks."""
        results = {
            "vector": [
                RetrievalResult("a", "a", 1.0, "vector"),
                RetrievalResult("b", "b", 0.5, "vector"),
            ]
        }
        fused_k10 = reciprocal_rank_fusion(results, k=10)
        fused_k100 = reciprocal_rank_fusion(results, k=100)

        # Score ratio should be closer to 1.0 with higher k
        ratio_k10 = fused_k10[0].rrf_score / fused_k10[1].rrf_score
        ratio_k100 = fused_k100[0].rrf_score / fused_k100[1].rrf_score
        assert ratio_k100 < ratio_k10

    def test_empty_input(self):
        fused = reciprocal_rank_fusion({}, k=60)
        assert fused == []

    def test_strategy_scores_tracked(self):
        results = {
            "vector": [RetrievalResult("a", "text", 0.95, "vector")],
            "graph": [RetrievalResult("a", "text", 0.8, "graph")],
        }
        fused = reciprocal_rank_fusion(results, k=60)
        assert "vector" in fused[0].strategy_scores
        assert "graph" in fused[0].strategy_scores
        assert fused[0].strategy_scores["vector"] == pytest.approx(0.95)


# --- BM25 Search Strategy Tests ---


class TestBM25Strategy:
    @pytest.mark.asyncio
    async def test_search_without_index(self):
        strategy = BM25SearchStrategy()
        query = RetrievalQuery(text="test query")
        results = await strategy.search(query)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_after_build(self):
        strategy = BM25SearchStrategy()
        strategy.build_index(
            doc_ids=["d1", "d2"],
            texts=["Federal Reserve monetary policy", "Bitcoin price prediction"],
        )
        query = RetrievalQuery(text="Federal Reserve", top_k=5)
        results = await strategy.search(query)
        assert len(results) >= 1
        assert results[0].source == "bm25"
        assert results[0].chunk_id == "d1"


# --- Model Tests ---


class TestModels:
    def test_retrieval_result_hash(self):
        r1 = RetrievalResult("id1", "text", 0.5, "vector")
        r2 = RetrievalResult("id1", "different text", 0.9, "bm25")
        assert hash(r1) == hash(r2)
        assert r1 == r2

    def test_retrieval_result_set(self):
        r1 = RetrievalResult("id1", "text", 0.5, "vector")
        r2 = RetrievalResult("id1", "text", 0.9, "bm25")
        r3 = RetrievalResult("id2", "text", 0.5, "vector")
        assert len({r1, r2, r3}) == 2

    def test_retrieval_query_defaults(self):
        q = RetrievalQuery(text="test")
        assert q.collection == "news_articles"
        assert q.top_k == 20
        assert q.final_k == 10
        assert q.enable_vector is True
        assert q.enable_bm25 is True
        assert q.enable_graph is True
        assert q.enable_recency is True
