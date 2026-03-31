#!/usr/bin/env python3
"""Retrieval benchmarking script.

Generates 20 test queries across different categories, runs all retrieval
strategies, and logs precision and timing metrics.

Usage:
    python -m scripts.benchmark_retrieval
    # or
    python scripts/benchmark_retrieval.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oracle.config import settings
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager
from oracle.models import RetrievalQuery
from oracle.retrieval.engine import HybridRetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")

# 20 test queries spanning different prediction market domains
TEST_QUERIES = [
    # Politics
    {"query": "Will Donald Trump win the 2024 presidential election?", "category": "politics"},
    {"query": "What are the chances of a government shutdown?", "category": "politics"},
    {"query": "Senate confirmation hearing nominees", "category": "politics"},
    {"query": "Biden administration executive orders", "category": "politics"},
    # Economics
    {"query": "Federal Reserve interest rate decision", "category": "economics"},
    {"query": "Will inflation exceed 3% this year?", "category": "economics"},
    {"query": "US GDP growth forecast recession probability", "category": "economics"},
    {"query": "Tariff impact on consumer prices", "category": "economics"},
    # Crypto
    {"query": "Bitcoin price prediction above $100k", "category": "crypto"},
    {"query": "Ethereum ETF SEC approval likelihood", "category": "crypto"},
    # Geopolitics
    {"query": "Ukraine Russia ceasefire negotiations", "category": "geopolitics"},
    {"query": "China Taiwan military tensions", "category": "geopolitics"},
    {"query": "NATO expansion and defense spending", "category": "geopolitics"},
    # Legal
    {"query": "Supreme Court rulings on federal agency power", "category": "legal"},
    {"query": "Trump indictment trial verdict probability", "category": "legal"},
    # Tech
    {"query": "OpenAI GPT-5 release date prediction", "category": "tech"},
    {"query": "AI regulation legislation Congress", "category": "tech"},
    # Sports
    {"query": "Super Bowl winner odds prediction", "category": "sports"},
    # Multi-domain
    {"query": "Impact of sanctions on oil prices and inflation", "category": "multi"},
    {"query": "Election outcome effect on stock market", "category": "multi"},
]


async def run_benchmark():
    """Run the full retrieval benchmark."""
    logger.info("=" * 60)
    logger.info("Oracle Retrieval Benchmark")
    logger.info("=" * 60)

    # Initialize services
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    qdrant = QdrantManager(host=settings.qdrant_host, port=settings.qdrant_port)

    engine = HybridRetrievalEngine(neo4j=neo4j, qdrant=qdrant)

    results_log = []
    strategy_timings: dict[str, list[float]] = {
        "vector": [], "bm25": [], "graph": [],
        "fusion": [], "rerank": [], "expansion": [], "total": [],
    }
    strategy_counts: dict[str, list[int]] = {
        "vector": [], "bm25": [], "graph": [], "fused": [], "final": [],
    }

    for i, test in enumerate(TEST_QUERIES):
        logger.info("-" * 40)
        logger.info("Query %d/%d [%s]: %s", i + 1, len(TEST_QUERIES), test["category"], test["query"])

        query = RetrievalQuery(
            text=test["query"],
            collection="news_articles",
            top_k=20,
            final_k=10,
        )

        try:
            results, metrics = await engine.retrieve(query)

            # Log metrics
            entry = {
                "query_index": i,
                "query": test["query"],
                "category": test["category"],
                "final_count": metrics.final_count,
                "metrics": {
                    "vector_time_ms": round(metrics.vector_time_ms, 1),
                    "bm25_time_ms": round(metrics.bm25_time_ms, 1),
                    "graph_time_ms": round(metrics.graph_time_ms, 1),
                    "fusion_time_ms": round(metrics.fusion_time_ms, 1),
                    "rerank_time_ms": round(metrics.rerank_time_ms, 1),
                    "expansion_time_ms": round(metrics.expansion_time_ms, 1),
                    "total_time_ms": round(metrics.total_time_ms, 1),
                    "vector_count": metrics.vector_count,
                    "bm25_count": metrics.bm25_count,
                    "graph_count": metrics.graph_count,
                    "fused_count": metrics.fused_count,
                    "strategies_used": metrics.strategies_used,
                },
                "top_results": [
                    {
                        "chunk_id": r.chunk_id,
                        "rrf_score": round(r.rrf_score, 6),
                        "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                        "sources": r.sources,
                        "text_preview": r.text[:200],
                    }
                    for r in results[:5]
                ],
            }
            results_log.append(entry)

            # Accumulate timings
            strategy_timings["vector"].append(metrics.vector_time_ms)
            strategy_timings["bm25"].append(metrics.bm25_time_ms)
            strategy_timings["graph"].append(metrics.graph_time_ms)
            strategy_timings["fusion"].append(metrics.fusion_time_ms)
            strategy_timings["rerank"].append(metrics.rerank_time_ms)
            strategy_timings["expansion"].append(metrics.expansion_time_ms)
            strategy_timings["total"].append(metrics.total_time_ms)
            strategy_counts["vector"].append(metrics.vector_count)
            strategy_counts["bm25"].append(metrics.bm25_count)
            strategy_counts["graph"].append(metrics.graph_count)
            strategy_counts["fused"].append(metrics.fused_count)
            strategy_counts["final"].append(metrics.final_count)

            logger.info(
                "  Results: %d final (vector=%d, bm25=%d, graph=%d, fused=%d) in %.1fms",
                metrics.final_count,
                metrics.vector_count,
                metrics.bm25_count,
                metrics.graph_count,
                metrics.fused_count,
                metrics.total_time_ms,
            )

        except Exception as e:
            logger.error("  FAILED: %s", e)
            results_log.append({
                "query_index": i,
                "query": test["query"],
                "category": test["category"],
                "error": str(e),
            })

    # Summary statistics
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    successful = [r for r in results_log if "error" not in r]
    logger.info("Queries: %d total, %d successful, %d failed",
                len(TEST_QUERIES), len(successful), len(TEST_QUERIES) - len(successful))

    if strategy_timings["total"]:
        logger.info("\nTiming (ms) — avg / p50 / p95 / max:")
        for name, times in strategy_timings.items():
            if not times:
                continue
            sorted_t = sorted(times)
            avg = sum(sorted_t) / len(sorted_t)
            p50 = sorted_t[len(sorted_t) // 2]
            p95 = sorted_t[int(len(sorted_t) * 0.95)]
            mx = sorted_t[-1]
            logger.info("  %-12s %7.1f / %7.1f / %7.1f / %7.1f", name, avg, p50, p95, mx)

        logger.info("\nResult counts — avg:")
        for name, counts in strategy_counts.items():
            if counts:
                avg = sum(counts) / len(counts)
                logger.info("  %-12s %5.1f", name, avg)

    # Multi-source coverage: how many results came from 2+ strategies
    multi_source_counts = []
    for entry in successful:
        top = entry.get("top_results", [])
        multi = sum(1 for r in top if len(r.get("sources", [])) > 1)
        multi_source_counts.append(multi)
    if multi_source_counts:
        avg_multi = sum(multi_source_counts) / len(multi_source_counts)
        logger.info("\nMulti-source results (top 5): avg %.1f / query", avg_multi)

    # Save detailed results
    output_path = Path(__file__).parent.parent / "data" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_count": len(TEST_QUERIES),
                "successful": len(successful),
                "failed": len(TEST_QUERIES) - len(successful),
                "results": results_log,
                "summary": {
                    name: {
                        "avg": round(sum(times) / len(times), 1) if times else 0,
                        "p50": round(sorted(times)[len(times) // 2], 1) if times else 0,
                        "max": round(max(times), 1) if times else 0,
                    }
                    for name, times in strategy_timings.items()
                },
            },
            f,
            indent=2,
        )
    logger.info("\nDetailed results saved to: %s", output_path)

    # Cleanup
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
