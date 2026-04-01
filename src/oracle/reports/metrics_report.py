"""Metrics report generator — produces comprehensive performance reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from oracle.evaluation.post_resolution import PostResolutionEvaluator
from oracle.observability.tracer import LLMTracer

logger = structlog.get_logger()


class MetricsReportGenerator:
    """Generates comprehensive Markdown performance reports from Oracle data."""

    def __init__(self, db_path: str = "oracle.db") -> None:
        self._db_path = db_path
        self._evaluator = PostResolutionEvaluator(db_path=db_path)
        self._tracer = LLMTracer(db_path=db_path)

    async def initialize(self) -> None:
        await self._evaluator.initialize()
        await self._tracer.initialize()

    async def generate_report(self) -> dict[str, Any]:
        """Generate a full metrics report as structured data."""
        prediction_stats = await self._evaluator.aggregate_stats()
        cost_summary = await self._tracer.get_cost_summary()
        latency_pcts = await self._tracer.get_latency_percentiles()
        portfolio_stats = await self._get_portfolio_stats()
        kg_stats = await self._get_knowledge_graph_stats()
        calibration_data = await self._get_calibration_data()
        top_predictions = await self._get_top_predictions()
        worst_predictions = await self._get_worst_predictions()
        cache_stats = await self._get_cache_stats()
        model_routing = await self._get_model_routing_split()

        total_predictions = prediction_stats.get("total_predictions", 0)
        accuracy = prediction_stats.get("overall_accuracy", 0.0)
        brier = prediction_stats.get("brier_score", 0.0)
        alpha_rate = prediction_stats.get("alpha_rate", 0.0)
        total_cost = cost_summary.get("total_cost_usd", 0.0)
        avg_cost = cost_summary.get("avg_cost_usd", 0.0)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": {
                "total_predictions": total_predictions,
                "accuracy_rate": accuracy,
                "brier_score": brier,
                "alpha_rate": alpha_rate,
                "total_cost_usd": total_cost,
                "avg_cost_per_prediction": avg_cost,
                "portfolio_pnl": portfolio_stats.get("pnl", 0.0),
            },
            "prediction_performance": {
                "by_category": prediction_stats.get("by_category", {}),
                "calibration": calibration_data,
                "top_predictions": top_predictions,
                "worst_predictions": worst_predictions,
            },
            "infrastructure_performance": {
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "model_routing_split": model_routing,
                "latency_p50_ms": latency_pcts.get("p50", 0.0),
                "latency_p99_ms": latency_pcts.get("p99", 0.0),
                "total_traces": cost_summary.get("total_traces", 0),
                "avg_latency_ms": cost_summary.get("avg_latency_ms", 0.0),
            },
            "cost_analysis": {
                "total_cost_usd": total_cost,
                "cost_per_prediction": avg_cost,
                "total_prompt_tokens": cost_summary.get("total_prompt_tokens", 0),
                "total_completion_tokens": cost_summary.get("total_completion_tokens", 0),
            },
            "knowledge_graph": kg_stats,
            "resume_bullets": _generate_resume_bullets(
                total_predictions, accuracy, brier, alpha_rate, total_cost, avg_cost,
                portfolio_stats, kg_stats,
            ),
        }
        return report

    async def generate_markdown(self) -> str:
        """Generate the report as formatted Markdown."""
        r = await self.generate_report()
        es = r["executive_summary"]
        pp = r["prediction_performance"]
        ip = r["infrastructure_performance"]
        ca = r["cost_analysis"]
        kg = r["knowledge_graph"]

        lines = [
            "# Oracle Metrics Report",
            f"*Generated: {r['generated_at']}*",
            "",
            "## 1. Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Predictions | {es['total_predictions']} |",
            f"| Accuracy Rate | {es['accuracy_rate']:.1%} |",
            f"| Brier Score | {es['brier_score']:.4f} |",
            f"| Alpha Rate | {es['alpha_rate']:.1%} |",
            f"| Portfolio P&L | ${es['portfolio_pnl']:.2f} |",
            f"| Total Cost | ${es['total_cost_usd']:.2f} |",
            f"| Avg Cost/Prediction | ${es['avg_cost_per_prediction']:.4f} |",
            "",
            "## 2. Prediction Performance",
            "",
            "### By Category",
            "",
            "| Category | Count | Accuracy | Brier Score |",
            "|----------|-------|----------|-------------|",
        ]

        for cat, data in pp.get("by_category", {}).items():
            lines.append(
                f"| {cat} | {data['count']} | {data['accuracy']:.1%} | {data['brier_score']:.4f} |"
            )

        lines.extend([
            "",
            "### Calibration",
            "",
            "| Bucket | Predicted | Actual | Count |",
            "|--------|-----------|--------|-------|",
        ])

        for bucket in pp.get("calibration", []):
            lines.append(
                f"| {bucket.get('bucket', '')} | {bucket.get('predicted_avg', 0):.2f} "
                f"| {bucket.get('actual_rate', 0):.2f} | {bucket.get('count', 0)} |"
            )

        lines.extend([
            "",
            "### Top Predictions",
            "",
        ])
        for p in pp.get("top_predictions", []):
            lines.append(
                f"- **{p.get('market_id', '???')}**: Brier={p.get('brier_score', 0):.4f}, "
                f"Correct={'Yes' if p.get('is_correct') else 'No'}"
            )

        lines.extend([
            "",
            "## 3. Infrastructure Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Cache Hit Rate | {ip['cache_hit_rate']:.1%} |",
            f"| Latency p50 | {ip['latency_p50_ms']:.0f}ms |",
            f"| Latency p99 | {ip['latency_p99_ms']:.0f}ms |",
            f"| Total LLM Traces | {ip['total_traces']} |",
            f"| Avg Latency | {ip['avg_latency_ms']:.0f}ms |",
            "",
            "### Model Routing Split",
            "",
        ])
        for model, count in ip.get("model_routing_split", {}).items():
            lines.append(f"- **{model}**: {count} calls")

        lines.extend([
            "",
            "## 4. Cost Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Cost | ${ca['total_cost_usd']:.2f} |",
            f"| Cost per Prediction | ${ca['cost_per_prediction']:.4f} |",
            f"| Total Prompt Tokens | {ca['total_prompt_tokens']:,} |",
            f"| Total Completion Tokens | {ca['total_completion_tokens']:,} |",
            "",
            "## 5. Knowledge Graph Stats",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Entities | {kg.get('total_nodes', 0)} |",
            f"| Total Edges | {kg.get('total_edges', 0)} |",
            "",
            "## 6. Resume Bullets",
            "",
        ])
        for bullet in r.get("resume_bullets", []):
            lines.append(f"- {bullet}")

        lines.append("")
        return "\n".join(lines)

    # --- Data fetchers ---

    async def _get_portfolio_stats(self) -> dict[str, Any]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT total_value, cash, pnl FROM portfolio_history "
                "ORDER BY id DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return {"total_value": 10000.0, "cash": 10000.0, "pnl": 0.0}
                return {
                    "total_value": row[0],
                    "cash": row[1],
                    "pnl": row[2],
                }

    async def _get_knowledge_graph_stats(self) -> dict[str, Any]:
        # Returns zeroes if Neo4j unavailable — report still generates
        try:
            from oracle.knowledge.neo4j_client import Neo4jClient
            from oracle.config import settings

            client = Neo4jClient(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
            )
            stats = await client.get_stats()
            await client.close()
            return stats
        except Exception:
            return {"total_nodes": 0, "total_edges": 0, "node_counts": {}, "edge_counts": {}}

    async def _get_calibration_data(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT confidence_bucket, "
                "AVG(predicted_prob) as predicted_avg, "
                "AVG(CAST(actual_outcome AS REAL)) as actual_rate, "
                "COUNT(*) as cnt "
                "FROM prediction_outcomes "
                "GROUP BY confidence_bucket ORDER BY confidence_bucket"
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "bucket": row[0],
                        "predicted_avg": round(row[1], 4),
                        "actual_rate": round(row[2], 4),
                        "count": row[3],
                    }
                    for row in rows
                ]

    async def _get_top_predictions(self, limit: int = 5) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT trade_id, market_id, brier_score, is_correct, is_alpha "
                "FROM prediction_outcomes "
                "WHERE is_correct = 1 ORDER BY brier_score ASC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "trade_id": r[0],
                        "market_id": r[1],
                        "brier_score": r[2],
                        "is_correct": bool(r[3]),
                        "is_alpha": bool(r[4]),
                    }
                    for r in rows
                ]

    async def _get_worst_predictions(self, limit: int = 5) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT trade_id, market_id, brier_score, is_correct "
                "FROM prediction_outcomes "
                "ORDER BY brier_score DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "trade_id": r[0],
                        "market_id": r[1],
                        "brier_score": r[2],
                        "is_correct": bool(r[3]),
                    }
                    for r in rows
                ]

    async def _get_cache_stats(self) -> dict[str, Any]:
        try:
            from oracle.agents.cache import get_cache
            cache = get_cache()
            return {
                "size": cache.size,
                "hits": cache.stats.hits,
                "misses": cache.stats.misses,
                "hit_rate": cache.stats.hit_rate,
            }
        except Exception:
            return {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}

    async def _get_model_routing_split(self) -> dict[str, int]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT model, COUNT(*) as cnt FROM llm_traces "
                "WHERE model != '' GROUP BY model ORDER BY cnt DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return {r[0]: r[1] for r in rows}


def _generate_resume_bullets(
    total_predictions: int,
    accuracy: float,
    brier: float,
    alpha_rate: float,
    total_cost: float,
    avg_cost: float,
    portfolio_stats: dict[str, Any],
    kg_stats: dict[str, Any],
) -> list[str]:
    """Auto-generate resume bullet points based on actual metrics."""
    bullets = []

    if total_predictions > 0:
        bullets.append(
            f"Built autonomous AI prediction engine achieving {accuracy:.0%} accuracy "
            f"across {total_predictions} predictions with {brier:.3f} Brier score"
        )

    if alpha_rate > 0:
        bullets.append(
            f"Generated alpha on {alpha_rate:.0%} of trades by diverging >10% from "
            f"market consensus and being correct"
        )

    pnl = portfolio_stats.get("pnl", 0.0)
    if pnl != 0:
        direction = "profit" if pnl > 0 else "loss"
        bullets.append(
            f"Paper trading portfolio: ${pnl:+.2f} {direction} via conviction-weighted "
            f"position sizing with multi-agent risk management"
        )

    if total_cost > 0:
        bullets.append(
            f"Optimized LLM costs to ${avg_cost:.4f}/prediction (${total_cost:.2f} total) "
            f"via model routing and semantic caching"
        )

    total_nodes = kg_stats.get("total_nodes", 0)
    total_edges = kg_stats.get("total_edges", 0)
    if total_nodes > 0:
        bullets.append(
            f"Maintained knowledge graph with {total_nodes} entities and {total_edges} "
            f"relationships for multi-hop reasoning"
        )

    bullets.append(
        "Designed multi-agent architecture (Research, Quant, Risk, PM) with "
        "LLM-as-judge evaluation, hallucination detection, and calibration tracking"
    )

    return bullets
