"""Backtest runner — evaluates Oracle against resolved Polymarket markets.

Usage:
    uv run python scripts/backtest.py
    uv run python scripts/backtest.py --limit 100 --category politics
    uv run python scripts/backtest.py --consistency-runs 2 --output results.json

Requirements:
    ORACLE_ANTHROPIC_API_KEY  — for research synthesis, reflection, judge
    ORACLE_NEWSAPI_KEY        — for news at prediction time (free tier = last 30 days)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import structlog

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oracle.config import settings
from oracle.evaluation.backtest_metrics import BacktestAggregator, PredictionRecord
from oracle.evaluation.hallucination import HallucinationDetector
from oracle.evaluation.judge import EvaluationJudge
from oracle.evaluation.post_resolution import PostResolutionEvaluator
from oracle.agents.reflection import reflect
from oracle.agents.research import ResearchAgent, ResearchReport, search_knowledge_base, fetch_latest_news, get_market_data
from oracle.agents.messages import MessageBus

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com"


async def fetch_resolved_markets(limit: int = 50, category: str | None = None) -> list[dict]:
    """Fetch resolved markets from Polymarket Gamma API."""
    params: dict = {
        "resolved": "true",
        "limit": str(limit),
        "order": "resolutionTime",
        "ascending": "false",
    }
    if category:
        params["category"] = category

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{GAMMA_API}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        # Gamma API returns list directly or wrapped
        if isinstance(data, list):
            return data
        return data.get("markets", data.get("data", []))


def parse_outcome(market: dict) -> tuple[bool, float] | None:
    """Extract (resolved_yes, final_price) from a resolved market.

    Returns None if the market outcome can't be determined.
    """
    outcome = market.get("outcome", "").lower()
    prices = market.get("outcomePrices", [])

    if outcome in ("yes", "1"):
        final_price = float(prices[0]) if prices else 1.0
        return True, final_price
    elif outcome in ("no", "0"):
        final_price = float(prices[1]) if len(prices) > 1 else 0.0
        return False, final_price

    # Try resolvedAt + finalPrice fallback
    if market.get("resolved") and prices:
        try:
            p = float(prices[0])
            return p >= 0.5, p
        except (ValueError, TypeError):
            pass

    return None


async def run_prediction_pipeline(
    market: dict,
    judge: EvaluationJudge,
    hallucination_detector: HallucinationDetector,
    consistency_runs: int,
) -> tuple[ResearchReport, dict, dict, dict, float]:
    """Run the full research → reflection → judge → hallucination pipeline.

    Returns:
        (report, reflection_result, judge_result, consistency_result, latency_seconds)
    """
    market_id = market.get("id", str(uuid.uuid4()))
    question = market.get("question", "")
    market_price = float((market.get("outcomePrices") or [0.5])[0])

    t0 = time.perf_counter()

    # 1. Research
    bus = MessageBus()
    agent = ResearchAgent(bus)
    report = await agent.generate_report(market_id, question)

    # 2. Reflection
    reflection = await reflect(
        question=question,
        thesis=report.thesis,
        confidence=report.confidence,
        momentum=market_price - 0.5,  # crude momentum proxy
        evidence_count=len(report.evidence),
    )
    report.confidence = reflection.adjusted_confidence

    # 3. LLM judge
    sources = [e.get("text", "") for e in report.evidence[:5]]
    judge_result = await judge.evaluate(report.thesis, sources)

    # 4. Hallucination detection
    halluc_result = await hallucination_detector.detect(report.thesis, sources)

    # 5. Judge consistency (optional, costs extra API calls)
    consistency_result: dict = {}
    if consistency_runs >= 2:
        consistency_result = await judge.evaluate_consistency(
            report.thesis, sources, n_runs=consistency_runs
        )

    latency = time.perf_counter() - t0
    return report, reflection.to_dict(), judge_result.to_dict(), consistency_result, latency, halluc_result


async def run_backtest(
    limit: int = 50,
    category: str | None = None,
    consistency_runs: int = 0,
    output_path: str | None = None,
    db_path: str = "oracle.db",
) -> None:
    print(f"\nFetching {limit} resolved markets from Polymarket...")
    markets = await fetch_resolved_markets(limit=limit, category=category)

    if not markets:
        print("No resolved markets found. Check your Gamma API connection.")
        return

    print(f"Found {len(markets)} resolved markets. Running prediction pipeline...\n")

    evaluator = PostResolutionEvaluator(db_path=db_path)
    await evaluator.initialize()

    judge = EvaluationJudge()
    hallucination_detector = HallucinationDetector()
    aggregator = BacktestAggregator()

    records: list[PredictionRecord] = []
    skipped = 0

    for i, market in enumerate(markets, 1):
        market_id = market.get("id", "?")
        question = market.get("question", "?")[:70]
        outcome_data = parse_outcome(market)

        if outcome_data is None:
            skipped += 1
            logger.debug("backtest.skip_ambiguous_outcome", market_id=market_id)
            continue

        actual_outcome, final_price = outcome_data
        market_price_at_entry = float((market.get("outcomePrices") or [0.5])[0])
        market_category = market.get("groupItemTagSlug", market.get("category", "other"))

        print(f"[{i}/{len(markets)}] {question}...")

        try:
            report, reflection, judge_result, consistency_result, latency, halluc_result = (
                await run_prediction_pipeline(market, judge, hallucination_detector, consistency_runs)
            )
        except Exception as e:
            logger.warning("backtest.pipeline_error", market_id=market_id, error=str(e))
            skipped += 1
            continue

        predicted_prob = report.confidence
        predicted_direction = "yes" if predicted_prob >= 0.5 else "no"
        bias_flagged = len(reflection.get("biases_detected", [])) > 0
        hallucination_flagged = halluc_result.hallucination_rate > 0.0

        # Persist to DB
        resolution_result = await evaluator.evaluate_prediction(
            trade_id=str(uuid.uuid4()),
            market_id=market_id,
            predicted_prob=predicted_prob,
            predicted_direction=predicted_direction,
            actual_outcome=actual_outcome,
            actual_final_price=final_price,
            market_consensus=market_price_at_entry,
            category=market_category,
            hallucination_flagged=hallucination_flagged,
            bias_flagged=bias_flagged,
            latency_seconds=latency,
        )

        # Build in-memory record for aggregator
        record = PredictionRecord(
            trade_id=resolution_result.trade_id if hasattr(resolution_result, "trade_id") else market_id,
            predicted_prob=predicted_prob,
            market_prob_at_entry=market_price_at_entry,
            actual_outcome=actual_outcome,
            is_correct=resolution_result.is_correct,
            brier_score=resolution_result.brier_score,
            confidence_bucket=resolution_result.confidence_bucket,
            category=market_category,
            latency_seconds=latency,
            hallucination_flagged=hallucination_flagged,
            bias_flagged=bias_flagged,
            judge_scores_run1=judge_result.get("scores", {}),
            judge_scores_run2=consistency_result.get("per_dimension_variance", {}),  # placeholder
        )
        records.append(record)

        status = "✓" if resolution_result.is_correct else "✗"
        print(
            f"  {status} predicted={predicted_prob:.2f} actual={'YES' if actual_outcome else 'NO'} "
            f"brier={resolution_result.brier_score:.3f} latency={latency:.1f}s"
        )

    print(f"\n{len(records)} predictions evaluated, {skipped} skipped.\n")

    if not records:
        print("No records to report.")
        return

    # Compute and print full report
    report_obj = aggregator.compute(records)
    aggregator.print_summary(report_obj)

    # Also pull DB aggregate stats (includes all historical predictions)
    db_stats = await evaluator.aggregate_stats()
    print(f"DB cumulative stats (all-time): {json.dumps(db_stats, indent=2)}\n")

    # Optionally write JSON output
    if output_path:
        output = {
            "backtest_report": report_obj.to_dict(),
            "db_cumulative": db_stats,
            "markets_attempted": len(markets),
            "markets_evaluated": len(records),
            "markets_skipped": skipped,
        }
        Path(output_path).write_text(json.dumps(output, indent=2))
        print(f"Full results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle backtest runner")
    parser.add_argument("--limit", type=int, default=50, help="Number of resolved markets to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (politics, crypto, sports, etc.)")
    parser.add_argument("--consistency-runs", type=int, default=0, help="Judge consistency runs per market (0=skip, costs extra API calls)")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to this file path")
    parser.add_argument("--db", type=str, default="oracle.db", help="SQLite DB path")
    args = parser.parse_args()

    if not settings.anthropic_api_key:
        print("ERROR: ORACLE_ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    asyncio.run(run_backtest(
        limit=args.limit,
        category=args.category,
        consistency_runs=args.consistency_runs,
        output_path=args.output,
        db_path=args.db,
    ))


if __name__ == "__main__":
    main()
