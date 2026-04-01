"""Prometheus metrics for Oracle — prediction performance, cost, infrastructure."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from fastapi import APIRouter

router = APIRouter(tags=["observability"])

# Custom registry to avoid default process/platform collectors in tests
REGISTRY = CollectorRegistry()

# --- Prediction metrics ---

predictions_total = Counter(
    "oracle_predictions_total",
    "Total number of predictions made",
    labelnames=["category", "outcome"],
    registry=REGISTRY,
)

brier_score = Gauge(
    "oracle_brier_score",
    "Current aggregate Brier score (lower is better)",
    registry=REGISTRY,
)

accuracy_rate = Gauge(
    "oracle_accuracy_rate",
    "Current aggregate prediction accuracy rate",
    registry=REGISTRY,
)

cache_hit_rate = Gauge(
    "oracle_cache_hit_rate",
    "Tool cache hit rate",
    registry=REGISTRY,
)

cost_per_prediction = Histogram(
    "oracle_cost_per_prediction",
    "Cost in USD per prediction",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

# --- LLM metrics ---

llm_latency_seconds = Histogram(
    "oracle_llm_latency_seconds",
    "LLM API call latency in seconds",
    labelnames=["model", "agent"],
    registry=REGISTRY,
)

# --- Trading metrics ---

trade_conviction = Histogram(
    "oracle_trade_conviction",
    "Conviction score at trade execution",
    buckets=(60, 65, 70, 75, 80, 85, 90, 95, 100),
    registry=REGISTRY,
)

portfolio_value = Gauge(
    "oracle_portfolio_value",
    "Current total portfolio value in USD",
    registry=REGISTRY,
)

active_positions = Gauge(
    "oracle_active_positions",
    "Number of currently open positions",
    registry=REGISTRY,
)

# --- Extended backtest / calibration metrics ---

log_loss = Gauge(
    "oracle_log_loss",
    "Current aggregate log loss (lower is better; random≈0.69)",
    registry=REGISTRY,
)

ev_per_trade = Gauge(
    "oracle_ev_per_trade",
    "Average expected value per trade (predicted_prob - market_prob_at_entry)",
    registry=REGISTRY,
)

market_adjusted_return = Gauge(
    "oracle_market_adjusted_return",
    "Accuracy rate minus average market-implied probability at entry",
    registry=REGISTRY,
)

judge_consistency = Gauge(
    "oracle_judge_consistency",
    "LLM judge consistency score across repeated evaluations (0–1)",
    registry=REGISTRY,
)

bias_detection_recall = Gauge(
    "oracle_bias_detection_recall",
    "Of bias-flagged predictions, fraction that resolved against market consensus",
    registry=REGISTRY,
)

hallucination_catch_rate = Gauge(
    "oracle_hallucination_catch_rate",
    "Fraction of predictions where hallucination detector flagged ≥1 claim",
    registry=REGISTRY,
)

resolution_rate = Gauge(
    "oracle_resolution_rate",
    "Fraction of evaluated markets that have resolved",
    registry=REGISTRY,
)

pipeline_latency_seconds = Histogram(
    "oracle_pipeline_latency_seconds",
    "End-to-end prediction pipeline latency in seconds",
    buckets=(5, 10, 15, 20, 30, 45, 60, 90, 120),
    registry=REGISTRY,
)

hit_rate_by_tier = Gauge(
    "oracle_hit_rate_by_tier",
    "Prediction accuracy per confidence bucket",
    labelnames=["bucket"],
    registry=REGISTRY,
)

calibration_error = Gauge(
    "oracle_calibration_error",
    "Mean absolute deviation between predicted probability and actual resolution rate",
    registry=REGISTRY,
)

# --- Ingestion metrics ---

ingestion_docs_total = Counter(
    "oracle_ingestion_docs_total",
    "Total documents ingested",
    labelnames=["source"],
    registry=REGISTRY,
)


# --- FastAPI endpoint ---


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import Response

    data = generate_latest(REGISTRY)
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")
