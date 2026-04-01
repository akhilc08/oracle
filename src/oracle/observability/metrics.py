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
