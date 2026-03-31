"""Evaluation API endpoints — stats, calibration, post-mortems."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from oracle.evaluation.calibration import CalibrationMonitor
from oracle.evaluation.post_mortem import PostMortemGenerator
from oracle.evaluation.post_resolution import PostResolutionEvaluator

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

DB_PATH = "oracle.db"


@router.get("/stats")
async def get_evaluation_stats():
    """Aggregate prediction accuracy, Brier score, and alpha rate."""
    evaluator = PostResolutionEvaluator(db_path=DB_PATH)
    await evaluator.initialize()
    stats = await evaluator.aggregate_stats()
    return stats


@router.get("/calibration")
async def get_calibration_data():
    """Calibration chart data for frontend plotting."""
    monitor = CalibrationMonitor(db_path=DB_PATH)
    await monitor.initialize()
    data = await monitor.get_calibration_chart_data()
    return data


@router.get("/post-mortem/{trade_id}")
async def get_post_mortem(trade_id: str):
    """Retrieve post-mortem analysis for a specific trade."""
    generator = PostMortemGenerator(db_path=DB_PATH)
    await generator.initialize()
    pm = await generator.get_post_mortem(trade_id)
    if pm is None:
        raise HTTPException(status_code=404, detail=f"No post-mortem found for trade {trade_id}")
    return pm.to_dict()
