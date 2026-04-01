"""Reports API endpoints — full metrics report."""

from __future__ import annotations

from fastapi import APIRouter

from oracle.reports.metrics_report import MetricsReportGenerator

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/metrics")
async def metrics_report():
    """Full metrics report as JSON."""
    generator = MetricsReportGenerator()
    await generator.initialize()
    return await generator.generate_report()
