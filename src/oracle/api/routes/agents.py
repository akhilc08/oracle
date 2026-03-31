"""API routes for the multi-agent system."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter(prefix="/agents", tags=["agents"])


class EvaluateRequest(BaseModel):
    """Request to evaluate a market through the agent pipeline."""

    market_id: str
    question: str
    category: str = "other"
    hours_to_resolution: float | None = None


@router.post("/start")
async def start_agents(request: Request) -> dict[str, Any]:
    """Start the multi-agent system."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    if system.is_running:
        return {"status": "already_running", **system.status()}

    await system.start()
    return {"status": "started", **system.status()}


@router.get("/status")
async def agent_status(request: Request) -> dict[str, Any]:
    """Get the current status of the agent system."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        return {"status": "not_initialized"}

    return system.status()


@router.post("/stop")
async def stop_agents(request: Request) -> dict[str, Any]:
    """Stop the multi-agent system."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    await system.stop()
    return {"status": "stopped"}


@router.post("/evaluate")
async def evaluate_market(request: Request, body: EvaluateRequest) -> dict[str, Any]:
    """Submit a market for evaluation through the full agent pipeline."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    if not system.is_running:
        raise HTTPException(status_code=400, detail="Agent system not running. POST /agents/start first.")

    trace_id = await system.evaluate_market(
        market_id=body.market_id,
        question=body.question,
        category=body.category,
        hours_to_resolution=body.hours_to_resolution,
    )

    return {
        "status": "evaluation_started",
        "trace_id": trace_id,
        "market_id": body.market_id,
    }


@router.get("/portfolio")
async def get_portfolio(request: Request) -> dict[str, Any]:
    """Get current portfolio state."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    return system.trading_engine.get_portfolio_state()


@router.get("/cache/stats")
async def cache_stats(request: Request) -> dict[str, Any]:
    """Get tool cache statistics."""
    system = getattr(request.app.state, "agent_system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    cache = system.cache
    return {
        "size": cache.size,
        "hits": cache.stats.hits,
        "misses": cache.stats.misses,
        "hit_rate": round(cache.stats.hit_rate, 3),
    }
