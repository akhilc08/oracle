"""Observability API endpoints — traces, cost tracking."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from oracle.observability.tracer import get_tracer

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/traces")
async def list_traces(
    agent: str | None = None,
    market_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Paginated trace list with optional filters."""
    tracer = get_tracer()
    await tracer.initialize()
    traces = await tracer.get_traces(
        agent=agent,
        market_id=market_id,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return {"traces": traces, "count": len(traces), "limit": limit, "offset": offset}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Full trace detail."""
    tracer = get_tracer()
    await tracer.initialize()
    trace = await tracer.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@router.get("/costs")
async def cost_summary():
    """Aggregate cost statistics."""
    tracer = get_tracer()
    await tracer.initialize()
    return await tracer.get_cost_summary()
