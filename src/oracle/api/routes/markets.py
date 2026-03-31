"""Polymarket market endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("/")
async def list_markets(request: Request, limit: int = 50, active_only: bool = True) -> dict:
    """List tracked Polymarket markets."""
    markets = await request.app.state.neo4j.get_markets(limit=limit, active_only=active_only)
    return {"count": len(markets), "markets": markets}


@router.get("/{market_id}")
async def get_market(request: Request, market_id: str) -> dict:
    """Get detailed market information including related entities."""
    market = await request.app.state.neo4j.get_market_detail(market_id)
    return market
