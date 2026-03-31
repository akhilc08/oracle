"""Ingestion API endpoints — trigger and monitor data pipelines."""

from fastapi import APIRouter, Request

from oracle.ingestion.entity_resolver import EntityResolver
from oracle.ingestion.news_pipeline import NewsPipeline
from oracle.ingestion.polymarket_client import PolymarketClient

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/news")
async def trigger_news_ingestion(request: Request, query: str | None = None) -> dict:
    """Trigger a news ingestion cycle."""
    entity_resolver = EntityResolver()
    pipeline = NewsPipeline(
        neo4j=request.app.state.neo4j,
        qdrant=request.app.state.qdrant,
        entity_resolver=entity_resolver,
    )
    default_query = "politics OR economy OR markets OR Supreme Court OR elections"
    stats = await pipeline.ingest(query=query or default_query)
    return {"status": "completed", "stats": stats}


@router.post("/markets")
async def trigger_market_sync(request: Request) -> dict:
    """Trigger Polymarket market sync."""
    client = PolymarketClient(neo4j=request.app.state.neo4j)
    stats = await client.sync_markets()
    return {"status": "completed", "stats": stats}


@router.get("/markets/movers")
async def get_market_movers(request: Request, threshold: float = 0.05) -> dict:
    """Get markets with significant price movements."""
    client = PolymarketClient(neo4j=request.app.state.neo4j)
    movers = await client.detect_movers(threshold=threshold)
    return {"count": len(movers), "movers": movers}
