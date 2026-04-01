"""Ingestion API endpoints — trigger and monitor data pipelines."""

from fastapi import APIRouter, Request

from oracle.ingestion.entity_resolver import EntityResolver
from oracle.ingestion.news_pipeline import NewsPipeline
from oracle.ingestion.polymarket_client import PolymarketClient
from oracle.ingestion.scheduler import IngestionScheduler

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

# Module-level scheduler instance (initialized on first status request or app startup)
_scheduler: IngestionScheduler | None = None


def _get_scheduler(request: Request) -> IngestionScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = IngestionScheduler(
            neo4j=getattr(request.app.state, "neo4j", None),
            qdrant=getattr(request.app.state, "qdrant", None),
        )
    return _scheduler


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


@router.get("/status")
async def ingestion_status(request: Request) -> dict:
    """Show last run time and document count per ingestion source."""
    scheduler = _get_scheduler(request)
    return scheduler.get_status()
