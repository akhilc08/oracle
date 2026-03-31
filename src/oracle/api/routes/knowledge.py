"""Knowledge graph API endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/stats")
async def graph_stats(request: Request) -> dict:
    """Get knowledge graph statistics."""
    stats = await request.app.state.neo4j.get_stats()
    return stats


@router.get("/entities/{entity_type}")
async def list_entities(request: Request, entity_type: str, limit: int = 50) -> dict:
    """List entities of a given type."""
    entities = await request.app.state.neo4j.get_entities(entity_type, limit)
    return {"entity_type": entity_type, "count": len(entities), "entities": entities}
