"""Health check endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Check system health including Neo4j and Qdrant connectivity."""
    neo4j_ok = False
    qdrant_ok = False

    try:
        neo4j_ok = await request.app.state.neo4j.verify_connectivity()
    except Exception:
        pass

    try:
        qdrant_ok = await request.app.state.qdrant.verify_connectivity()
    except Exception:
        pass

    status = "healthy" if (neo4j_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "services": {
            "neo4j": "up" if neo4j_ok else "down",
            "qdrant": "up" if qdrant_ok else "down",
        },
    }
