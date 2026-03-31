"""FastAPI application for Oracle."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oracle.agents import AgentSystem
from oracle.api.routes import agents, evaluation, health, ingestion, knowledge, markets, retrieval
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager
from oracle.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and cleanup application resources."""
    logger.info("oracle.startup", version="0.1.0")

    # Initialize Neo4j
    app.state.neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await app.state.neo4j.setup_schema()

    # Initialize Qdrant
    app.state.qdrant = QdrantManager(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    await app.state.qdrant.setup_collections()

    # Initialize Agent System
    app.state.agent_system = AgentSystem()

    logger.info("oracle.ready")
    yield

    # Cleanup
    if app.state.agent_system.is_running:
        await app.state.agent_system.stop()
    await app.state.neo4j.close()
    logger.info("oracle.shutdown")


app = FastAPI(
    title="Oracle",
    description="Autonomous AI Prediction Engine for Polymarket",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(knowledge.router, prefix="/api/v1")
app.include_router(markets.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(retrieval.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")
