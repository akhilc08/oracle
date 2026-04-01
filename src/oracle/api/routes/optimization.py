"""Phase 5 API endpoints — routing stats, cache stats, prompt registry, A/B testing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["optimization"])

DB_PATH = "oracle.db"


class ABTestCreate(BaseModel):
    name: str
    prompt_a_version: int
    prompt_b_version: int
    metric: str = "brier_score"


@router.get("/routing/stats")
async def get_routing_stats():
    """Model routing split — percentage local vs Claude."""
    from oracle.routing.classifier import ComplexityClassifier

    classifier = ComplexityClassifier()
    return classifier.routing_stats


@router.get("/cache/stats")
async def get_cache_stats():
    """Semantic cache hit rate and entry count."""
    from oracle.cache.semantic_cache import SemanticCache

    cache = SemanticCache()
    return cache.get_stats()


@router.get("/prompts")
async def list_prompts():
    """List all active prompt templates."""
    from oracle.prompts.registry import PromptRegistry

    registry = PromptRegistry(db_path=DB_PATH)
    registry.initialize()
    templates = registry.list_active()
    return [t.to_dict() for t in templates]


@router.post("/prompts/ab-test")
async def create_ab_test(body: ABTestCreate):
    """Create a new A/B test between two prompt versions."""
    from oracle.prompts.ab_testing import ABTestManager

    manager = ABTestManager(db_path=DB_PATH)
    manager.initialize()
    test_id = manager.create_test(
        name=body.name,
        prompt_a_version=body.prompt_a_version,
        prompt_b_version=body.prompt_b_version,
        metric=body.metric,
    )
    return {"test_id": test_id, "status": "created"}
