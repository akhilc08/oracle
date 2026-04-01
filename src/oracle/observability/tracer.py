"""LLM trace system — records every LLM call with cost tracking and latency."""

from __future__ import annotations

import functools
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

import aiosqlite
import structlog

logger = structlog.get_logger()

TRACE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_traces (
    trace_id TEXT PRIMARY KEY,
    parent_trace_id TEXT DEFAULT NULL,
    agent TEXT NOT NULL,
    prompt_template TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0.0,
    model TEXT DEFAULT '',
    market_id TEXT DEFAULT '',
    evaluation_scores TEXT DEFAULT '{}',
    cost_usd REAL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
"""

# Cost per 1M tokens (USD)
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
}

# Default cost if model not found
DEFAULT_COST = {"input": 1.00, "output": 5.00}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    costs = MODEL_COSTS.get(model, DEFAULT_COST)
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class TraceRecord:
    """In-flight trace record that gets populated during execution."""

    trace_id: str
    agent: str
    prompt_template: str = ""
    market_id: str = ""
    parent_trace_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    evaluation_scores: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    created_at: str = ""

    def record(
        self,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        evaluation_scores: dict[str, Any] | None = None,
    ) -> None:
        """Record completion data for this trace."""
        if model:
            self.model = model
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if evaluation_scores:
            self.evaluation_scores.update(evaluation_scores)
        self.cost_usd = estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


class LLMTracer:
    """Stores LLM call traces in SQLite for observability and cost tracking."""

    def __init__(self, db_path: str = "oracle.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create the llm_traces table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(TRACE_SCHEMA_SQL)
            await db.commit()

    @asynccontextmanager
    async def trace(
        self,
        agent: str,
        template: str = "",
        market_id: str = "",
        parent_trace_id: str | None = None,
    ) -> AsyncIterator[TraceRecord]:
        """Context manager for tracing an LLM call.

        Usage:
            async with tracer.trace("research", "synthesis", market_id) as t:
                response = await call_llm(...)
                t.record(model="claude-haiku-4-5-20251001", prompt_tokens=500, completion_tokens=200)
        """
        record = TraceRecord(
            trace_id=uuid.uuid4().hex,
            agent=agent,
            prompt_template=template,
            market_id=market_id,
            parent_trace_id=parent_trace_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        start = time.monotonic()
        try:
            yield record
        finally:
            record.latency_ms = round((time.monotonic() - start) * 1000, 2)
            record.cost_usd = estimate_cost(
                record.model, record.prompt_tokens, record.completion_tokens
            )
            await self._save_trace(record)

    async def _save_trace(self, record: TraceRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO llm_traces "
                "(trace_id, parent_trace_id, agent, prompt_template, prompt_tokens, "
                "completion_tokens, latency_ms, model, market_id, evaluation_scores, "
                "cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.trace_id,
                    record.parent_trace_id,
                    record.agent,
                    record.prompt_template,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.latency_ms,
                    record.model,
                    record.market_id,
                    json.dumps(record.evaluation_scores),
                    record.cost_usd,
                    record.created_at,
                ),
            )
            await db.commit()

    async def get_traces(
        self,
        agent: str | None = None,
        market_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get paginated trace list with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if market_id:
            conditions.append("market_id = ?")
            params.append(market_id)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            query = (
                f"SELECT * FROM llm_traces {where} "
                f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            params.extend([limit, offset])
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [_row_to_dict(row) for row in rows]

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get full detail for a single trace."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM llm_traces WHERE trace_id = ?", (trace_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return _row_to_dict(row) if row else None

    async def get_cost_summary(self) -> dict[str, Any]:
        """Aggregate cost statistics."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) as total, SUM(cost_usd) as total_cost, "
                "AVG(cost_usd) as avg_cost, SUM(prompt_tokens) as total_prompt, "
                "SUM(completion_tokens) as total_completion, "
                "AVG(latency_ms) as avg_latency "
                "FROM llm_traces"
            ) as cursor:
                row = await cursor.fetchone()
                if not row or row[0] == 0:
                    return {
                        "total_traces": 0,
                        "total_cost_usd": 0.0,
                        "avg_cost_usd": 0.0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "avg_latency_ms": 0.0,
                    }
                return {
                    "total_traces": row[0],
                    "total_cost_usd": round(row[1] or 0, 4),
                    "avg_cost_usd": round(row[2] or 0, 6),
                    "total_prompt_tokens": row[3] or 0,
                    "total_completion_tokens": row[4] or 0,
                    "avg_latency_ms": round(row[5] or 0, 2),
                }

    async def get_latency_percentiles(self) -> dict[str, float]:
        """Compute p50 and p99 latency from traces."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT latency_ms FROM llm_traces ORDER BY latency_ms"
            ) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    return {"p50": 0.0, "p99": 0.0}
                latencies = [r[0] for r in rows]
                n = len(latencies)
                p50 = latencies[int(n * 0.5)] if n > 0 else 0.0
                p99 = latencies[int(n * 0.99)] if n > 0 else 0.0
                return {"p50": round(p50, 2), "p99": round(p99, 2)}


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to dict, parsing JSON fields."""
    d = dict(row)
    if "evaluation_scores" in d and isinstance(d["evaluation_scores"], str):
        try:
            d["evaluation_scores"] = json.loads(d["evaluation_scores"])
        except json.JSONDecodeError:
            d["evaluation_scores"] = {}
    return d


def traced(agent: str, template: str = "") -> Callable:
    """Decorator for tracing async agent methods.

    Usage:
        @traced(agent="research")
        async def synthesize(self, market_id: str, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            market_id = kwargs.get("market_id", "")
            async with tracer.trace(agent, template, market_id):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# --- Singleton ---

_tracer: LLMTracer | None = None


def get_tracer(db_path: str = "oracle.db") -> LLMTracer:
    """Get or create the global LLMTracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer(db_path=db_path)
    return _tracer
