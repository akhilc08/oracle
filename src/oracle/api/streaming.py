"""SSE streaming backend — real-time event broadcasting to connected clients."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = structlog.get_logger()

router = APIRouter(tags=["streaming"])


class SSEManager:
    """Manages Server-Sent Event connections and broadcasts events to all clients.

    Each connected client gets an asyncio.Queue. publish() fans out to all queues.
    Supports Last-Event-ID for reconnection.
    """

    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[str | None]] = set()
        self._sequence: int = 0
        self._recent_events: list[dict[str, Any]] = []  # Ring buffer for reconnection
        self._max_recent = 500

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def connect(self) -> asyncio.Queue[str | None]:
        """Register a new SSE client. Returns the queue to read from."""
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        self._clients.add(queue)
        logger.info("sse.client_connected", total_clients=len(self._clients))
        return queue

    def disconnect(self, queue: asyncio.Queue[str | None]) -> None:
        """Remove a client queue."""
        self._clients.discard(queue)
        logger.info("sse.client_disconnected", total_clients=len(self._clients))

    async def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        agent: str | None = None,
    ) -> None:
        """Broadcast an event to all connected clients.

        Args:
            event_type: One of agent_action, trade_executed, market_resolved,
                        graph_updated, alert, portfolio_update
            data: Event payload
            agent: Agent name (research, quant, risk, pm)
        """
        self._sequence += 1
        event = {
            "id": uuid.uuid4().hex,
            "seq": self._sequence,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "payload": data,
        }

        # Store for reconnection
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_recent:
            self._recent_events = self._recent_events[-self._max_recent :]

        # Format as SSE
        sse_data = _format_sse(event)

        # Fan out to all clients
        disconnected: list[asyncio.Queue[str | None]] = []
        for queue in self._clients:
            try:
                queue.put_nowait(sse_data)
            except asyncio.QueueFull:
                disconnected.append(queue)

        for q in disconnected:
            self._clients.discard(q)

    def get_events_since(self, last_event_seq: int) -> list[str]:
        """Get all SSE-formatted events after a given sequence number."""
        return [
            _format_sse(e) for e in self._recent_events if e["seq"] > last_event_seq
        ]


def _format_sse(event: dict[str, Any]) -> str:
    """Format an event dict as an SSE message string."""
    lines = [
        f"id: {event['seq']}",
        f"event: {event['type']}",
        f"data: {json.dumps(event)}",
        "",
        "",
    ]
    return "\n".join(lines)


# --- Singleton ---

_sse_manager: SSEManager | None = None


def get_sse_manager() -> SSEManager:
    """Get or create the global SSEManager singleton."""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEManager()
    return _sse_manager


# --- FastAPI endpoint ---


@router.get("/stream/events")
async def sse_stream(request: Request) -> StreamingResponse:
    """SSE endpoint — streams real-time events to the dashboard."""
    manager = get_sse_manager()
    queue = manager.connect()

    # Check Last-Event-ID for reconnection
    last_id = request.headers.get("Last-Event-ID")
    if last_id:
        try:
            seq = int(last_id)
            missed = manager.get_events_since(seq)
            for msg in missed:
                await queue.put(msg)
        except (ValueError, TypeError):
            pass

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if data is None:
                        break
                    yield data
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            manager.disconnect(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
