"""Message bus and message schema for inter-agent communication."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MessageType(str, Enum):
    """Types of messages exchanged between agents."""

    RESEARCH_REQUEST = "RESEARCH_REQUEST"
    RESEARCH_RESULT = "RESEARCH_RESULT"
    ANALYSIS_REQUEST = "ANALYSIS_REQUEST"
    ANALYSIS_RESULT = "ANALYSIS_RESULT"
    RISK_CHECK = "RISK_CHECK"
    RISK_RESULT = "RISK_RESULT"
    TRADE_PROPOSAL = "TRADE_PROPOSAL"
    TRADE_APPROVED = "TRADE_APPROVED"
    TRADE_REJECTED = "TRADE_REJECTED"


@dataclass
class Message:
    """Inter-agent message."""

    from_agent: str
    to_agent: str
    type: MessageType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class MessageBus:
    """Async message bus using per-agent queues.

    Each agent gets its own asyncio.Queue. Messages are routed
    by to_agent field. Supports broadcast via to_agent="*".
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Message]] = {}
        self._history: list[Message] = []

    def register(self, agent_id: str) -> None:
        """Register an agent on the bus."""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
            logger.info("bus.register", agent_id=agent_id)

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the bus."""
        self._queues.pop(agent_id, None)

    async def send(self, message: Message) -> None:
        """Route a message to the target agent's queue."""
        self._history.append(message)
        logger.debug(
            "bus.send",
            msg_id=message.id,
            from_agent=message.from_agent,
            to_agent=message.to_agent,
            type=message.type.value,
        )

        if message.to_agent == "*":
            # Broadcast to all except sender
            for agent_id, queue in self._queues.items():
                if agent_id != message.from_agent:
                    await queue.put(message)
        else:
            queue = self._queues.get(message.to_agent)
            if queue is not None:
                await queue.put(message)
            else:
                logger.warning("bus.no_recipient", to_agent=message.to_agent)

    async def receive(self, agent_id: str, timeout: float | None = None) -> Message | None:
        """Wait for the next message for an agent. Returns None on timeout."""
        queue = self._queues.get(agent_id)
        if queue is None:
            return None
        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except asyncio.TimeoutError:
            return None

    def pending_count(self, agent_id: str) -> int:
        """Number of pending messages for an agent."""
        queue = self._queues.get(agent_id)
        return queue.qsize() if queue else 0

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._queues.keys())
