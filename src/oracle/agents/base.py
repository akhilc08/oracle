"""Abstract base class for all Oracle agents."""

from __future__ import annotations

import abc
from typing import Any, Callable

import structlog

from oracle.agents.messages import Message, MessageBus

logger = structlog.get_logger()


class BaseAgent(abc.ABC):
    """Abstract base agent.

    Every agent has:
    - agent_id / name for routing
    - A reference to the shared message bus
    - A tools registry for callable tools
    - A run() loop that processes incoming messages
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        bus: MessageBus,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.bus = bus
        self._tools: dict[str, Callable] = {}
        self._running = False

        # Register on the bus
        self.bus.register(self.agent_id)

    def register_tool(self, name: str, func: Callable) -> None:
        """Add a tool to this agent's registry."""
        self._tools[name] = func

    def get_tool(self, name: str) -> Callable | None:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    @property
    def is_running(self) -> bool:
        return self._running

    async def send(self, message: Message) -> None:
        """Send a message via the bus."""
        await self.bus.send(message)

    async def receive(self, timeout: float | None = 5.0) -> Message | None:
        """Wait for next message addressed to this agent."""
        return await self.bus.receive(self.agent_id, timeout=timeout)

    @abc.abstractmethod
    async def handle_message(self, message: Message) -> None:
        """Process a single incoming message. Subclasses must implement."""
        ...

    async def run(self) -> None:
        """Main agent loop — receive and handle messages until stopped."""
        self._running = True
        logger.info("agent.started", agent_id=self.agent_id, name=self.name)

        while self._running:
            msg = await self.receive(timeout=2.0)
            if msg is not None:
                try:
                    await self.handle_message(msg)
                except Exception:
                    logger.exception(
                        "agent.handle_error",
                        agent_id=self.agent_id,
                        msg_type=msg.type.value,
                    )

    def stop(self) -> None:
        """Signal the agent to stop its run loop."""
        self._running = False
        logger.info("agent.stopped", agent_id=self.agent_id)

    @property
    def tools_list(self) -> list[str]:
        """Names of registered tools."""
        return list(self._tools.keys())

    def status(self) -> dict[str, Any]:
        """Agent status summary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "running": self._running,
            "tools": self.tools_list,
            "pending_messages": self.bus.pending_count(self.agent_id),
        }
