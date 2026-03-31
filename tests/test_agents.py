"""Tests for Phase 3 multi-agent system."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from oracle.agents.cache import ToolCache, cached_tool, get_cache
from oracle.agents.messages import Message, MessageBus, MessageType
from oracle.agents.paper_trading import PaperTradingEngine
from oracle.agents.portfolio_manager import PortfolioManagerAgent, conviction_to_size
from oracle.agents.reflection import _heuristic_reflection
from oracle.agents.risk import (
    MAX_CATEGORY_EXPOSURE_PCT,
    MAX_EXPIRING_RISK_PCT,
    MAX_SINGLE_MARKET_PCT,
    STOP_LOSS_THRESHOLD,
    RiskAgent,
    RiskCheckResult,
)


# --- Message Bus Tests ---


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_register_and_send(self):
        bus = MessageBus()
        bus.register("agent_a")
        bus.register("agent_b")

        msg = Message(
            from_agent="agent_a",
            to_agent="agent_b",
            type=MessageType.RESEARCH_REQUEST,
            payload={"market_id": "m1"},
        )
        await bus.send(msg)

        received = await bus.receive("agent_b", timeout=1.0)
        assert received is not None
        assert received.from_agent == "agent_a"
        assert received.type == MessageType.RESEARCH_REQUEST
        assert received.payload["market_id"] == "m1"

    @pytest.mark.asyncio
    async def test_message_not_delivered_to_wrong_agent(self):
        bus = MessageBus()
        bus.register("agent_a")
        bus.register("agent_b")

        msg = Message(
            from_agent="agent_a",
            to_agent="agent_b",
            type=MessageType.RESEARCH_REQUEST,
            payload={},
        )
        await bus.send(msg)

        received = await bus.receive("agent_a", timeout=0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_broadcast(self):
        bus = MessageBus()
        bus.register("sender")
        bus.register("recv1")
        bus.register("recv2")

        msg = Message(
            from_agent="sender",
            to_agent="*",
            type=MessageType.RESEARCH_REQUEST,
            payload={"broadcast": True},
        )
        await bus.send(msg)

        r1 = await bus.receive("recv1", timeout=1.0)
        r2 = await bus.receive("recv2", timeout=1.0)
        sender_msg = await bus.receive("sender", timeout=0.1)

        assert r1 is not None
        assert r2 is not None
        assert sender_msg is None  # Sender doesn't get broadcast

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        bus = MessageBus()
        bus.register("agent_a")
        received = await bus.receive("agent_a", timeout=0.05)
        assert received is None

    def test_pending_count(self):
        bus = MessageBus()
        bus.register("agent_a")
        assert bus.pending_count("agent_a") == 0
        assert bus.pending_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_message_history(self):
        bus = MessageBus()
        bus.register("a")
        bus.register("b")
        msg = Message(from_agent="a", to_agent="b", type=MessageType.RISK_CHECK, payload={})
        await bus.send(msg)
        assert len(bus.history) == 1
        assert bus.history[0].id == msg.id

    def test_unregister(self):
        bus = MessageBus()
        bus.register("agent_a")
        assert "agent_a" in bus.registered_agents
        bus.unregister("agent_a")
        assert "agent_a" not in bus.registered_agents


# --- Risk Agent Tests ---


class TestRiskAgent:
    def _make_proposal(self, **overrides) -> dict:
        base = {
            "market_id": "market_1",
            "size_pct": 5.0,
            "category": "politics",
            "portfolio": {
                "positions": {},
                "cash": 10000.0,
                "total_value": 10000.0,
            },
            "hours_to_resolution": None,
            "current_pnl_pct": None,
        }
        base.update(overrides)
        return base

    def test_clean_proposal_approved(self):
        bus = MessageBus()
        agent = RiskAgent(bus)
        result = agent.check_risk(self._make_proposal())
        assert result.approved is True
        assert result.violations == []

    def test_single_market_violation(self):
        """Guardrail 1: >10% in a single market."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        result = agent.check_risk(self._make_proposal(size_pct=15.0))
        assert result.approved is False
        assert any("Single market" in v for v in result.violations)

    def test_single_market_with_existing_position(self):
        """Existing position + new trade exceeds 10%."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        proposal = self._make_proposal(
            size_pct=6.0,
            portfolio={
                "positions": {
                    "market_1": {"value": 600.0, "category": "politics"},
                },
                "cash": 9400.0,
                "total_value": 10000.0,
            },
        )
        result = agent.check_risk(proposal)
        assert result.approved is False
        assert any("Single market" in v for v in result.violations)

    def test_category_concentration_violation(self):
        """Guardrail 2: >30% in a single category."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        proposal = self._make_proposal(
            size_pct=5.0,
            category="crypto",
            portfolio={
                "positions": {
                    "btc_market": {"value": 2800.0, "category": "crypto"},
                },
                "cash": 7200.0,
                "total_value": 10000.0,
            },
        )
        result = agent.check_risk(proposal)
        assert result.approved is False
        assert any("Category" in v for v in result.violations)

    def test_expiring_market_violation(self):
        """Guardrail 3: >5% at risk on markets resolving within 24h."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        proposal = self._make_proposal(
            size_pct=6.0,
            hours_to_resolution=12.0,
        )
        result = agent.check_risk(proposal)
        assert result.approved is False
        assert any("Expiring" in v for v in result.violations)

    def test_stop_loss_violation(self):
        """Guardrail 4: Position down ≥50%."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        proposal = self._make_proposal(current_pnl_pct=-55.0)
        result = agent.check_risk(proposal)
        assert result.approved is False
        assert any("Stop-loss" in v for v in result.violations)

    def test_multiple_violations(self):
        """Multiple guardrails can fire at once."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        proposal = self._make_proposal(
            size_pct=15.0,
            hours_to_resolution=6.0,
            current_pnl_pct=-60.0,
        )
        result = agent.check_risk(proposal)
        assert result.approved is False
        assert len(result.violations) >= 3

    @pytest.mark.asyncio
    async def test_risk_message_routing(self):
        """Risk agent sends TRADE_APPROVED/TRADE_REJECTED via bus."""
        bus = MessageBus()
        agent = RiskAgent(bus)
        bus.register("pm")

        # Send an approvable proposal
        msg = Message(
            from_agent="pm",
            to_agent="risk",
            type=MessageType.RISK_CHECK,
            payload=self._make_proposal(),
        )
        await agent.handle_message(msg)

        response = await bus.receive("pm", timeout=1.0)
        assert response is not None
        assert response.type == MessageType.TRADE_APPROVED


# --- Conviction Scoring Tests ---


class TestConvictionScoring:
    def test_below_threshold_no_trade(self):
        assert conviction_to_size(59.9) == 0.0
        assert conviction_to_size(0) == 0.0
        assert conviction_to_size(50) == 0.0

    def test_low_conviction_small_size(self):
        assert conviction_to_size(60) == 2.0
        assert conviction_to_size(65) == 2.0
        assert conviction_to_size(74.9) == 2.0

    def test_medium_conviction(self):
        assert conviction_to_size(75) == 5.0
        assert conviction_to_size(80) == 5.0
        assert conviction_to_size(89.9) == 5.0

    def test_high_conviction_max_size(self):
        assert conviction_to_size(90) == 10.0
        assert conviction_to_size(95) == 10.0
        assert conviction_to_size(100) == 10.0


# --- Paper Trading Tests ---


class TestPaperTrading:
    @pytest.mark.asyncio
    async def test_execute_trade(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path, initial_cash=10000.0)
        await engine.initialize()

        trade = await engine.execute_trade(
            market_id="market_1",
            direction="yes",
            size_pct=5.0,
            price=0.60,
            conviction=75.0,
            trace_id="trace_1",
            category="politics",
        )

        assert trade.market_id == "market_1"
        assert trade.direction == "yes"
        assert trade.value == pytest.approx(500.0, abs=1.0)
        assert engine.cash == pytest.approx(9500.0, abs=1.0)
        assert "market_1" in engine.positions

    @pytest.mark.asyncio
    async def test_close_position_win(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path, initial_cash=10000.0)
        await engine.initialize()

        await engine.execute_trade(
            market_id="market_1",
            direction="yes",
            size_pct=5.0,
            price=0.60,
            conviction=75.0,
        )

        result = await engine.close_position("market_1", outcome="yes")
        assert result["realized_pnl"] > 0  # Bought at 0.60, settled at 1.0
        assert "market_1" not in engine.positions

    @pytest.mark.asyncio
    async def test_close_position_loss(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path, initial_cash=10000.0)
        await engine.initialize()

        await engine.execute_trade(
            market_id="market_1",
            direction="yes",
            size_pct=5.0,
            price=0.60,
            conviction=75.0,
        )

        result = await engine.close_position("market_1", outcome="no")
        assert result["realized_pnl"] < 0  # Bought yes at 0.60, outcome is no → 0

    @pytest.mark.asyncio
    async def test_portfolio_state(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path, initial_cash=10000.0)
        await engine.initialize()

        state = engine.get_portfolio_state()
        assert state["cash"] == 10000.0
        assert state["total_value"] == 10000.0
        assert state["pnl"] == 0.0
        assert state["position_count"] == 0

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path)
        await engine.initialize()

        result = await engine.close_position("nonexistent", outcome="yes")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_trade_exceeding_cash(self, tmp_path):
        """Trade size is capped at available cash."""
        db_path = str(tmp_path / "test.db")
        engine = PaperTradingEngine(db_path=db_path, initial_cash=100.0)
        await engine.initialize()

        trade = await engine.execute_trade(
            market_id="m1",
            direction="yes",
            size_pct=200.0,  # Request 200% — should cap at cash
            price=0.50,
            conviction=90.0,
        )
        assert trade.value <= 100.0
        assert engine.cash >= 0


# --- Tool Cache Tests ---


class TestToolCache:
    def test_put_and_get(self):
        cache = ToolCache()
        cache.put("my_tool", {"key": "val"}, "result_data", ttl=60)
        hit, value = cache.get("my_tool", {"key": "val"})
        assert hit is True
        assert value == "result_data"

    def test_miss(self):
        cache = ToolCache()
        hit, value = cache.get("my_tool", {"key": "val"})
        assert hit is False
        assert value is None

    def test_ttl_expiry(self):
        cache = ToolCache()
        cache.put("my_tool", {"k": "v"}, "data", ttl=0)  # Expires immediately
        time.sleep(0.01)
        hit, value = cache.get("my_tool", {"k": "v"})
        assert hit is False

    def test_stats(self):
        cache = ToolCache()
        cache.put("tool", {}, "data", ttl=60)

        cache.get("tool", {})  # Hit
        cache.get("tool", {})  # Hit
        cache.get("other_tool", {})  # Miss

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == pytest.approx(2 / 3)

    def test_invalidate(self):
        cache = ToolCache()
        cache.put("tool", {"a": 1}, "data", ttl=60)
        cache.invalidate("tool", {"a": 1})
        hit, _ = cache.get("tool", {"a": 1})
        assert hit is False

    def test_clear(self):
        cache = ToolCache()
        cache.put("t1", {}, "d1", ttl=60)
        cache.put("t2", {}, "d2", ttl=60)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_deterministic_key(self):
        cache = ToolCache()
        # Same kwargs in different order should produce same key
        key1 = cache._make_key("tool", {"a": 1, "b": 2})
        key2 = cache._make_key("tool", {"b": 2, "a": 1})
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_cached_tool_decorator(self):
        call_count = 0

        @cached_tool(ttl=60)
        async def my_tool(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Reset the global cache for clean test
        cache = get_cache()
        cache.clear()

        result1 = await my_tool(5)
        result2 = await my_tool(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Second call was cached


# --- Reflection Tests ---


class TestReflection:
    def test_heuristic_overconfidence(self):
        result = _heuristic_reflection(confidence=0.9, momentum=0.0, evidence_count=1)
        assert "overconfidence" in result.biases_detected
        assert result.adjusted_confidence < 0.9

    def test_heuristic_recency_bias(self):
        result = _heuristic_reflection(confidence=0.8, momentum=0.8, evidence_count=5)
        assert "recency bias" in result.biases_detected

    def test_heuristic_anchoring(self):
        # Momentum of 0.5 → market-implied = 0.75, confidence = 0.75 → anchoring
        result = _heuristic_reflection(confidence=0.75, momentum=0.5, evidence_count=5)
        assert "anchoring bias" in result.biases_detected

    def test_heuristic_no_bias(self):
        # confidence=0.6, momentum=0.0 → market_implied=0.5, gap=0.1 → no anchoring
        result = _heuristic_reflection(confidence=0.6, momentum=0.0, evidence_count=5)
        assert result.biases_detected == []
        assert result.adjusted_confidence == 0.6
