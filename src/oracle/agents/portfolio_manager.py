"""Portfolio Manager Agent — orchestrates the full prediction pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from oracle.agents.base import BaseAgent
from oracle.agents.messages import Message, MessageBus, MessageType
from oracle.agents.paper_trading import PaperTradingEngine
from oracle.agents.reflection import reflect

logger = structlog.get_logger()

# Conviction → position size mapping
CONVICTION_THRESHOLDS = [
    (90, 10.0),  # >90 = up to risk limit (10%)
    (75, 5.0),   # 75-90 = 5%
    (60, 2.0),   # 60-75 = 2%
]
MIN_CONVICTION = 60  # Below 60 = no trade


def conviction_to_size(conviction: float) -> float:
    """Map conviction score (0-100) to position size (% of portfolio).

    <60 = no trade (0%)
    60-75 = 2%
    75-90 = 5%
    >90 = up to 10% (risk limit)
    """
    if conviction < MIN_CONVICTION:
        return 0.0
    for threshold, size in CONVICTION_THRESHOLDS:
        if conviction >= threshold:
            return size
    return 0.0


class PortfolioManagerAgent(BaseAgent):
    """Orchestrator agent — manages the full pipeline.

    Pipeline:
    1. Detect opportunity (new market, price move, approaching resolution)
    2. Request research from Research Agent
    3. Request quant analysis from Quantitative Agent
    4. Run reflection for bias check
    5. Compute conviction score
    6. Request risk check from Risk Agent
    7. Execute trade via Paper Trading Engine (if approved)
    """

    def __init__(self, bus: MessageBus, trading_engine: PaperTradingEngine) -> None:
        super().__init__(agent_id="portfolio_manager", name="Portfolio Manager", bus=bus)
        self.trading_engine = trading_engine
        self._pending_research: dict[str, dict[str, Any]] = {}
        self._pending_analysis: dict[str, dict[str, Any]] = {}
        self._pending_risk: dict[str, dict[str, Any]] = {}

    async def handle_message(self, message: Message) -> None:
        """Route messages by type."""
        handlers = {
            MessageType.RESEARCH_RESULT: self._handle_research_result,
            MessageType.ANALYSIS_RESULT: self._handle_analysis_result,
            MessageType.TRADE_APPROVED: self._handle_trade_approved,
            MessageType.TRADE_REJECTED: self._handle_trade_rejected,
        }
        handler = handlers.get(message.type)
        if handler:
            await handler(message)

    async def evaluate_opportunity(
        self,
        market_id: str,
        question: str,
        category: str = "other",
        hours_to_resolution: float | None = None,
    ) -> str:
        """Kick off the full evaluation pipeline for a market.

        Returns the trace_id for tracking.
        """
        trace_id = Message(
            from_agent=self.agent_id,
            to_agent="research",
            type=MessageType.RESEARCH_REQUEST,
            payload={},
        ).trace_id

        # Store pipeline context
        self._pending_research[trace_id] = {
            "market_id": market_id,
            "question": question,
            "category": category,
            "hours_to_resolution": hours_to_resolution,
        }

        # 1. Request research
        await self.send(Message(
            from_agent=self.agent_id,
            to_agent="research",
            type=MessageType.RESEARCH_REQUEST,
            payload={"market_id": market_id, "question": question},
            trace_id=trace_id,
        ))

        # 2. Request quant analysis in parallel
        await self.send(Message(
            from_agent=self.agent_id,
            to_agent="quantitative",
            type=MessageType.ANALYSIS_REQUEST,
            payload={"market_id": market_id, "question": question},
            trace_id=trace_id,
        ))

        logger.info(
            "pm.pipeline_started",
            market_id=market_id,
            trace_id=trace_id,
        )
        return trace_id

    async def _handle_research_result(self, message: Message) -> None:
        """Collect research result and check if quant is also done."""
        trace_id = message.trace_id
        context = self._pending_research.get(trace_id)
        if not context:
            return

        context["research"] = message.payload
        logger.info("pm.research_received", trace_id=trace_id)

        # Check if quant analysis is also available
        if "analysis" in context:
            await self._proceed_to_decision(trace_id, context)

    async def _handle_analysis_result(self, message: Message) -> None:
        """Collect quant result and check if research is also done."""
        trace_id = message.trace_id
        context = self._pending_research.get(trace_id)
        if not context:
            return

        context["analysis"] = message.payload
        logger.info("pm.analysis_received", trace_id=trace_id)

        # Check if research is also available
        if "research" in context:
            await self._proceed_to_decision(trace_id, context)

    async def _proceed_to_decision(
        self, trace_id: str, context: dict[str, Any]
    ) -> None:
        """Once both research and quant are in, run reflection and risk check."""
        research = context["research"]
        analysis = context["analysis"]
        market_id = context["market_id"]
        question = context["question"]

        # Compute raw conviction from research confidence + quant signals
        research_confidence = research.get("confidence", 0.5)
        quant_momentum = analysis.get("price_momentum", 0.0)
        quant_accuracy = analysis.get("historical_accuracy", 0.5)

        # Weighted conviction: 60% research, 25% quant accuracy, 15% momentum signal
        raw_conviction = (
            research_confidence * 60
            + quant_accuracy * 25
            + (abs(quant_momentum) * 0.5 + 0.5) * 15  # Normalize momentum to 0-1 range
        )

        # 3. Reflection step — bias detection
        reflection = await reflect(
            question=question,
            thesis=research.get("thesis", ""),
            confidence=research_confidence,
            momentum=quant_momentum,
            evidence_count=len(research.get("evidence", [])),
        )

        # Adjust conviction if biases detected
        if reflection.biases_detected:
            adjustment = len(reflection.biases_detected) * 5  # -5 per bias
            raw_conviction = max(0, raw_conviction - adjustment)
            logger.info(
                "pm.bias_adjustment",
                biases=reflection.biases_detected,
                adjustment=-adjustment,
                trace_id=trace_id,
            )

        conviction = round(raw_conviction, 1)
        size_pct = conviction_to_size(conviction)

        logger.info(
            "pm.conviction",
            market_id=market_id,
            conviction=conviction,
            size_pct=size_pct,
            biases=reflection.biases_detected,
            trace_id=trace_id,
        )

        if size_pct == 0:
            logger.info("pm.no_trade", market_id=market_id, conviction=conviction)
            # Clean up
            self._pending_research.pop(trace_id, None)
            return

        # Use quant recommended size as a cap
        quant_recommended = analysis.get("recommended_size", 10.0)
        size_pct = min(size_pct, quant_recommended)

        # Determine direction from research thesis + price
        current_price = analysis.get("current_price", 0.5)
        direction = "yes" if research_confidence > 0.5 else "no"

        # 4. Risk check
        portfolio_state = self.trading_engine.get_portfolio_state()
        proposal = {
            "market_id": market_id,
            "direction": direction,
            "size_pct": size_pct,
            "price": current_price,
            "conviction": conviction,
            "category": context.get("category", "other"),
            "hours_to_resolution": context.get("hours_to_resolution"),
            "portfolio": portfolio_state,
            "trace_id": trace_id,
            "reflection": reflection.to_dict(),
        }

        self._pending_risk[trace_id] = proposal

        await self.send(Message(
            from_agent=self.agent_id,
            to_agent="risk",
            type=MessageType.RISK_CHECK,
            payload=proposal,
            trace_id=trace_id,
        ))

    async def _handle_trade_approved(self, message: Message) -> None:
        """Execute the approved trade."""
        trace_id = message.trace_id
        proposal = self._pending_risk.pop(trace_id, None)
        if not proposal:
            proposal = message.payload

        market_id = proposal.get("market_id", "")
        adjusted_size = message.payload.get("adjusted_size", proposal.get("size_pct", 0))

        trade = await self.trading_engine.execute_trade(
            market_id=market_id,
            direction=proposal.get("direction", "yes"),
            size_pct=adjusted_size,
            price=proposal.get("price", 0.5),
            conviction=proposal.get("conviction", 0.0),
            trace_id=trace_id,
            category=proposal.get("category", "other"),
            hours_to_resolution=proposal.get("hours_to_resolution"),
        )

        logger.info(
            "pm.trade_executed",
            market_id=market_id,
            trade_id=trade.id,
            value=trade.value,
            trace_id=trace_id,
        )

        # Clean up
        self._pending_research.pop(trace_id, None)

    async def _handle_trade_rejected(self, message: Message) -> None:
        """Log the rejection and clean up."""
        trace_id = message.trace_id
        violations = message.payload.get("violations", [])
        market_id = message.payload.get("market_id", "")

        logger.info(
            "pm.trade_rejected",
            market_id=market_id,
            violations=violations,
            trace_id=trace_id,
        )

        # Clean up
        self._pending_research.pop(trace_id, None)
        self._pending_risk.pop(trace_id, None)
