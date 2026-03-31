"""Quantitative Agent — market metrics, price history, and statistical analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from oracle.agents.base import BaseAgent
from oracle.agents.cache import cached_tool
from oracle.agents.messages import Message, MessageBus, MessageType
from oracle.config import settings

logger = structlog.get_logger()


@dataclass
class QuantReport:
    """Quantitative analysis report for a market."""

    market_id: str
    current_price: float = 0.0
    volume_24h: float = 0.0
    price_momentum: float = 0.0  # -1.0 to 1.0
    similar_markets: list[dict[str, Any]] = field(default_factory=list)
    historical_accuracy: float = 0.0
    recommended_size: float = 0.0  # percentage of portfolio
    liquidity_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "current_price": self.current_price,
            "volume_24h": self.volume_24h,
            "price_momentum": self.price_momentum,
            "similar_markets": self.similar_markets,
            "historical_accuracy": self.historical_accuracy,
            "recommended_size": self.recommended_size,
            "liquidity_score": self.liquidity_score,
            "timestamp": self.timestamp.isoformat(),
        }


# --- Tool functions ---


@cached_tool(ttl=60)
async def get_price_history(market_id: str) -> list[dict[str, Any]]:
    """Fetch price history from Polymarket Gamma API."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.polymarket_api_base}/markets/{market_id}",
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # Extract price snapshots if available
            return data.get("priceHistory", [])
    except Exception as e:
        logger.warning("tool.get_price_history.error", market_id=market_id, error=str(e))
        return []


@cached_tool(ttl=60)
async def calculate_market_metrics(market_id: str) -> dict[str, Any]:
    """Calculate volume, liquidity, and price momentum for a market."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.polymarket_api_base}/markets/{market_id}",
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

        volume = float(data.get("volume", 0) or 0)
        liquidity = float(data.get("liquidity", 0) or 0)

        # Parse price for momentum
        price = data.get("outcomePrices") or data.get("lastTradePrice")
        if isinstance(price, str):
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = 0.5
        elif isinstance(price, list) and price:
            try:
                price = float(price[0])
            except (ValueError, TypeError, IndexError):
                price = 0.5
        else:
            price = price or 0.5

        # Simple momentum: deviation from 0.5 center
        momentum = (price - 0.5) * 2  # Maps [0,1] to [-1,1]

        return {
            "volume_24h": volume,
            "liquidity": liquidity,
            "current_price": price,
            "momentum": momentum,
            "spread": data.get("spread", 0),
        }
    except Exception as e:
        logger.warning("tool.calculate_market_metrics.error", error=str(e))
        return {"volume_24h": 0, "liquidity": 0, "current_price": 0.5, "momentum": 0, "spread": 0}


@cached_tool(ttl=600)
async def find_similar_historical_markets(
    question: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Find similar resolved markets via Qdrant similarity search."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/api/v1/retrieval/search",
                json={"query": question, "top_k": top_k},
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [
                {
                    "text": r.get("text", ""),
                    "score": r.get("score", 0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]
    except Exception as e:
        logger.warning("tool.find_similar_historical.error", error=str(e))
        return []


@cached_tool(ttl=300)
async def correlate_markets(market_ids: tuple[str, ...]) -> dict[str, Any]:
    """Analyze co-movement between markets.

    Returns correlation matrix approximation based on category overlap.
    """
    correlations: dict[str, float] = {}
    categories: dict[str, str] = {}

    for mid in market_ids:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{settings.polymarket_api_base}/markets/{mid}",
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                question = (data.get("question", "") or "").lower()
                # Simple category detection
                if any(kw in question for kw in ["election", "president", "vote"]):
                    categories[mid] = "politics"
                elif any(kw in question for kw in ["bitcoin", "crypto", "eth"]):
                    categories[mid] = "crypto"
                else:
                    categories[mid] = "other"
        except Exception:
            categories[mid] = "unknown"

    # Pairwise correlation based on category match
    for i, m1 in enumerate(market_ids):
        for m2 in market_ids[i + 1:]:
            key = f"{m1}:{m2}"
            if categories.get(m1) == categories.get(m2) and categories.get(m1) != "unknown":
                correlations[key] = 0.7  # Same category = correlated
            else:
                correlations[key] = 0.1

    return {"correlations": correlations, "categories": categories}


class QuantitativeAgent(BaseAgent):
    """Performs quantitative analysis on prediction markets.

    Tools: price history, market metrics, similar markets, correlation analysis.
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(agent_id="quantitative", name="Quantitative Agent", bus=bus)
        self.register_tool("get_price_history", get_price_history)
        self.register_tool("calculate_market_metrics", calculate_market_metrics)
        self.register_tool("find_similar_historical_markets", find_similar_historical_markets)
        self.register_tool("correlate_markets", correlate_markets)

    async def handle_message(self, message: Message) -> None:
        """Handle ANALYSIS_REQUEST messages."""
        if message.type != MessageType.ANALYSIS_REQUEST:
            return

        market_id = message.payload.get("market_id", "")
        question = message.payload.get("question", "")
        trace_id = message.trace_id

        logger.info(
            "quant.start",
            market_id=market_id,
            trace_id=trace_id,
        )

        report = await self.analyze(market_id, question, trace_id)

        await self.send(Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type=MessageType.ANALYSIS_RESULT,
            payload=report.to_dict(),
            trace_id=trace_id,
        ))

    async def analyze(
        self, market_id: str, question: str = "", trace_id: str = ""
    ) -> QuantReport:
        """Run full quantitative analysis on a market."""
        report = QuantReport(market_id=market_id)

        # 1. Market metrics
        metrics = await calculate_market_metrics(market_id=market_id)
        report.current_price = metrics.get("current_price", 0.5)
        report.volume_24h = metrics.get("volume_24h", 0)
        report.price_momentum = metrics.get("momentum", 0)
        report.liquidity_score = self._score_liquidity(
            metrics.get("volume_24h", 0), metrics.get("liquidity", 0)
        )

        # 2. Similar historical markets
        if question:
            similar = await find_similar_historical_markets(question=question, top_k=5)
            report.similar_markets = similar
            report.historical_accuracy = self._estimate_accuracy(similar)

        # 3. Recommended position size based on liquidity and confidence
        report.recommended_size = self._calculate_size(
            liquidity_score=report.liquidity_score,
            momentum_strength=abs(report.price_momentum),
            historical_accuracy=report.historical_accuracy,
        )

        logger.info(
            "quant.complete",
            market_id=market_id,
            price=report.current_price,
            momentum=report.price_momentum,
            recommended_size=report.recommended_size,
            trace_id=trace_id,
        )
        return report

    @staticmethod
    def _score_liquidity(volume: float, liquidity: float) -> float:
        """Score market liquidity from 0-1."""
        # Log-scaled: $100k volume = ~0.5, $1M+ = ~0.9
        if volume <= 0:
            return 0.1
        score = min(1.0, math.log10(max(volume, 1)) / 7)  # log10(10M) = 7
        return round(score, 3)

    @staticmethod
    def _estimate_accuracy(similar_markets: list[dict[str, Any]]) -> float:
        """Estimate prediction accuracy from similar resolved markets."""
        if not similar_markets:
            return 0.5  # No data → assume 50%
        # Average similarity scores as proxy for prediction confidence
        scores = [m.get("score", 0.5) for m in similar_markets]
        return round(sum(scores) / len(scores), 3)

    @staticmethod
    def _calculate_size(
        liquidity_score: float,
        momentum_strength: float,
        historical_accuracy: float,
    ) -> float:
        """Calculate recommended position size as % of portfolio.

        Higher liquidity, stronger momentum, and better historical accuracy
        → larger recommended position.
        """
        base = 2.0  # Minimum 2% if trading
        liquidity_bonus = liquidity_score * 3.0  # Up to 3% for high liquidity
        momentum_bonus = momentum_strength * 2.0  # Up to 2% for strong momentum
        accuracy_bonus = max(0, (historical_accuracy - 0.5)) * 6.0  # Up to 3% for accuracy > 0.5

        size = base + liquidity_bonus + momentum_bonus + accuracy_bonus
        return round(min(size, 10.0), 2)  # Cap at 10%
