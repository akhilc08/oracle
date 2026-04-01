"""Polymarket API integration — poll active markets, store data, link to knowledge graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from oracle.config import settings
from oracle.knowledge.neo4j_client import Neo4jClient

logger = structlog.get_logger()

# Polymarket Gamma API endpoints
MARKETS_ENDPOINT = "/markets"
EVENTS_ENDPOINT = "/events"


class PolymarketClient:
    """Client for Polymarket Gamma API.

    Polls active markets, creates Market nodes in Neo4j,
    and links them to related entities.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._base_url = settings.polymarket_api_base
        self._known_market_ids: set[str] = set()

    async def fetch_active_markets(
        self,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch markets from Polymarket Gamma API."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base_url}{MARKETS_ENDPOINT}",
                params={
                    "limit": limit,
                    "active": str(active).lower(),
                    "closed": str(closed).lower(),
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            markets = resp.json()

        if isinstance(markets, list):
            logger.info("polymarket.fetched", count=len(markets))
            return markets

        # Some endpoints wrap in a data key
        data = markets.get("data", markets.get("markets", []))
        logger.info("polymarket.fetched", count=len(data))
        return data

    async def fetch_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch events (groups of related markets)."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base_url}{EVENTS_ENDPOINT}",
                params={"limit": limit, "active": "true"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        events = data if isinstance(data, list) else data.get("data", [])
        logger.info("polymarket.events_fetched", count=len(events))
        return events

    async def sync_markets(self) -> dict[str, int]:
        """Full sync: fetch markets → create/update Market nodes → detect category.

        Returns stats: markets_synced, new_markets, updated_markets.
        """
        markets = await self.fetch_active_markets(limit=200)
        stats = {"markets_synced": 0, "new_markets": 0, "updated_markets": 0}

        for market in markets:
            market_id = self._extract_id(market)
            if not market_id:
                continue

            is_new = market_id not in self._known_market_ids

            properties = self._market_to_properties(market)
            await self._neo4j.merge_entity("Market", properties)
            self._known_market_ids.add(market_id)

            stats["markets_synced"] += 1
            if is_new:
                stats["new_markets"] += 1
            else:
                stats["updated_markets"] += 1

            # Link to category event if available
            event_slug = market.get("groupItemTitle") or market.get("eventSlug")
            if event_slug:
                await self._neo4j.merge_entity("Event", {
                    "name": event_slug,
                    "status": "active",
                    "date": market.get("endDate", ""),
                })
                await self._neo4j.create_relationship(
                    from_label="Event",
                    from_key_value=event_slug,
                    rel_type="AFFECTS",
                    to_label="Market",
                    to_key_value=market_id,
                )

        logger.info("polymarket.sync_complete", **stats)
        return stats

    def _extract_id(self, market: dict[str, Any]) -> str | None:
        """Extract market ID from API response."""
        return (
            market.get("id")
            or market.get("conditionId")
            or market.get("questionID")
        )

    def _market_to_properties(self, market: dict[str, Any]) -> dict[str, Any]:
        """Convert API market data to Neo4j node properties."""
        market_id = self._extract_id(market)

        # Extract price — different API versions use different fields
        price = (
            market.get("outcomePrices")
            or market.get("lastTradePrice")
            or market.get("bestAsk")
        )
        if isinstance(price, str):
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = None
        if isinstance(price, list) and price:
            try:
                price = float(price[0])
            except (ValueError, TypeError, IndexError):
                price = None

        volume = market.get("volume") or market.get("volumeNum") or 0
        if isinstance(volume, str):
            try:
                volume = float(volume)
            except (ValueError, TypeError):
                volume = 0

        return {
            "polymarket_id": market_id,
            "question": market.get("question", market.get("title", "")),
            "current_price": price,
            "volume": volume,
            "resolution_date": market.get("endDate", ""),
            "category": self._detect_category(market),
            "active": market.get("active", True),
            "description": (market.get("description", "") or "")[:500],
        }

    @staticmethod
    def _detect_category(market: dict[str, Any]) -> str:
        """Detect market category from question/tags."""
        question = (market.get("question", "") or market.get("title", "")).lower()
        tags = [t.lower() for t in (market.get("tags", []) or [])]
        all_text = question + " " + " ".join(tags)

        import re

        categories = {
            "politics": ["election", "president", "congress", "senate", "vote", "poll", "democrat", "republican", "trump", "biden"],
            "economics": ["fed", "interest rate", "inflation", "gdp", "recession", "unemployment", "cpi", "tariff"],
            "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "defi", "nft"],
            "sports": ["nba", "nfl", "mlb", "soccer", "championship", "super bowl", "world cup"],
            "tech": [r"\bai\b", "openai", "google", "apple", r"\bmeta\b", "microsoft", "tesla"],
            "legal": ["supreme court", "ruling", "lawsuit", "indictment", "trial", "verdict"],
            "geopolitics": ["war", "ukraine", "russia", "china", "taiwan", "nato", "sanctions"],
        }

        for category, keywords in categories.items():
            if any(re.search(kw, all_text) for kw in keywords):
                return category

        return "other"

    async def detect_movers(
        self, threshold: float = 0.05
    ) -> list[dict[str, Any]]:
        """Detect markets with significant price movements (>5% move).

        Compares current prices against stored prices in Neo4j.
        """
        markets = await self.fetch_active_markets()
        movers = []

        for market in markets:
            market_id = self._extract_id(market)
            if not market_id:
                continue

            current_props = self._market_to_properties(market)
            current_price = current_props.get("current_price")
            if current_price is None:
                continue

            # Get stored price from Neo4j
            stored = await self._neo4j.get_market_detail(market_id)
            if "error" in stored:
                continue

            stored_price = stored.get("market", {}).get("current_price")
            if stored_price is None:
                continue

            delta = abs(current_price - stored_price)
            if delta >= threshold:
                movers.append({
                    "market_id": market_id,
                    "question": current_props["question"],
                    "previous_price": stored_price,
                    "current_price": current_price,
                    "delta": delta,
                    "direction": "up" if current_price > stored_price else "down",
                    "category": current_props["category"],
                })

        logger.info("polymarket.movers_detected", count=len(movers), threshold=threshold)
        return movers
