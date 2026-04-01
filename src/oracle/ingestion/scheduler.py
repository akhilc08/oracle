"""Ingestion scheduler — coordinates all ingestion pipelines on their respective intervals."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

from oracle.config import settings
from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()


class IngestionScheduler:
    """Coordinates all ingestion pipelines with their respective schedules.

    Intervals:
    - twitter: continuous streaming (when API key available)
    - reddit: every 30 minutes
    - audio: daily (86400s)
    - gov_scrapers: every 6 hours (21600s)
    - polling: every 12 hours (43200s)
    - news: every 15 minutes (900s)
    """

    SCHEDULES: dict[str, int] = {
        "news": 900,        # 15 minutes
        "reddit": 1800,     # 30 minutes
        "gov_scrapers": 21600,   # 6 hours
        "polling": 43200,   # 12 hours
        "audio": 86400,     # daily
    }

    # Source ordering for startup — most critical first
    SOURCE_ORDER = ["news", "reddit", "gov_scrapers", "polling", "audio", "twitter"]

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        qdrant: QdrantManager | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._entity_resolver = EntityResolver()
        self._running = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._last_run: dict[str, str] = {}
        self._doc_counts: dict[str, int] = {}
        self._errors: dict[str, str] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get status of all ingestion sources."""
        sources: dict[str, Any] = {}
        for source in self.SOURCE_ORDER:
            sources[source] = {
                "last_run": self._last_run.get(source, "never"),
                "doc_count": self._doc_counts.get(source, 0),
                "interval_seconds": self.SCHEDULES.get(source, 0),
                "status": "running" if source in self._tasks and not self._tasks[source].done() else "idle",
                "error": self._errors.get(source, ""),
            }
        return {"running": self._running, "sources": sources}

    async def start(self) -> None:
        """Start all scheduled ingestion tasks."""
        if self._running:
            logger.warning("scheduler.already_running")
            return

        self._running = True
        logger.info("scheduler.starting", sources=self.SOURCE_ORDER)

        # Start periodic tasks in source order
        for source in self.SOURCE_ORDER:
            if source == "twitter":
                self._tasks[source] = asyncio.create_task(
                    self._run_twitter_stream(), name=f"ingest_{source}"
                )
            else:
                interval = self.SCHEDULES[source]
                self._tasks[source] = asyncio.create_task(
                    self._run_periodic(source, interval), name=f"ingest_{source}"
                )

        logger.info("scheduler.started", task_count=len(self._tasks))

    async def stop(self) -> None:
        """Stop all ingestion tasks."""
        self._running = False
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("scheduler.stopped")

    async def _run_periodic(self, source: str, interval: int) -> None:
        """Run a source's ingestion on a fixed interval."""
        while self._running:
            try:
                await self._run_source(source)
            except Exception as e:
                self._errors[source] = str(e)
                logger.error("scheduler.source_error", source=source, error=str(e))
            await asyncio.sleep(interval)

    async def _run_source(self, source: str) -> None:
        """Execute a single ingestion source."""
        logger.info("scheduler.running_source", source=source)
        now = datetime.now(timezone.utc).isoformat()

        if source == "news":
            from oracle.ingestion.news_pipeline import NewsPipeline
            pipeline = NewsPipeline(
                neo4j=self._neo4j,
                qdrant=self._qdrant,
                entity_resolver=self._entity_resolver,
            )
            stats = await pipeline.ingest()
            self._doc_counts[source] = self._doc_counts.get(source, 0) + stats.get("chunks_stored", 0)

        elif source == "reddit":
            from oracle.ingestion.reddit_client import RedditClient
            client = RedditClient(qdrant=self._qdrant, entity_resolver=self._entity_resolver)
            posts = await client.fetch_all_subreddits()
            stats = await client.ingest_posts(posts)
            self._doc_counts[source] = self._doc_counts.get(source, 0) + stats.get("posts_ingested", 0)

        elif source == "gov_scrapers":
            from oracle.ingestion.gov_scrapers import GovScraper
            scraper = GovScraper(
                neo4j=self._neo4j, qdrant=self._qdrant, entity_resolver=self._entity_resolver,
            )
            stats = await scraper.ingest_all()
            total = stats.get("bills", 0) + stats.get("sec_filings", 0) + stats.get("court_opinions", 0)
            self._doc_counts[source] = self._doc_counts.get(source, 0) + total

        elif source == "polling":
            from oracle.ingestion.polling_scrapers import PollingScraper
            scraper = PollingScraper(qdrant=self._qdrant)
            stats = await scraper.ingest_all()
            self._doc_counts[source] = self._doc_counts.get(source, 0) + stats.get("polls_fetched", 0)

        elif source == "audio":
            from oracle.ingestion.audio_ingestion import AudioIngestionPipeline
            pipeline = AudioIngestionPipeline(qdrant=self._qdrant)
            # Audio sources are processed individually; for scheduled runs we skip
            # since they require specific URLs. Placeholder for when sources are configured.
            self._doc_counts.setdefault(source, 0)

        self._last_run[source] = now
        self._errors.pop(source, None)
        logger.info("scheduler.source_complete", source=source)

    async def _run_twitter_stream(self) -> None:
        """Run continuous Twitter streaming (when API key available)."""
        if not settings.twitter_bearer_token:
            logger.warning("scheduler.twitter_skipped", msg="No Twitter bearer token")
            self._last_run["twitter"] = "skipped (no API key)"
            return

        from oracle.ingestion.twitter_client import TwitterClient
        client = TwitterClient(
            neo4j=self._neo4j, qdrant=self._qdrant, entity_resolver=self._entity_resolver,
        )

        # Get keywords from active markets
        from oracle.ingestion.polymarket_client import PolymarketClient
        pm = PolymarketClient(neo4j=self._neo4j) if self._neo4j else None
        keywords = ["election", "fed", "inflation", "trump", "bitcoin"]

        if pm:
            try:
                markets = await pm.fetch_active_markets(limit=50)
                from oracle.ingestion.twitter_client import extract_keywords_from_markets
                market_keywords = extract_keywords_from_markets(markets)
                if market_keywords:
                    keywords = market_keywords
            except Exception:
                pass

        batch: list[dict[str, Any]] = []
        try:
            async for tweet in client.stream_filtered(keywords):
                if not self._running:
                    break
                batch.append(tweet)
                if len(batch) >= 50:
                    await client.ingest_tweets(batch)
                    self._doc_counts["twitter"] = self._doc_counts.get("twitter", 0) + len(batch)
                    self._last_run["twitter"] = datetime.now(timezone.utc).isoformat()
                    batch = []
        except Exception as e:
            self._errors["twitter"] = str(e)
            logger.error("scheduler.twitter_error", error=str(e))

    async def run_once(self, source: str) -> dict[str, Any]:
        """Run a single source immediately (for API triggers)."""
        try:
            await self._run_source(source)
            return {"status": "completed", "source": source, "last_run": self._last_run.get(source, "")}
        except Exception as e:
            return {"status": "error", "source": source, "error": str(e)}
