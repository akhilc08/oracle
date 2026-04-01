"""Twitter/X API v2 integration — stream and search tweets for market-relevant signals."""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx
import structlog

from oracle.config import settings
from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()

TWITTER_API_BASE = "https://api.twitter.com/2"

POSITIVE_WORDS = frozenset({
    "bullish", "surge", "rally", "gain", "up", "rise", "positive", "optimistic",
    "win", "boom", "soar", "strong", "growth", "breakout", "moon", "pump",
    "approval", "pass", "support", "victory", "success", "beat", "exceed",
})

NEGATIVE_WORDS = frozenset({
    "bearish", "crash", "drop", "fall", "down", "decline", "negative", "pessimistic",
    "loss", "bust", "plunge", "weak", "recession", "dump", "reject", "fail",
    "disapproval", "oppose", "defeat", "miss", "concern", "risk", "crisis",
})


def detect_sentiment(text: str) -> str:
    """Simple keyword-based sentiment detection."""
    words = set(re.findall(r"\w+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def clean_tweet_text(text: str) -> str:
    """Remove URLs, mentions prefix, and excessive whitespace."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords_from_markets(markets: list[dict[str, Any]]) -> list[str]:
    """Auto-generate Twitter filter keywords from active Polymarket markets."""
    keywords: set[str] = set()
    stop_words = {"the", "will", "be", "is", "a", "an", "of", "in", "to", "for", "on", "by", "or", "and", "at"}
    for market in markets:
        question = market.get("question", "") or market.get("title", "")
        # Extract capitalized words and significant terms
        words = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
        keywords.update(w for w in words if w.lower() not in stop_words)
        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', question)
        keywords.update(quoted)
    return sorted(keywords)[:50]  # Twitter filter rules have limits


class TwitterClient:
    """Client for Twitter API v2 — search and stream tweets for prediction signals."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        qdrant: QdrantManager | None = None,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._entity_resolver = entity_resolver or EntityResolver()
        self._request_count = 0
        self._rate_limit_reset: float = 0
        self._backoff_seconds = 1.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {settings.twitter_bearer_token}"}

    async def _request_with_backoff(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a request with exponential backoff on 429 rate limits."""
        max_retries = 5
        backoff = self._backoff_seconds

        for attempt in range(max_retries):
            resp = await client.request(method, url, **kwargs)
            self._request_count += 1

            if resp.status_code == 429:
                reset = resp.headers.get("x-rate-limit-reset")
                if reset:
                    wait = max(float(reset) - datetime.now(timezone.utc).timestamp(), 1.0)
                else:
                    wait = backoff
                logger.warning("twitter.rate_limited", wait=wait, attempt=attempt + 1)
                await asyncio.sleep(min(wait, 300))  # Cap at 5 min
                backoff *= 2
                continue

            resp.raise_for_status()
            self._backoff_seconds = 1.0  # Reset on success
            return resp

        raise httpx.HTTPStatusError(
            "Rate limit exceeded after max retries",
            request=httpx.Request(method, url),
            response=resp,
        )

    async def search_recent(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search tweets from the last 7 days."""
        if not settings.twitter_bearer_token:
            logger.warning("twitter.no_bearer_token", msg="TWITTER_BEARER_TOKEN not configured")
            return []

        # Twitter API max per request is 100
        max_results = min(max_results, 100)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await self._request_with_backoff(
                client,
                "GET",
                f"{TWITTER_API_BASE}/tweets/search/recent",
                headers=self._headers(),
                params={
                    "query": query,
                    "max_results": max_results,
                    "tweet.fields": "created_at,author_id,text,public_metrics",
                },
            )
            data = resp.json()

        tweets_raw = data.get("data", [])
        tweets = [self._process_tweet(t) for t in tweets_raw]
        logger.info("twitter.search_complete", query=query, count=len(tweets))
        return tweets

    async def stream_filtered(
        self,
        keywords: list[str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields tweets matching filtered stream rules.

        Sets up rules based on keywords, then connects to the filtered stream.
        """
        if not settings.twitter_bearer_token:
            logger.warning("twitter.no_bearer_token", msg="TWITTER_BEARER_TOKEN not configured")
            return

        async with httpx.AsyncClient(timeout=None) as client:
            # Set stream rules
            rules = [{"value": kw, "tag": kw} for kw in keywords[:25]]  # Max 25 rules
            await self._request_with_backoff(
                client,
                "POST",
                f"{TWITTER_API_BASE}/tweets/search/stream/rules",
                headers=self._headers(),
                json={"add": rules},
            )

            # Connect to stream
            async with client.stream(
                "GET",
                f"{TWITTER_API_BASE}/tweets/search/stream",
                headers=self._headers(),
                params={"tweet.fields": "created_at,author_id,text"},
            ) as stream:
                async for line in stream.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        tweet_data = data.get("data", {})
                        if tweet_data:
                            yield self._process_tweet(tweet_data)
                    except Exception:
                        logger.debug("twitter.stream_parse_error", line=line[:100])
                        continue

    def _process_tweet(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Process a raw tweet into our standard model."""
        text = clean_tweet_text(raw.get("text", ""))
        return {
            "id": raw.get("id", ""),
            "text": text,
            "author_id": raw.get("author_id", ""),
            "created_at": raw.get("created_at", datetime.now(timezone.utc).isoformat()),
            "entities": [],  # Populated during ingestion
            "sentiment": detect_sentiment(text),
            "market_ids": [],
            "source": "twitter",
        }

    async def ingest_tweets(self, tweets: list[dict[str, Any]]) -> dict[str, int]:
        """Ingest processed tweets into Qdrant + Neo4j."""
        if not tweets:
            return {"tweets_ingested": 0, "entities_extracted": 0}

        stats = {"tweets_ingested": 0, "entities_extracted": 0}

        # Entity extraction
        for tweet in tweets:
            entities = await self._entity_resolver.extract_and_resolve(tweet["text"])
            tweet["entities"] = [e["properties"]["name"] for e in entities]

            # Merge entities to Neo4j
            if self._neo4j:
                for entity in entities:
                    await self._neo4j.merge_entity(entity["label"], entity["properties"])
                    stats["entities_extracted"] += 1

        # Ingest to Qdrant
        if self._qdrant:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            texts = [t["text"] for t in tweets if t["text"]]
            if texts:
                embeddings = embedder.embed(texts)
                ids = []
                vectors = []
                payloads = []

                for tweet, embedding in zip(tweets, embeddings):
                    chunk_id = hashlib.md5(f"twitter:{tweet['id']}".encode()).hexdigest()
                    ids.append(chunk_id)
                    vectors.append(embedding)
                    payloads.append({
                        "text": tweet["text"],
                        "source_url": f"https://twitter.com/i/status/{tweet['id']}",
                        "publication_date": tweet["created_at"],
                        "author": tweet["author_id"],
                        "source_authority_score": 0.4,
                        "entity_ids": tweet["entities"],
                        "market_ids": tweet["market_ids"],
                        "sentiment": tweet["sentiment"],
                        "source": "twitter",
                    })

                await self._qdrant.upsert_chunks(
                    collection="social_media",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                )
                stats["tweets_ingested"] = len(ids)

        logger.info("twitter.ingestion_complete", **stats)
        return stats
