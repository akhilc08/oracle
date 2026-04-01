"""Reddit API integration — monitor subreddits for market-relevant signals."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from oracle.config import settings
from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()

REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"

DEFAULT_SUBREDDITS = [
    "politics",
    "Economics",
    "investing",
    "wallstreetbets",
    "geopolitics",
    "worldnews",
    "Crypto_Currency_News",
]


class RedditClient:
    """Client for Reddit API — fetch hot posts and comments from relevant subreddits."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        entity_resolver: EntityResolver | None = None,
        subreddits: list[str] | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._entity_resolver = entity_resolver or EntityResolver()
        self._subreddits = subreddits or DEFAULT_SUBREDDITS
        self._access_token: str | None = None
        self._token_expires: float = 0
        self._seen_post_ids: set[str] = set()
        self._last_fetched: dict[str, float] = {}

    async def _authenticate(self) -> str:
        """Obtain OAuth2 access token via client_credentials flow."""
        now = datetime.now(timezone.utc).timestamp()
        if self._access_token and now < self._token_expires:
            return self._access_token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                REDDIT_AUTH_URL,
                auth=(settings.reddit_client_id, settings.reddit_client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "Oracle/1.0 (prediction engine)"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

        self._access_token = data["access_token"]
        self._token_expires = now + data.get("expires_in", 3600) - 60
        logger.info("reddit.authenticated")
        return self._access_token

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Oracle/1.0 (prediction engine)",
        }

    async def fetch_hot(
        self, subreddit: str, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Fetch hot posts from a subreddit."""
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            logger.warning("reddit.no_credentials", msg="Reddit API credentials not configured")
            return []

        token = await self._authenticate()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{REDDIT_API_BASE}/r/{subreddit}/hot",
                headers=self._headers(token),
                params={"limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            post_id = post_data.get("id", "")

            if post_id in self._seen_post_ids:
                continue

            posts.append(self._process_post(post_data, subreddit))
            self._seen_post_ids.add(post_id)

        self._last_fetched[subreddit] = datetime.now(timezone.utc).timestamp()
        logger.info("reddit.fetched_hot", subreddit=subreddit, count=len(posts))
        return posts

    async def fetch_comments(
        self, post_id: str, subreddit: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch top-level comments for a post."""
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            return []

        token = await self._authenticate()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{REDDIT_API_BASE}/r/{subreddit}/comments/{post_id}",
                headers=self._headers(token),
                params={"limit": limit, "depth": 1},
            )
            resp.raise_for_status()
            data = resp.json()

        comments = []
        if len(data) > 1:
            for child in data[1].get("data", {}).get("children", []):
                comment_data = child.get("data", {})
                body = comment_data.get("body", "")
                if body and body != "[deleted]" and body != "[removed]":
                    comments.append({
                        "id": comment_data.get("id", ""),
                        "text": body,
                        "score": comment_data.get("score", 0),
                        "created_at": datetime.fromtimestamp(
                            comment_data.get("created_utc", 0), tz=timezone.utc
                        ).isoformat(),
                    })

        return comments[:limit]

    def _process_post(self, post_data: dict[str, Any], subreddit: str) -> dict[str, Any]:
        """Convert Reddit API post to our standard model."""
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        text = f"{title}\n\n{selftext}".strip() if selftext else title

        return {
            "id": post_data.get("id", ""),
            "subreddit": subreddit,
            "title": title,
            "text": text,
            "score": post_data.get("score", 0),
            "created_at": datetime.fromtimestamp(
                post_data.get("created_utc", 0), tz=timezone.utc
            ).isoformat(),
            "entities": [],
            "market_ids": [],
            "source": "reddit",
        }

    async def fetch_all_subreddits(self) -> list[dict[str, Any]]:
        """Fetch hot posts from all monitored subreddits."""
        all_posts: list[dict[str, Any]] = []
        for subreddit in self._subreddits:
            posts = await self.fetch_hot(subreddit)
            all_posts.extend(posts)
        return all_posts

    async def ingest_posts(self, posts: list[dict[str, Any]]) -> dict[str, int]:
        """Ingest processed posts into Qdrant."""
        if not posts:
            return {"posts_ingested": 0, "entities_extracted": 0}

        stats = {"posts_ingested": 0, "entities_extracted": 0}

        # Entity extraction
        for post in posts:
            entities = await self._entity_resolver.extract_and_resolve(post["text"])
            post["entities"] = [e["properties"]["name"] for e in entities]
            stats["entities_extracted"] += len(entities)

        # Ingest to Qdrant
        if self._qdrant:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            texts = [p["text"] for p in posts if p["text"]]
            if texts:
                embeddings = embedder.embed(texts)
                ids = []
                vectors = []
                payloads = []

                for post, embedding in zip(posts, embeddings):
                    chunk_id = hashlib.md5(f"reddit:{post['id']}".encode()).hexdigest()
                    ids.append(chunk_id)
                    vectors.append(embedding)
                    payloads.append({
                        "text": post["text"],
                        "source_url": f"https://reddit.com/r/{post['subreddit']}/comments/{post['id']}",
                        "publication_date": post["created_at"],
                        "author": f"r/{post['subreddit']}",
                        "source_authority_score": 0.3,
                        "entity_ids": post["entities"],
                        "market_ids": post["market_ids"],
                        "source": "reddit",
                        "subreddit": post["subreddit"],
                        "score": post["score"],
                    })

                await self._qdrant.upsert_chunks(
                    collection="social_media",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                )
                stats["posts_ingested"] = len(ids)

        logger.info("reddit.ingestion_complete", **stats)
        return stats
