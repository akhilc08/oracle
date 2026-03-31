"""News ingestion pipeline: NewsAPI fetch → semantic chunk → embed → Qdrant + entity extraction → Neo4j."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from oracle.config import settings
from oracle.ingestion.chunker import Chunk, semantic_chunk
from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.embeddings import EmbeddingService
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()


class NewsPipeline:
    """End-to-end news ingestion pipeline.

    Flow: NewsAPI → deduplicate → semantic chunk → embed → Qdrant
                                                       → entity extraction → Neo4j
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        qdrant: QdrantManager,
        entity_resolver: EntityResolver,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._entity_resolver = entity_resolver
        self._embedder = embedder or EmbeddingService.get_instance()
        self._seen_urls: set[str] = set()

    async def fetch_news(
        self,
        query: str = "politics OR economy OR markets OR Supreme Court OR elections",
        page_size: int = 100,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Fetch articles from NewsAPI."""
        if not settings.newsapi_key:
            logger.warning("news.no_api_key", msg="NewsAPI key not configured")
            return []

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": page_size,
                    "language": language,
                    "sortBy": "publishedAt",
                    "apiKey": settings.newsapi_key,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        logger.info("news.fetched", count=len(articles), query=query)
        return articles

    def _deduplicate(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate articles by URL."""
        unique = []
        for article in articles:
            url = article.get("url", "")
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                unique.append(article)
        return unique

    def _article_to_text(self, article: dict[str, Any]) -> str:
        """Extract full text from article (title + description + content)."""
        parts = []
        if article.get("title"):
            parts.append(article["title"])
        if article.get("description"):
            parts.append(article["description"])
        if article.get("content"):
            parts.append(article["content"])
        return "\n\n".join(parts)

    async def ingest(
        self,
        query: str = "politics OR economy OR markets OR Supreme Court OR elections",
    ) -> dict[str, int]:
        """Run full ingestion cycle.

        Returns stats: articles_fetched, chunks_stored, entities_extracted.
        """
        # 1. Fetch
        articles = await self.fetch_news(query=query)
        articles = self._deduplicate(articles)

        if not articles:
            return {"articles_fetched": 0, "chunks_stored": 0, "entities_extracted": 0}

        stats = {"articles_fetched": len(articles), "chunks_stored": 0, "entities_extracted": 0}

        # Process in batches
        all_chunks: list[tuple[Chunk, dict[str, Any]]] = []

        for article in articles:
            text = self._article_to_text(article)
            if not text or len(text) < 100:
                continue

            # 2. Semantic chunk
            chunks = semantic_chunk(
                text=text,
                embed_fn=self._embedder.embed,
                min_tokens=200,
                max_tokens=500,
            )

            pub_date = article.get("publishedAt", datetime.now(timezone.utc).isoformat())
            source_name = article.get("source", {}).get("name", "unknown")
            url = article.get("url", "")

            for chunk in chunks:
                chunk.metadata.update({
                    "source_url": url,
                    "publication_date": pub_date,
                    "author": article.get("author", "unknown"),
                    "source_name": source_name,
                    "source_authority_score": self._score_source(source_name),
                    "article_title": article.get("title", ""),
                })
                all_chunks.append((chunk, article))

        if not all_chunks:
            return stats

        # 3. Embed all chunks in batch
        chunk_texts = [c.text for c, _ in all_chunks]
        embeddings = self._embedder.embed(chunk_texts)

        # 4. Store in Qdrant
        ids = []
        vectors = []
        payloads = []

        for i, ((chunk, _article), embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk_id = hashlib.md5(
                f"{chunk.metadata.get('source_url', '')}:{chunk.index}".encode()
            ).hexdigest()

            ids.append(chunk_id)
            vectors.append(embedding)
            payloads.append({
                "text": chunk.text,
                "entity_ids": [],
                "market_ids": [],
                **chunk.metadata,
            })

        await self._qdrant.upsert_chunks(
            collection="news_articles",
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )
        stats["chunks_stored"] = len(ids)

        # 5. Entity extraction → Neo4j
        for chunk, article in all_chunks:
            entities = await self._entity_resolver.extract_and_resolve(chunk.text)
            stats["entities_extracted"] += len(entities)

            for entity in entities:
                await self._neo4j.merge_entity(entity["label"], entity["properties"])

        logger.info(
            "news.ingestion_complete",
            articles=stats["articles_fetched"],
            chunks=stats["chunks_stored"],
            entities=stats["entities_extracted"],
        )
        return stats

    @staticmethod
    def _score_source(source_name: str) -> float:
        """Assign authority score to news sources (0-1)."""
        tier1 = {"reuters", "associated press", "ap news", "bbc news", "the new york times",
                 "the washington post", "the wall street journal", "financial times", "bloomberg"}
        tier2 = {"cnn", "nbc news", "abc news", "cbs news", "the guardian", "politico",
                 "the hill", "axios", "npr"}

        name_lower = source_name.lower()
        if name_lower in tier1:
            return 0.95
        if name_lower in tier2:
            return 0.80
        return 0.50
