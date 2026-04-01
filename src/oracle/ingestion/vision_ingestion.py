"""Vision model pipeline — analyze charts/images using Claude vision for market signals."""

from __future__ import annotations

import base64
import hashlib
import httpx
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from oracle.config import settings
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()

# Domains likely to contain financial/political charts
RELEVANT_DOMAINS = frozenset({
    "twitter.com", "x.com", "pbs.twimg.com",
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "tradingview.com", "sec.gov",
    "fivethirtyeight.com", "realclearpolitics.com",
    "fred.stlouisfed.org", "bls.gov", "bea.gov",
    "polymarket.com", "metaculus.com",
})


@dataclass
class ChartAnalysis:
    """Structured analysis of a chart/image."""

    chart_type: str = "other"  # price_chart | poll_chart | economic_indicator | other
    key_trend: str = ""
    key_values: dict[str, Any] = field(default_factory=dict)
    market_relevance: list[str] = field(default_factory=list)
    summary: str = ""


def is_relevant_domain(url: str) -> bool:
    """Check if a URL is from a domain likely to contain financial/political content."""
    from urllib.parse import urlparse
    try:
        hostname = urlparse(url).hostname or ""
        return any(domain in hostname for domain in RELEVANT_DOMAINS)
    except Exception:
        return False


class VisionIngestionPipeline:
    """Pipeline for analyzing charts and images using Claude vision API."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._qdrant = qdrant
        self._model = model

    async def analyze_chart(
        self,
        image_url_or_path: str,
    ) -> ChartAnalysis:
        """Analyze a chart image using Claude vision.

        Accepts either a URL or a local file path.
        """
        if not settings.anthropic_api_key:
            logger.warning("vision.no_api_key", msg="ANTHROPIC_API_KEY not configured")
            return ChartAnalysis(summary="API key not configured")

        # Build image content
        image_content = await self._prepare_image(image_url_or_path)
        if not image_content:
            return ChartAnalysis(summary="Could not load image")

        # Call Claude vision API
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this chart/image. Provide a JSON response with these fields:\n"
                                    '- chart_type: "price_chart" | "poll_chart" | "economic_indicator" | "other"\n'
                                    '- key_trend: brief description (e.g., "upward momentum", "sharp decline")\n'
                                    "- key_values: dict of notable values (e.g., current, peak, trough)\n"
                                    "- market_relevance: list of prediction market topics this relates to\n"
                                    "- summary: 2-3 sentence plain English description\n"
                                    "Return ONLY valid JSON, no markdown."
                                ),
                            },
                        ],
                    }
                ],
            )

            return self._parse_response(response.content[0].text)

        except Exception as e:
            logger.error("vision.analysis_error", error=str(e))
            return ChartAnalysis(summary=f"Analysis failed: {str(e)}")

    async def _prepare_image(self, image_url_or_path: str) -> dict[str, Any] | None:
        """Prepare image content for Claude API — either URL or base64-encoded file."""
        if image_url_or_path.startswith(("http://", "https://")):
            # Download and base64 encode
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(image_url_or_path)
                    resp.raise_for_status()

                content_type = resp.headers.get("content-type", "image/png")
                media_type = content_type.split(";")[0].strip()
                if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                    media_type = "image/png"

                b64 = base64.b64encode(resp.content).decode("utf-8")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                }
            except httpx.HTTPError as e:
                logger.warning("vision.download_failed", url=image_url_or_path, error=str(e))
                return None
        else:
            # Local file
            path = Path(image_url_or_path)
            if not path.exists():
                logger.warning("vision.file_not_found", path=image_url_or_path)
                return None

            suffix = path.suffix.lower()
            media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".gif": "image/gif", ".webp": "image/webp"}
            media_type = media_map.get(suffix, "image/png")

            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }

    def _parse_response(self, text: str) -> ChartAnalysis:
        """Parse Claude's JSON response into a ChartAnalysis."""
        import json

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
            return ChartAnalysis(
                chart_type=data.get("chart_type", "other"),
                key_trend=data.get("key_trend", ""),
                key_values=data.get("key_values", {}),
                market_relevance=data.get("market_relevance", []),
                summary=data.get("summary", ""),
            )
        except json.JSONDecodeError:
            logger.warning("vision.parse_error", text=text[:200])
            return ChartAnalysis(summary=text[:500])

    async def process_url(self, url: str) -> ChartAnalysis:
        """Download an image and analyze it."""
        if not is_relevant_domain(url):
            logger.debug("vision.irrelevant_domain", url=url)
            return ChartAnalysis(summary="Domain not relevant for financial analysis")

        analysis = await self.analyze_chart(url)

        # Ingest analysis text into Qdrant
        if self._qdrant and analysis.summary:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            text = (
                f"Chart Analysis ({analysis.chart_type}): {analysis.key_trend}\n"
                f"{analysis.summary}\n"
                f"Key values: {analysis.key_values}"
            )
            embeddings = embedder.embed([text])
            chunk_id = hashlib.md5(f"vision:{url}".encode()).hexdigest()

            await self._qdrant.upsert_chunks(
                collection="news_articles",
                ids=[chunk_id],
                vectors=embeddings,
                payloads=[{
                    "text": text,
                    "source_url": url,
                    "publication_date": "",
                    "author": "vision_pipeline",
                    "source_authority_score": 0.6,
                    "entity_ids": [],
                    "market_ids": analysis.market_relevance,
                    "source": "chart_analysis",
                    "chart_type": analysis.chart_type,
                }],
            )

        return analysis
