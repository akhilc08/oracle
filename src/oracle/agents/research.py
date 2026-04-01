"""Research Agent — monitors markets and generates structured research reports."""

from __future__ import annotations

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
class ResearchReport:
    """Structured research report for a market."""

    market_id: str
    question: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    thesis: str = ""
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "evidence": self.evidence,
            "key_entities": self.key_entities,
            "thesis": self.thesis,
            "confidence": self.confidence,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat(),
        }


# --- Tool functions (cacheable) ---


@cached_tool(ttl=600)
async def search_knowledge_base(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search the hybrid retrieval engine for relevant evidence."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/api/v1/retrieval/search",
                json={"query": query, "top_k": top_k},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
    except Exception as e:
        logger.warning("tool.search_knowledge_base.error", error=str(e))
        return []


@cached_tool(ttl=300)
async def query_graph(cypher: str) -> list[dict[str, Any]]:
    """Execute a Cypher query against the Neo4j knowledge graph."""
    try:
        from oracle.knowledge.neo4j_client import Neo4jClient

        client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        try:
            result = await client.run_query(cypher)
            return result
        finally:
            await client.close()
    except Exception as e:
        logger.warning("tool.query_graph.error", error=str(e))
        return []


@cached_tool(ttl=60)
async def get_market_data(market_id: str) -> dict[str, Any]:
    """Fetch current market data from Polymarket Gamma API."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{settings.polymarket_api_base}/markets/{market_id}",
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("tool.get_market_data.error", market_id=market_id, error=str(e))
        return {}


@cached_tool(ttl=900)
async def fetch_latest_news(query: str, page_size: int = 5) -> list[dict[str, Any]]:
    """Fetch latest news articles from NewsAPI."""
    if not settings.newsapi_key:
        return []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": page_size,
                    "sortBy": "publishedAt",
                    "apiKey": settings.newsapi_key,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json().get("articles", [])
    except Exception as e:
        logger.warning("tool.fetch_latest_news.error", error=str(e))
        return []


async def _call_claude(prompt: str) -> str:
    """Call Claude via the local Claude Code CLI. Falls back to empty string on failure."""
    try:
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            logger.warning("research.claude_call_failed", error=stderr.decode()[:200])
            return ""
        return stdout.decode().strip()
    except Exception as e:
        logger.warning("research.claude_call_failed", error=str(e))
        return ""


def _relevance_score(question: str, text: str) -> float:
    """Score how relevant a piece of evidence text is to the question (0.0-1.0)."""
    q_words = set(question.lower().split()) - {"a", "an", "the", "is", "will", "does", "to", "of", "in", "on", "at", "?"}
    t_words = set(text.lower().split())
    if not q_words:
        return 0.3
    overlap = len(q_words & t_words) / len(q_words)
    return min(1.0, overlap * 2)  # scale: 50% word overlap → score=1.0


def _simple_synthesis(question: str, evidence: list[dict[str, Any]]) -> tuple[str, float]:
    """Local stub for simple synthesis when Claude API is unavailable."""
    if not evidence:
        return "Insufficient evidence to form thesis.", 0.3

    # Re-score news items by relevance to the question
    scored = []
    for e in evidence:
        if e.get("type") == "news":
            score = _relevance_score(question, e.get("text", ""))
        else:
            score = e.get("score", 0.5)
        scored.append(score)

    avg_score = sum(scored) / len(scored)
    relevant_count = sum(1 for s in scored if s > 0.1)

    # No relevant evidence → abstain at 0.5 rather than bias toward NO
    if relevant_count == 0 or avg_score < 0.05:
        confidence = 0.5
    else:
        confidence = min(0.85, 0.45 + (relevant_count * 0.04) + (avg_score * 0.3))

    snippets = [e.get("text", "")[:100] for e in evidence[:3]]
    thesis = f"Based on {relevant_count} relevant evidence items: {'; '.join(snippets)}"
    return thesis, confidence


class ResearchAgent(BaseAgent):
    """Monitors Polymarket and generates research reports.

    Monitors for:
    - New market listings
    - Markets with >5% price move in 24h
    - Markets approaching resolution (<72h)

    Model routing:
    - Simple lookups → local stub
    - Complex synthesis → Claude API (claude-3-5-haiku)
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(agent_id="research", name="Research Agent", bus=bus)
        self.register_tool("search_knowledge_base", search_knowledge_base)
        self.register_tool("query_graph", query_graph)
        self.register_tool("get_market_data", get_market_data)
        self.register_tool("fetch_latest_news", fetch_latest_news)

    async def handle_message(self, message: Message) -> None:
        """Handle RESEARCH_REQUEST messages."""
        if message.type != MessageType.RESEARCH_REQUEST:
            return

        market_id = message.payload.get("market_id", "")
        question = message.payload.get("question", "")
        trace_id = message.trace_id

        logger.info(
            "research.start",
            market_id=market_id,
            question=question[:80],
            trace_id=trace_id,
        )

        report = await self.generate_report(market_id, question, trace_id)

        await self.send(Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type=MessageType.RESEARCH_RESULT,
            payload=report.to_dict(),
            trace_id=trace_id,
        ))

    async def generate_report(
        self, market_id: str, question: str, trace_id: str = ""
    ) -> ResearchReport:
        """Run the full research pipeline for a market."""
        report = ResearchReport(market_id=market_id, question=question)

        # 1. Get market data
        market_data = await get_market_data(market_id=market_id)
        if market_data:
            report.evidence.append({
                "type": "market_data",
                "text": f"Current price: {market_data.get('outcomePrices', 'N/A')}, "
                        f"Volume: {market_data.get('volume', 'N/A')}",
                "score": 1.0,
            })

        # 2. Search knowledge base
        kb_results = await search_knowledge_base(query=question, top_k=5)
        for r in kb_results:
            report.evidence.append({
                "type": "knowledge_base",
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
            })
            for src in r.get("sources", []):
                if src not in report.sources:
                    report.sources.append(src)

        # 3. Fetch latest news
        news = await fetch_latest_news(query=question, page_size=5)
        for article in news:
            report.evidence.append({
                "type": "news",
                "text": article.get("title", "") + ": " + (article.get("description", "") or ""),
                "score": 0.7,
            })
            url = article.get("url")
            if url and url not in report.sources:
                report.sources.append(url)

        # 4. Synthesize thesis — route based on complexity
        use_claude = len(report.evidence) >= 1
        if use_claude:
            evidence_text = "\n".join(
                f"- [{e['type']}] {e['text'][:200]}" for e in report.evidence
            )
            prompt = (
                f"You are a calibrated prediction market analyst. Your job is to estimate "
                f"the probability that the answer to this question is YES.\n\n"
                f"Question: '{question}'\n\nEvidence:\n{evidence_text}\n\n"
                f"Output the probability of YES (0.0=certainly NO, 1.0=certainly YES). "
                f"Be calibrated: 0.8 means you'd be right 80% of the time on questions like this. "
                f"Do not hedge toward 0.5 unless you genuinely have no information. "
                f"Format exactly:\nTHESIS: <2-3 sentence reasoning>\nCONFIDENCE: <float>"
            )
            response = await _call_claude(prompt)
            if response:
                if "THESIS:" in response:
                    report.thesis = response.split("THESIS:")[1].split("CONFIDENCE:")[0].strip()
                else:
                    report.thesis = response[:500]
                if "CONFIDENCE:" in response:
                    try:
                        report.confidence = float(
                            response.split("CONFIDENCE:")[1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        report.confidence = 0.5
                else:
                    report.confidence = 0.5
            else:
                report.thesis, report.confidence = _simple_synthesis(question, report.evidence)
        else:
            report.thesis, report.confidence = _simple_synthesis(question, report.evidence)

        logger.info(
            "research.complete",
            market_id=market_id,
            evidence_count=len(report.evidence),
            confidence=report.confidence,
            trace_id=trace_id,
        )
        return report
