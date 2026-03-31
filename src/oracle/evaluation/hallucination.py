"""Hallucination detection — verify claims against sources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from oracle.config import settings

logger = structlog.get_logger()

SIMILARITY_THRESHOLD = 0.75

EXTRACT_CLAIMS_PROMPT = """Extract all factual claims from the following text. Return ONLY a JSON array of strings, each a single atomic claim.

TEXT:
{text}

Example output: ["Claim 1", "Claim 2", "Claim 3"]
"""

CONTRADICTION_PROMPT = """Analyze these claims for logical contradictions. Return ONLY valid JSON.

CLAIMS:
{claims}

Return: {{"contradictions": [["claim_a", "claim_b", "explanation"], ...], "has_contradictions": true/false}}
If no contradictions, return: {{"contradictions": [], "has_contradictions": false}}
"""


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""

    claim: str
    verified: bool = False
    confidence: float = 0.0
    matching_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "verified": self.verified,
            "confidence": self.confidence,
            "matching_source": self.matching_source,
        }


@dataclass
class HallucinationResult:
    """Result of hallucination detection on a thesis."""

    grounded_claims: list[ClaimVerification] = field(default_factory=list)
    ungrounded_claims: list[ClaimVerification] = field(default_factory=list)
    hallucination_rate: float = 0.0
    has_self_contradictions: bool = False
    contradictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded_claims": [c.to_dict() for c in self.grounded_claims],
            "ungrounded_claims": [c.to_dict() for c in self.ungrounded_claims],
            "hallucination_rate": self.hallucination_rate,
            "has_self_contradictions": self.has_self_contradictions,
            "contradictions": self.contradictions,
        }


class HallucinationDetector:
    """Detects ungrounded claims and self-contradictions in trade theses."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        self._embedding_service = None

    def _get_embedding_service(self):
        """Lazy-load embedding service."""
        if self._embedding_service is None:
            from oracle.knowledge.embeddings import EmbeddingService
            self._embedding_service = EmbeddingService.get_instance()
        return self._embedding_service

    async def extract_claims(self, text: str) -> list[str]:
        """Extract atomic factual claims from text using Claude API."""
        if self._api_key:
            try:
                return await self._extract_claims_claude(text)
            except Exception as e:
                logger.warning("hallucination.extract_failed", error=str(e))

        return self._extract_claims_heuristic(text)

    async def _extract_claims_claude(self, text: str) -> list[str]:
        """Use Claude to extract claims."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        prompt = EXTRACT_CLAIMS_PROMPT.format(text=text)

        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text_resp = response.content[0].text.strip()
        if text_resp.startswith("```"):
            text_resp = text_resp.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text_resp)

    def _extract_claims_heuristic(self, text: str) -> list[str]:
        """Simple sentence-based claim extraction as fallback."""
        sentences = text.replace(".\n", ". ").split(". ")
        claims = []
        for s in sentences:
            s = s.strip().rstrip(".")
            if len(s.split()) >= 4:  # Skip very short fragments
                claims.append(s)
        return claims

    def verify_claim(self, claim: str, sources: list[str]) -> ClaimVerification:
        """Verify a single claim against sources using cosine similarity on embeddings."""
        if not sources:
            return ClaimVerification(claim=claim, verified=False, confidence=0.0)

        embedding_service = self._get_embedding_service()
        claim_embedding = np.array(embedding_service.embed([claim])[0])
        source_embeddings = np.array(embedding_service.embed(sources))

        # Cosine similarity (embeddings are already normalized)
        similarities = source_embeddings @ claim_embedding

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return ClaimVerification(
            claim=claim,
            verified=best_score >= SIMILARITY_THRESHOLD,
            confidence=round(best_score, 4),
            matching_source=sources[best_idx] if best_score >= SIMILARITY_THRESHOLD else None,
        )

    async def check_self_contradictions(self, claims: list[str]) -> dict[str, Any]:
        """Check claim pairs for logical contradictions."""
        if len(claims) < 2:
            return {"contradictions": [], "has_contradictions": False}

        if self._api_key:
            try:
                return await self._check_contradictions_claude(claims)
            except Exception as e:
                logger.warning("hallucination.contradiction_check_failed", error=str(e))

        return {"contradictions": [], "has_contradictions": False}

    async def _check_contradictions_claude(self, claims: list[str]) -> dict[str, Any]:
        """Use Claude to detect contradictions between claims."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CONTRADICTION_PROMPT.format(claims=claims_text)

        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    async def detect(self, thesis_text: str, sources: list[str]) -> HallucinationResult:
        """Run full hallucination detection pipeline.

        Args:
            thesis_text: The trade thesis to check.
            sources: Source texts to verify claims against.

        Returns:
            HallucinationResult with grounded/ungrounded claims and hallucination rate.
        """
        claims = await self.extract_claims(thesis_text)

        if not claims:
            return HallucinationResult()

        grounded: list[ClaimVerification] = []
        ungrounded: list[ClaimVerification] = []

        for claim in claims:
            verification = self.verify_claim(claim, sources)
            if verification.verified:
                grounded.append(verification)
            else:
                ungrounded.append(verification)

        total = len(grounded) + len(ungrounded)
        hallucination_rate = len(ungrounded) / total if total > 0 else 0.0

        # Check for self-contradictions
        contradiction_result = await self.check_self_contradictions(claims)

        return HallucinationResult(
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            hallucination_rate=round(hallucination_rate, 4),
            has_self_contradictions=contradiction_result.get("has_contradictions", False),
            contradictions=contradiction_result.get("contradictions", []),
        )
