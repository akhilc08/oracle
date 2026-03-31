"""Reflection step — post-prediction self-critique for bias detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from oracle.config import settings

logger = structlog.get_logger()


@dataclass
class ReflectionResult:
    """Result of the reflection/self-critique step."""

    biases_detected: list[str] = field(default_factory=list)
    adjusted_confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "biases_detected": self.biases_detected,
            "adjusted_confidence": self.adjusted_confidence,
            "reasoning": self.reasoning,
        }


REFLECTION_PROMPT = """You are a prediction market analyst performing a self-critique on a trade decision.

Market Question: {question}
Research Thesis: {thesis}
Research Confidence: {confidence}
Quantitative Momentum: {momentum}
Evidence Count: {evidence_count}

Check for these cognitive biases:
1. **Anchoring bias**: Is the confidence overly influenced by the current market price?
2. **Recency bias**: Is recent news being given disproportionate weight over base rates?
3. **Confirmation bias**: Is evidence being selectively interpreted to support the thesis?
4. **Overconfidence**: Is the confidence score justified by the quality/quantity of evidence?

Respond in this exact format:
BIASES: <comma-separated list of detected biases, or "none">
ADJUSTED_CONFIDENCE: <float 0.0-1.0>
REASONING: <2-3 sentences explaining your assessment>
"""


async def reflect(
    question: str,
    thesis: str,
    confidence: float,
    momentum: float,
    evidence_count: int,
) -> ReflectionResult:
    """Run the reflection step via Claude API.

    Falls back to heuristic analysis if API is unavailable.
    """
    result = ReflectionResult(adjusted_confidence=confidence)

    # Try Claude API first
    if settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            prompt = REFLECTION_PROMPT.format(
                question=question,
                thesis=thesis,
                confidence=confidence,
                momentum=momentum,
                evidence_count=evidence_count,
            )
            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return _parse_reflection_response(text, confidence)
        except Exception as e:
            logger.warning("reflection.claude_failed", error=str(e))

    # Heuristic fallback
    return _heuristic_reflection(confidence, momentum, evidence_count)


def _parse_reflection_response(text: str, original_confidence: float) -> ReflectionResult:
    """Parse the structured reflection response from Claude."""
    result = ReflectionResult(adjusted_confidence=original_confidence)

    if "BIASES:" in text:
        biases_str = text.split("BIASES:")[1].split("ADJUSTED_CONFIDENCE:")[0].strip()
        if biases_str.lower() != "none":
            result.biases_detected = [b.strip() for b in biases_str.split(",") if b.strip()]

    if "ADJUSTED_CONFIDENCE:" in text:
        try:
            conf_str = text.split("ADJUSTED_CONFIDENCE:")[1].split("REASONING:")[0].strip()
            result.adjusted_confidence = float(conf_str.split()[0])
        except (ValueError, IndexError):
            pass

    if "REASONING:" in text:
        result.reasoning = text.split("REASONING:")[1].strip()

    return result


def _heuristic_reflection(
    confidence: float,
    momentum: float,
    evidence_count: int,
) -> ReflectionResult:
    """Heuristic bias detection when Claude API is unavailable."""
    biases: list[str] = []
    adjusted = confidence

    # Overconfidence check: high confidence with little evidence
    if confidence > 0.8 and evidence_count < 3:
        biases.append("overconfidence")
        adjusted = min(adjusted, 0.75)

    # Anchoring check: confidence very close to market-implied probability
    market_implied = (momentum + 1) / 2  # Convert [-1,1] momentum to [0,1]
    if abs(confidence - market_implied) < 0.05:
        biases.append("anchoring bias")
        # No adjustment — just flag it

    # Recency bias: strong momentum + high confidence suggests chasing
    if abs(momentum) > 0.6 and confidence > 0.7:
        biases.append("recency bias")
        adjusted = min(adjusted, confidence - 0.05)

    reasoning = (
        f"Heuristic analysis: {len(biases)} potential biases detected. "
        f"Evidence count: {evidence_count}, momentum: {momentum:.2f}. "
        f"{'Confidence adjusted downward.' if adjusted < confidence else 'No confidence adjustment needed.'}"
    )

    return ReflectionResult(
        biases_detected=biases,
        adjusted_confidence=round(adjusted, 3),
        reasoning=reasoning,
    )
