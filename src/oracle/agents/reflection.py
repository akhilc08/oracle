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

    # Try Claude CLI
    try:
        import asyncio
        prompt = REFLECTION_PROMPT.format(
            question=question,
            thesis=thesis,
            confidence=confidence,
            momentum=momentum,
            evidence_count=evidence_count,
        )
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode == 0:
            return _parse_reflection_response(stdout.decode().strip(), confidence)
        logger.warning("reflection.claude_failed", error=stderr.decode()[:200])
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
    """Heuristic bias detection when Claude CLI is unavailable."""
    biases: list[str] = []
    adjusted = confidence

    # Overconfidence: high confidence with very little evidence
    if confidence > 0.85 and evidence_count < 2:
        biases.append("overconfidence")
        adjusted = min(adjusted, 0.80)

    # Momentum alignment: boost confidence when market price confirms thesis direction
    if momentum > 0.2 and confidence > 0.5:
        adjusted = min(0.95, adjusted + momentum * 0.05)
    elif momentum < -0.2 and confidence < 0.5:
        adjusted = max(0.05, adjusted + momentum * 0.05)

    # Anchoring check: flag if confidence is very close to market-implied probability
    market_implied = (momentum + 1) / 2
    if abs(confidence - market_implied) < 0.03:
        biases.append("anchoring bias")

    reasoning = (
        f"Heuristic analysis: {len(biases)} potential biases detected. "
        f"Evidence count: {evidence_count}, momentum: {momentum:.2f}. "
        f"Confidence {'adjusted to ' + str(round(adjusted, 2)) if adjusted != confidence else 'unchanged'}."
    )

    return ReflectionResult(
        biases_detected=biases,
        adjusted_confidence=round(adjusted, 3),
        reasoning=reasoning,
    )
