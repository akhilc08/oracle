"""LLM-as-judge evaluation — pre-execution quality gates for trade theses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from oracle.config import settings

logger = structlog.get_logger()

EVALUATION_PROMPT = """You are an expert prediction market analyst evaluating a trade thesis.

THESIS:
{thesis}

SUPPORTING SOURCES:
{sources}

Score this thesis on 4 dimensions (0-10 each). Respond ONLY with valid JSON:
{{
  "groundedness": {{"score": <int>, "explanation": "<1-2 sentences>"}},
  "reasoning_quality": {{"score": <int>, "explanation": "<1-2 sentences>"}},
  "evidence_completeness": {{"score": <int>, "explanation": "<1-2 sentences>"}},
  "calibration_alignment": {{"score": <int>, "explanation": "<1-2 sentences>"}}
}}

Scoring guide:
- **Groundedness** (0-10): Every claim must be supported by a retrieved source. 10 = all claims traceable.
- **Reasoning quality** (0-10): Logical chain coherence. 10 = airtight reasoning with no gaps.
- **Evidence completeness** (0-10): Sufficient sources, counter-arguments addressed. 10 = comprehensive.
- **Calibration alignment** (0-10): Confidence level appropriate given evidence strength. 10 = perfectly calibrated.
"""

# Thresholds for passing
GROUNDEDNESS_THRESHOLD = 7
REASONING_THRESHOLD = 6
EVIDENCE_THRESHOLD = 5


@dataclass
class EvaluationResult:
    """Result of LLM-as-judge evaluation."""

    passed: bool = False
    scores: dict[str, int] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)
    overall_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scores": self.scores,
            "explanations": self.explanations,
            "overall_quality": self.overall_quality,
        }


class EvaluationJudge:
    """LLM-as-judge for evaluating trade thesis quality."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.anthropic_api_key

    async def evaluate(self, thesis: str, sources: list[str]) -> EvaluationResult:
        """Evaluate a trade thesis on 4 dimensions.

        Args:
            thesis: The trade thesis text to evaluate.
            sources: List of source texts supporting the thesis.

        Returns:
            EvaluationResult with scores, explanations, and pass/fail.
        """
        if self._api_key:
            try:
                return await self._evaluate_with_claude(thesis, sources)
            except Exception as e:
                logger.warning("judge.claude_failed", error=str(e))

        return self._evaluate_heuristic(thesis, sources)

    async def _evaluate_with_claude(
        self, thesis: str, sources: list[str]
    ) -> EvaluationResult:
        """Use Claude to evaluate the thesis."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        sources_text = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
        prompt = EVALUATION_PROMPT.format(thesis=thesis, sources=sources_text)

        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        return self._parse_response(text)

    def _parse_response(self, text: str) -> EvaluationResult:
        """Parse JSON response from Claude into EvaluationResult."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning("judge.parse_failed", text=text[:200])
            return EvaluationResult()

        dimensions = ["groundedness", "reasoning_quality", "evidence_completeness",
                       "calibration_alignment"]
        scores: dict[str, int] = {}
        explanations: dict[str, str] = {}

        for dim in dimensions:
            entry = data.get(dim, {})
            scores[dim] = int(entry.get("score", 0))
            explanations[dim] = entry.get("explanation", "")

        overall = sum(scores.values()) / len(dimensions) if dimensions else 0.0

        passed = (
            scores.get("groundedness", 0) >= GROUNDEDNESS_THRESHOLD
            and scores.get("reasoning_quality", 0) >= REASONING_THRESHOLD
            and scores.get("evidence_completeness", 0) >= EVIDENCE_THRESHOLD
        )

        return EvaluationResult(
            passed=passed,
            scores=scores,
            explanations=explanations,
            overall_quality=round(overall, 2),
        )

    def _evaluate_heuristic(self, thesis: str, sources: list[str]) -> EvaluationResult:
        """Heuristic fallback when Claude API is unavailable."""
        word_count = len(thesis.split())
        source_count = len(sources)

        groundedness = min(10, source_count * 2)
        reasoning = min(10, 4 + word_count // 50)
        evidence = min(10, source_count * 2)
        calibration = 5  # Neutral without LLM assessment

        scores = {
            "groundedness": groundedness,
            "reasoning_quality": reasoning,
            "evidence_completeness": evidence,
            "calibration_alignment": calibration,
        }
        explanations = {k: "Heuristic estimate (Claude unavailable)" for k in scores}
        overall = sum(scores.values()) / 4

        passed = (
            groundedness >= GROUNDEDNESS_THRESHOLD
            and reasoning >= REASONING_THRESHOLD
            and evidence >= EVIDENCE_THRESHOLD
        )

        return EvaluationResult(
            passed=passed,
            scores=scores,
            explanations=explanations,
            overall_quality=round(overall, 2),
        )
