"""Quality gate enforcement — orchestrates pre-execution checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from oracle.evaluation.judge import (
    EVIDENCE_THRESHOLD,
    GROUNDEDNESS_THRESHOLD,
    REASONING_THRESHOLD,
    EvaluationJudge,
    EvaluationResult,
)
from oracle.evaluation.hallucination import HallucinationDetector, HallucinationResult

logger = structlog.get_logger()

MAX_UNGROUNDED_CLAIMS = 1


@dataclass
class GateResult:
    """Result of the quality gate evaluation."""

    approved: bool = False
    evaluation: EvaluationResult = field(default_factory=EvaluationResult)
    hallucination_check: HallucinationResult = field(default_factory=HallucinationResult)
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "evaluation": self.evaluation.to_dict(),
            "hallucination_check": self.hallucination_check.to_dict(),
            "blocking_reasons": self.blocking_reasons,
        }


class TradeGate:
    """Orchestrates LLM-as-judge + hallucination detection before trade execution."""

    def __init__(self, api_key: str | None = None) -> None:
        self._judge = EvaluationJudge(api_key=api_key)
        self._detector = HallucinationDetector(api_key=api_key)

    async def evaluate_trade_proposal(
        self,
        thesis: str,
        sources: list[str],
    ) -> GateResult:
        """Run quality gates on a trade proposal.

        Args:
            thesis: The trade thesis / research report text.
            sources: Supporting source texts.

        Returns:
            GateResult with approval status and blocking reasons.
        """
        blocking_reasons: list[str] = []

        # 1. LLM-as-judge evaluation
        evaluation = await self._judge.evaluate(thesis, sources)

        scores = evaluation.scores
        if scores.get("groundedness", 0) < GROUNDEDNESS_THRESHOLD:
            blocking_reasons.append(
                f"Groundedness score {scores['groundedness']} < {GROUNDEDNESS_THRESHOLD}"
            )
        if scores.get("reasoning_quality", 0) < REASONING_THRESHOLD:
            blocking_reasons.append(
                f"Reasoning quality score {scores['reasoning_quality']} < {REASONING_THRESHOLD}"
            )
        if scores.get("evidence_completeness", 0) < EVIDENCE_THRESHOLD:
            blocking_reasons.append(
                f"Evidence completeness score {scores['evidence_completeness']} < {EVIDENCE_THRESHOLD}"
            )

        # 2. Hallucination detection
        hallucination_check = await self._detector.detect(thesis, sources)

        if len(hallucination_check.ungrounded_claims) > MAX_UNGROUNDED_CLAIMS:
            blocking_reasons.append(
                f"Ungrounded claims ({len(hallucination_check.ungrounded_claims)}) "
                f"> {MAX_UNGROUNDED_CLAIMS}"
            )

        if hallucination_check.has_self_contradictions:
            blocking_reasons.append("Self-contradictions detected in thesis")

        approved = len(blocking_reasons) == 0

        logger.info(
            "gate.evaluated",
            approved=approved,
            blocking_reasons=blocking_reasons,
            overall_quality=evaluation.overall_quality,
            hallucination_rate=hallucination_check.hallucination_rate,
        )

        return GateResult(
            approved=approved,
            evaluation=evaluation,
            hallucination_check=hallucination_check,
            blocking_reasons=blocking_reasons,
        )
