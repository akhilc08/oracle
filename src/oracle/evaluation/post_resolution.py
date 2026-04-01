"""Post-resolution evaluation — measure prediction accuracy after market resolves."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    predicted_prob REAL NOT NULL,
    predicted_direction TEXT NOT NULL,
    actual_outcome INTEGER NOT NULL,
    actual_final_price REAL NOT NULL,
    is_correct INTEGER NOT NULL,
    brier_score REAL NOT NULL,
    log_loss REAL DEFAULT 0.0,
    confidence_bucket TEXT NOT NULL,
    divergence_from_market REAL DEFAULT 0.0,
    market_prob_at_entry REAL DEFAULT 0.0,
    is_alpha INTEGER DEFAULT 0,
    hallucination_flagged INTEGER DEFAULT 0,
    bias_flagged INTEGER DEFAULT 0,
    latency_seconds REAL DEFAULT 0.0,
    category TEXT DEFAULT 'other',
    evaluated_at TEXT NOT NULL
);
"""

# Migration: add new columns to existing DBs that pre-date this schema
_MIGRATION_SQL = [
    "ALTER TABLE prediction_outcomes ADD COLUMN log_loss REAL DEFAULT 0.0",
    "ALTER TABLE prediction_outcomes ADD COLUMN market_prob_at_entry REAL DEFAULT 0.0",
    "ALTER TABLE prediction_outcomes ADD COLUMN hallucination_flagged INTEGER DEFAULT 0",
    "ALTER TABLE prediction_outcomes ADD COLUMN bias_flagged INTEGER DEFAULT 0",
    "ALTER TABLE prediction_outcomes ADD COLUMN latency_seconds REAL DEFAULT 0.0",
]


@dataclass
class ResolutionResult:
    """Result of evaluating a single prediction after market resolution."""

    trade_id: str
    is_correct: bool = False
    brier_score: float = 1.0
    confidence_bucket: str = "50-60%"
    generated_alpha: bool = False
    divergence_from_market: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "is_correct": self.is_correct,
            "brier_score": round(self.brier_score, 4),
            "confidence_bucket": self.confidence_bucket,
            "generated_alpha": self.generated_alpha,
            "divergence_from_market": round(self.divergence_from_market, 4),
        }


def _assign_bucket(prob: float) -> str:
    """Assign a confidence probability to its calibration bucket."""
    if prob < 0.6:
        return "50-60%"
    elif prob < 0.7:
        return "60-70%"
    elif prob < 0.8:
        return "70-80%"
    elif prob < 0.9:
        return "80-90%"
    else:
        return "90-100%"


class PostResolutionEvaluator:
    """Evaluates prediction accuracy after market resolution."""

    def __init__(self, db_path: str = "oracle.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create the prediction_outcomes table and apply any pending migrations."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)
            for stmt in _MIGRATION_SQL:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass  # Column already exists
            await db.commit()

    async def evaluate_prediction(
        self,
        trade_id: str,
        market_id: str,
        predicted_prob: float,
        predicted_direction: str,
        actual_outcome: bool,
        actual_final_price: float,
        market_consensus: float | None = None,
        category: str = "other",
        hallucination_flagged: bool = False,
        bias_flagged: bool = False,
        latency_seconds: float = 0.0,
    ) -> ResolutionResult:
        """Evaluate a prediction after market resolution.

        Args:
            trade_id: Unique trade identifier.
            market_id: Market identifier.
            predicted_prob: Oracle's predicted probability (0-1).
            predicted_direction: "yes" or "no".
            actual_outcome: True if resolved YES, False if NO.
            actual_final_price: Final market price before resolution.
            market_consensus: Market consensus probability at time of trade.
            category: Market category.

        Returns:
            ResolutionResult with accuracy metrics.
        """
        outcome_int = 1 if actual_outcome else 0

        # Binary accuracy: did we predict the right direction?
        predicted_yes = predicted_direction == "yes"
        is_correct = predicted_yes == actual_outcome

        # Brier score: (predicted_prob - outcome)^2
        # For "no" predictions, use 1 - predicted_prob as the "yes" probability
        effective_prob = predicted_prob if predicted_yes else (1 - predicted_prob)
        brier_score = (effective_prob - outcome_int) ** 2

        # Log loss: -[y*log(p) + (1-y)*log(1-p)] — clipped to avoid log(0)
        import math
        p_clipped = max(1e-7, min(1 - 1e-7, effective_prob))
        log_loss = -(outcome_int * math.log(p_clipped) + (1 - outcome_int) * math.log(1 - p_clipped))

        # Confidence bucket
        confidence_bucket = _assign_bucket(predicted_prob)

        # Alpha: diverged >10% from market consensus AND was correct
        divergence = 0.0
        generated_alpha = False
        if market_consensus is not None:
            divergence = abs(effective_prob - market_consensus)
            generated_alpha = divergence > 0.10 and is_correct

        result = ResolutionResult(
            trade_id=trade_id,
            is_correct=is_correct,
            brier_score=brier_score,
            confidence_bucket=confidence_bucket,
            generated_alpha=generated_alpha,
            divergence_from_market=divergence,
        )

        # Persist to SQLite
        await self._save_result(
            trade_id=trade_id,
            market_id=market_id,
            predicted_prob=predicted_prob,
            predicted_direction=predicted_direction,
            actual_outcome=outcome_int,
            actual_final_price=actual_final_price,
            is_correct=is_correct,
            brier_score=brier_score,
            log_loss=log_loss,
            confidence_bucket=confidence_bucket,
            divergence=divergence,
            market_prob_at_entry=market_consensus or 0.0,
            is_alpha=generated_alpha,
            hallucination_flagged=hallucination_flagged,
            bias_flagged=bias_flagged,
            latency_seconds=latency_seconds,
            category=category,
        )

        logger.info(
            "post_resolution.evaluated",
            trade_id=trade_id,
            is_correct=is_correct,
            brier_score=round(brier_score, 4),
            alpha=generated_alpha,
        )

        return result

    async def aggregate_stats(self) -> dict[str, Any]:
        """Compute all aggregate metrics across evaluated predictions."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(
                "SELECT COUNT(*), AVG(is_correct), AVG(brier_score), AVG(log_loss), "
                "AVG(is_alpha), AVG(hallucination_flagged), AVG(bias_flagged), "
                "AVG(predicted_prob - market_prob_at_entry), "
                "AVG(is_correct) - AVG(market_prob_at_entry), "
                "AVG(latency_seconds) "
                "FROM prediction_outcomes"
            ) as cursor:
                row = await cursor.fetchone()
                if not row or row[0] == 0:
                    return {"total_predictions": 0}

                total = row[0]
                overall_accuracy = row[1] or 0.0
                avg_brier = row[2] or 0.0
                avg_log_loss = row[3] or 0.0
                alpha_rate = row[4] or 0.0
                hallucination_catch_rate = row[5] or 0.0
                bias_flag_rate = row[6] or 0.0
                ev_per_trade = row[7] or 0.0
                market_adjusted_return = row[8] or 0.0
                avg_latency = row[9] or 0.0

            # Hit rate by confidence bucket
            hit_rate_by_tier: dict[str, Any] = {}
            async with db.execute(
                "SELECT confidence_bucket, COUNT(*), AVG(is_correct) "
                "FROM prediction_outcomes GROUP BY confidence_bucket"
            ) as cursor:
                async for row in cursor:
                    hit_rate_by_tier[row[0]] = {
                        "count": row[1],
                        "accuracy": round(row[2], 4) if row[2] is not None else None,
                    }

            # By category
            by_category: dict[str, Any] = {}
            async with db.execute(
                "SELECT category, COUNT(*), AVG(is_correct), AVG(brier_score), "
                "AVG(predicted_prob - market_prob_at_entry) "
                "FROM prediction_outcomes GROUP BY category"
            ) as cursor:
                async for row in cursor:
                    by_category[row[0]] = {
                        "count": row[1],
                        "accuracy": round(row[2], 4),
                        "brier_score": round(row[3], 4),
                        "ev_per_trade": round(row[4] or 0.0, 4),
                    }

            return {
                "total_predictions": total,
                "overall_accuracy": round(overall_accuracy, 4),
                "brier_score": round(avg_brier, 4),
                "log_loss": round(avg_log_loss, 4),
                "alpha_rate": round(alpha_rate, 4),
                "ev_per_trade": round(ev_per_trade, 4),
                "market_adjusted_return": round(market_adjusted_return, 4),
                "hallucination_catch_rate": round(hallucination_catch_rate, 4),
                "bias_flag_rate": round(bias_flag_rate, 4),
                "avg_latency_seconds": round(avg_latency, 3),
                "hit_rate_by_tier": hit_rate_by_tier,
                "by_category": by_category,
            }

    async def _save_result(
        self,
        trade_id: str,
        market_id: str,
        predicted_prob: float,
        predicted_direction: str,
        actual_outcome: int,
        actual_final_price: float,
        is_correct: bool,
        brier_score: float,
        log_loss: float,
        confidence_bucket: str,
        divergence: float,
        market_prob_at_entry: float,
        is_alpha: bool,
        hallucination_flagged: bool,
        bias_flagged: bool,
        latency_seconds: float,
        category: str,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO prediction_outcomes "
                "(trade_id, market_id, predicted_prob, predicted_direction, "
                "actual_outcome, actual_final_price, is_correct, brier_score, log_loss, "
                "confidence_bucket, divergence_from_market, market_prob_at_entry, "
                "is_alpha, hallucination_flagged, bias_flagged, latency_seconds, "
                "category, evaluated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade_id, market_id, predicted_prob, predicted_direction,
                    actual_outcome, actual_final_price, int(is_correct), brier_score, log_loss,
                    confidence_bucket, divergence, market_prob_at_entry,
                    int(is_alpha), int(hallucination_flagged), int(bias_flagged), latency_seconds,
                    category, datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
