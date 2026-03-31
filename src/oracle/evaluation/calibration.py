"""Calibration monitoring — track predicted vs actual outcome rates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calibration_records (
    market_id TEXT PRIMARY KEY,
    predicted_prob REAL NOT NULL,
    confidence_bucket TEXT NOT NULL,
    resolved INTEGER DEFAULT 0,
    resolved_yes INTEGER DEFAULT NULL,
    recorded_at TEXT NOT NULL,
    resolved_at TEXT DEFAULT NULL
);
"""

BUCKETS = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]


def _assign_bucket(prob: float) -> str:
    """Assign a probability to its calibration bucket."""
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


@dataclass
class BucketData:
    """Calibration data for a single bucket."""

    range: str
    predicted_avg: float = 0.0
    actual_rate: float = 0.0
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "range": self.range,
            "predicted_avg": round(self.predicted_avg, 4),
            "actual_rate": round(self.actual_rate, 4),
            "count": self.count,
        }


@dataclass
class CalibrationData:
    """Full calibration analysis."""

    buckets: list[BucketData] = field(default_factory=list)
    calibration_error: float = 0.0
    is_well_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "calibration_error": round(self.calibration_error, 4),
            "is_well_calibrated": self.is_well_calibrated,
        }


class CalibrationMonitor:
    """Tracks calibration — how well predicted probabilities match actual outcome rates."""

    def __init__(self, db_path: str = "oracle.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create calibration_records table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

    async def record_prediction(
        self, market_id: str, predicted_prob: float, confidence_bucket: str | None = None
    ) -> None:
        """Record a prediction when a trade is placed.

        Args:
            market_id: Market identifier.
            predicted_prob: Oracle's predicted probability.
            confidence_bucket: Override bucket assignment if desired.
        """
        bucket = confidence_bucket or _assign_bucket(predicted_prob)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO calibration_records "
                "(market_id, predicted_prob, confidence_bucket, resolved, recorded_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (market_id, predicted_prob, bucket, datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()

        logger.info(
            "calibration.prediction_recorded",
            market_id=market_id,
            predicted_prob=predicted_prob,
            bucket=bucket,
        )

    async def record_outcome(self, market_id: str, resolved_yes: bool) -> None:
        """Record the actual outcome when a market resolves.

        Args:
            market_id: Market identifier.
            resolved_yes: True if market resolved YES, False if NO.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE calibration_records SET resolved=1, resolved_yes=?, resolved_at=? "
                "WHERE market_id=?",
                (int(resolved_yes), datetime.now(timezone.utc).isoformat(), market_id),
            )
            await db.commit()

        logger.info(
            "calibration.outcome_recorded",
            market_id=market_id,
            resolved_yes=resolved_yes,
        )

    async def compute_calibration(self) -> CalibrationData:
        """Compute calibration data across all resolved predictions.

        Returns:
            CalibrationData with bucket breakdown and calibration error.
        """
        async with aiosqlite.connect(self._db_path) as db:
            bucket_data: list[BucketData] = []
            deviations: list[float] = []

            for bucket_range in BUCKETS:
                async with db.execute(
                    "SELECT AVG(predicted_prob), AVG(resolved_yes), COUNT(*) "
                    "FROM calibration_records "
                    "WHERE confidence_bucket=? AND resolved=1",
                    (bucket_range,),
                ) as cursor:
                    row = await cursor.fetchone()

                    count = row[2] if row else 0
                    if count == 0:
                        bucket_data.append(BucketData(range=bucket_range))
                        continue

                    predicted_avg = row[0] or 0.0
                    actual_rate = row[1] or 0.0

                    bucket_data.append(BucketData(
                        range=bucket_range,
                        predicted_avg=predicted_avg,
                        actual_rate=actual_rate,
                        count=count,
                    ))
                    deviations.append(abs(predicted_avg - actual_rate))

            calibration_error = sum(deviations) / len(deviations) if deviations else 0.0
            is_well_calibrated = calibration_error < 0.10

            return CalibrationData(
                buckets=bucket_data,
                calibration_error=calibration_error,
                is_well_calibrated=is_well_calibrated,
            )

    async def get_calibration_chart_data(self) -> dict[str, Any]:
        """Return JSON-serializable calibration data for frontend plotting."""
        cal = await self.compute_calibration()

        labels = [b.range for b in cal.buckets]
        predicted = [b.predicted_avg for b in cal.buckets]
        actual = [b.actual_rate for b in cal.buckets]
        counts = [b.count for b in cal.buckets]

        # Ideal calibration line (midpoints of each bucket)
        ideal = [0.55, 0.65, 0.75, 0.85, 0.95]

        return {
            "labels": labels,
            "predicted_avg": predicted,
            "actual_rate": actual,
            "ideal": ideal,
            "counts": counts,
            "calibration_error": round(cal.calibration_error, 4),
            "is_well_calibrated": cal.is_well_calibrated,
        }
