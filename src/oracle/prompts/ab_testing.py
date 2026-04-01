"""A/B testing framework for prompt templates."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from scipy import stats

from oracle.prompts.registry import PromptRegistry

logger = structlog.get_logger()

DEFAULT_DB_PATH = "oracle.db"
MIN_SAMPLES_PER_VARIANT = 30


@dataclass
class ABTestResult:
    """Result of an A/B test analysis."""

    winner: str  # "A", "B", or "inconclusive"
    p_value: float
    sample_sizes: dict[str, int]
    metric_means: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "p_value": round(self.p_value, 6),
            "sample_sizes": self.sample_sizes,
            "metric_means": {k: round(v, 6) for k, v in self.metric_means.items()},
        }


class ABTestManager:
    """SQLite-backed A/B testing for prompt templates.

    Creates tests between two prompt versions, randomly assigns variants,
    records results, and uses a two-sample t-test to determine winners.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Create A/B test tables."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                prompt_a_version INTEGER NOT NULL,
                prompt_b_version INTEGER NOT NULL,
                metric TEXT NOT NULL DEFAULT 'brier_score',
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                prediction_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                prediction_id TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id)
            )
        """)
        conn.commit()

    def create_test(
        self,
        name: str,
        prompt_a_version: int,
        prompt_b_version: int,
        metric: str = "brier_score",
    ) -> int:
        """Create a new A/B test.

        Args:
            name: Test name (typically the prompt template name).
            prompt_a_version: Version number of variant A.
            prompt_b_version: Version number of variant B.
            metric: The metric to compare (default: brier_score).

        Returns:
            The test ID.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """INSERT INTO ab_tests (name, prompt_a_version, prompt_b_version, metric, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (name, prompt_a_version, prompt_b_version, metric, now),
        )
        conn.commit()
        test_id = cursor.lastrowid or 0
        logger.info(
            "ab_test.created",
            test_id=test_id,
            name=name,
            variants=(prompt_a_version, prompt_b_version),
        )
        return test_id

    def assign_variant(self, test_id: int, prediction_id: str) -> str:
        """Randomly assign a prediction to variant A or B (50/50).

        Args:
            test_id: The A/B test ID.
            prediction_id: Unique prediction identifier.

        Returns:
            "A" or "B".
        """
        conn = self._get_conn()
        variant = random.choice(["A", "B"])
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO ab_assignments (test_id, prediction_id, variant, assigned_at)
               VALUES (?, ?, ?, ?)""",
            (test_id, prediction_id, variant, now),
        )
        conn.commit()
        return variant

    def record_result(
        self, test_id: int, prediction_id: str, metric_value: float
    ) -> None:
        """Record a metric result for a prediction in the test.

        Args:
            test_id: The A/B test ID.
            prediction_id: The prediction that was evaluated.
            metric_value: The metric value (e.g., Brier score).
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO ab_results (test_id, prediction_id, metric_value, recorded_at)
               VALUES (?, ?, ?, ?)""",
            (test_id, prediction_id, metric_value, now),
        )
        conn.commit()

    def analyze(self, test_id: int) -> ABTestResult:
        """Analyze an A/B test using a two-sample t-test.

        Requires at least MIN_SAMPLES_PER_VARIANT (30) per variant.
        Lower metric value = better (for Brier score).

        Returns:
            ABTestResult with winner, p-value, and statistics.
        """
        conn = self._get_conn()

        # Get results joined with assignments
        rows = conn.execute(
            """
            SELECT a.variant, r.metric_value
            FROM ab_results r
            JOIN ab_assignments a
              ON r.test_id = a.test_id AND r.prediction_id = a.prediction_id
            WHERE r.test_id = ?
            """,
            (test_id,),
        ).fetchall()

        values_a: list[float] = []
        values_b: list[float] = []
        for row in rows:
            if row["variant"] == "A":
                values_a.append(row["metric_value"])
            else:
                values_b.append(row["metric_value"])

        sample_sizes = {"A": len(values_a), "B": len(values_b)}
        mean_a = sum(values_a) / len(values_a) if values_a else 0.0
        mean_b = sum(values_b) / len(values_b) if values_b else 0.0
        metric_means = {"A": mean_a, "B": mean_b}

        # Need minimum samples
        if len(values_a) < MIN_SAMPLES_PER_VARIANT or len(values_b) < MIN_SAMPLES_PER_VARIANT:
            return ABTestResult(
                winner="inconclusive",
                p_value=1.0,
                sample_sizes=sample_sizes,
                metric_means=metric_means,
            )

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)

        # Determine winner (lower Brier score = better)
        if p_value < 0.05:
            winner = "A" if mean_a < mean_b else "B"
        else:
            winner = "inconclusive"

        return ABTestResult(
            winner=winner,
            p_value=float(p_value),
            sample_sizes=sample_sizes,
            metric_means=metric_means,
        )

    def promote_winner(self, test_id: int) -> str | None:
        """Promote the winning variant's prompt version to active.

        Args:
            test_id: The A/B test ID.

        Returns:
            The winner ("A" or "B") if promoted, None if inconclusive.
        """
        result = self.analyze(test_id)
        if result.winner == "inconclusive":
            logger.info("ab_test.promote_skipped", test_id=test_id, reason="inconclusive")
            return None

        conn = self._get_conn()
        test = conn.execute(
            "SELECT * FROM ab_tests WHERE id = ?", (test_id,)
        ).fetchone()

        if test is None:
            return None

        winning_version = (
            test["prompt_a_version"] if result.winner == "A" else test["prompt_b_version"]
        )

        # Update prompt registry
        registry = PromptRegistry(db_path=self.db_path)
        registry.initialize()
        registry.set_active_version(test["name"], winning_version)

        # Mark test as inactive
        conn.execute(
            "UPDATE ab_tests SET is_active = 0 WHERE id = ?", (test_id,)
        )
        conn.commit()

        logger.info(
            "ab_test.promoted",
            test_id=test_id,
            winner=result.winner,
            version=winning_version,
        )
        return result.winner

    def list_active_tests(self) -> list[dict[str, Any]]:
        """List all active A/B tests."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM ab_tests WHERE is_active = 1"
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
