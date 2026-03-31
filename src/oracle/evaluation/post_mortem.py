"""Post-mortem generation — analyze predictions after resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from oracle.config import settings

logger = structlog.get_logger()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS post_mortems (
    trade_id TEXT PRIMARY KEY,
    was_correct INTEGER NOT NULL,
    process_quality TEXT NOT NULL,
    good_luck_factor REAL NOT NULL,
    key_lessons TEXT NOT NULL,
    what_went_wrong TEXT DEFAULT NULL,
    what_went_right TEXT DEFAULT NULL,
    full_analysis TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

POST_MORTEM_PROMPT = """You are analyzing a prediction market trade after the market has resolved.

ORIGINAL THESIS:
{thesis}

ACTUAL OUTCOME: {outcome}
WAS PREDICTION CORRECT: {was_correct}

MARKET RESOLUTION DETAILS:
{resolution_details}

Analyze this prediction. Distinguish between:
- "Good process, bad luck" — the reasoning was sound but the outcome was unpredictable
- "Bad process, got lucky" — the reasoning was flawed but the outcome happened to be correct
- "Good process, good outcome" — sound reasoning led to correct prediction
- "Bad process, bad outcome" — flawed reasoning led to wrong prediction

Respond ONLY with valid JSON:
{{
  "process_quality": "good" or "bad",
  "good_luck_factor": <float 0.0-1.0>,
  "key_lessons": ["lesson 1", "lesson 2", ...],
  "what_went_wrong": "<string or null>",
  "what_went_right": "<string or null>",
  "full_analysis": "<2-4 paragraph analysis>"
}}
"""


@dataclass
class PostMortem:
    """Post-mortem analysis of a resolved prediction."""

    trade_id: str
    was_correct: bool = False
    process_quality: str = "bad"  # "good" or "bad"
    good_luck_factor: float = 0.5
    key_lessons: list[str] = field(default_factory=list)
    what_went_wrong: str | None = None
    what_went_right: str | None = None
    full_analysis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "was_correct": self.was_correct,
            "process_quality": self.process_quality,
            "good_luck_factor": round(self.good_luck_factor, 2),
            "key_lessons": self.key_lessons,
            "what_went_wrong": self.what_went_wrong,
            "what_went_right": self.what_went_right,
            "full_analysis": self.full_analysis,
        }


class PostMortemGenerator:
    """Generates post-mortem analyses for resolved predictions."""

    def __init__(self, db_path: str = "oracle.db", api_key: str | None = None) -> None:
        self._db_path = db_path
        self._api_key = api_key or settings.anthropic_api_key

    async def initialize(self) -> None:
        """Create post_mortems table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

    async def generate(
        self,
        trade_id: str,
        original_thesis: str,
        actual_outcome: bool,
        market_resolution_details: str = "",
    ) -> PostMortem:
        """Generate a post-mortem analysis for a resolved trade.

        Args:
            trade_id: Unique trade identifier.
            original_thesis: The original trade thesis text.
            actual_outcome: True if market resolved YES, False if NO.
            market_resolution_details: Additional context about resolution.

        Returns:
            PostMortem with analysis and lessons.
        """
        was_correct_str = "Yes" if actual_outcome else "No"
        outcome_str = "YES" if actual_outcome else "NO"

        post_mortem = PostMortem(trade_id=trade_id, was_correct=actual_outcome)

        if self._api_key:
            try:
                post_mortem = await self._generate_with_claude(
                    trade_id, original_thesis, outcome_str, was_correct_str,
                    market_resolution_details,
                )
            except Exception as e:
                logger.warning("post_mortem.claude_failed", error=str(e))
                post_mortem = self._generate_heuristic(
                    trade_id, original_thesis, actual_outcome,
                )
        else:
            post_mortem = self._generate_heuristic(
                trade_id, original_thesis, actual_outcome,
            )

        # Persist
        await self._save(post_mortem)

        logger.info(
            "post_mortem.generated",
            trade_id=trade_id,
            was_correct=post_mortem.was_correct,
            process_quality=post_mortem.process_quality,
        )

        return post_mortem

    async def _generate_with_claude(
        self,
        trade_id: str,
        thesis: str,
        outcome_str: str,
        was_correct_str: str,
        resolution_details: str,
    ) -> PostMortem:
        """Use Claude to generate post-mortem."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        prompt = POST_MORTEM_PROMPT.format(
            thesis=thesis,
            outcome=outcome_str,
            was_correct=was_correct_str,
            resolution_details=resolution_details or "No additional details provided.",
        )

        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)

        return PostMortem(
            trade_id=trade_id,
            was_correct=was_correct_str == "Yes",
            process_quality=data.get("process_quality", "bad"),
            good_luck_factor=float(data.get("good_luck_factor", 0.5)),
            key_lessons=data.get("key_lessons", []),
            what_went_wrong=data.get("what_went_wrong"),
            what_went_right=data.get("what_went_right"),
            full_analysis=data.get("full_analysis", ""),
        )

    def _generate_heuristic(
        self, trade_id: str, thesis: str, actual_outcome: bool
    ) -> PostMortem:
        """Heuristic post-mortem when Claude is unavailable."""
        word_count = len(thesis.split())
        has_evidence = word_count > 50

        if actual_outcome and has_evidence:
            process_quality = "good"
            luck_factor = 0.2
            what_went_right = "Thesis was well-supported with evidence and prediction was correct."
            what_went_wrong = None
            lessons = ["Continue with evidence-based approach."]
        elif actual_outcome and not has_evidence:
            process_quality = "bad"
            luck_factor = 0.7
            what_went_right = "Prediction was correct despite thin evidence."
            what_went_wrong = "Thesis lacked sufficient supporting evidence."
            lessons = ["Require more evidence before trading.", "Got lucky — don't mistake luck for skill."]
        elif not actual_outcome and has_evidence:
            process_quality = "good"
            luck_factor = 0.6
            what_went_right = "Research process was thorough."
            what_went_wrong = "Outcome went against well-reasoned thesis."
            lessons = ["Good process can still lead to wrong outcomes.", "Review if evidence was misinterpreted."]
        else:
            process_quality = "bad"
            luck_factor = 0.3
            what_went_right = None
            what_went_wrong = "Insufficient evidence and incorrect prediction."
            lessons = ["Require stronger evidence before trading.", "Improve research process."]

        full_analysis = (
            f"Prediction was {'correct' if actual_outcome else 'incorrect'}. "
            f"Process quality assessed as '{process_quality}'. "
            f"Thesis contained {word_count} words with "
            f"{'adequate' if has_evidence else 'insufficient'} evidence. "
            f"Luck factor: {luck_factor:.1f}."
        )

        return PostMortem(
            trade_id=trade_id,
            was_correct=actual_outcome,
            process_quality=process_quality,
            good_luck_factor=luck_factor,
            key_lessons=lessons,
            what_went_wrong=what_went_wrong,
            what_went_right=what_went_right,
            full_analysis=full_analysis,
        )

    async def get_post_mortem(self, trade_id: str) -> PostMortem | None:
        """Retrieve a stored post-mortem by trade ID."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT trade_id, was_correct, process_quality, good_luck_factor, "
                "key_lessons, what_went_wrong, what_went_right, full_analysis "
                "FROM post_mortems WHERE trade_id=?",
                (trade_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                return PostMortem(
                    trade_id=row[0],
                    was_correct=bool(row[1]),
                    process_quality=row[2],
                    good_luck_factor=row[3],
                    key_lessons=json.loads(row[4]),
                    what_went_wrong=row[5],
                    what_went_right=row[6],
                    full_analysis=row[7],
                )

    async def _save(self, pm: PostMortem) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO post_mortems "
                "(trade_id, was_correct, process_quality, good_luck_factor, "
                "key_lessons, what_went_wrong, what_went_right, full_analysis, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pm.trade_id, int(pm.was_correct), pm.process_quality,
                    pm.good_luck_factor, json.dumps(pm.key_lessons),
                    pm.what_went_wrong, pm.what_went_right, pm.full_analysis,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
