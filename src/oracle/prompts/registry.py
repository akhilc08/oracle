"""Prompt registry — versioned prompt template management backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

DEFAULT_DB_PATH = "oracle.db"

# Seed templates loaded on first run
SEED_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "research_plan",
        "template": (
            "Given market: {market_question}\n"
            "Recent context: {context}\n"
            "Create a 5-step research plan covering:\n"
            "1. Key claims to verify\n"
            "2. Data sources to check\n"
            "3. Expert opinions to seek\n"
            "4. Historical analogues\n"
            "5. Risk factors to monitor\n"
            "Return a structured JSON plan."
        ),
        "variables": ["market_question", "context"],
        "description": "Generate a structured research plan for a prediction market.",
    },
    {
        "name": "evidence_synthesis",
        "template": (
            "Synthesize the following {evidence_count} pieces of evidence for market: "
            "{market_question}\n{evidence}\n"
            "Provide: thesis, confidence (0-100), key_factors"
        ),
        "variables": ["evidence_count", "market_question", "evidence"],
        "description": "Synthesize multiple evidence items into a coherent thesis.",
    },
    {
        "name": "market_analysis",
        "template": (
            "Analyze market metrics for: {market_question}\n"
            "Current price: {current_price}\n"
            "Volume 24h: {volume_24h}\n"
            "Price momentum: {momentum}\n"
            "Provide: fair_value_estimate, momentum_signal, recommended_size_pct"
        ),
        "variables": ["market_question", "current_price", "volume_24h", "momentum"],
        "description": "Analyze quantitative market metrics and provide trading signals.",
    },
    {
        "name": "trade_thesis",
        "template": (
            "Generate a structured trade thesis for: {market_question}\n"
            "Research: {research_summary}\n"
            "Quant: {quant_summary}\n"
            "Risk: {risk_summary}\n"
            "Format: {thesis_schema}"
        ),
        "variables": ["market_question", "research_summary", "quant_summary",
                       "risk_summary", "thesis_schema"],
        "description": "Generate a structured trade thesis combining research, quant, and risk.",
    },
    {
        "name": "reflection",
        "template": (
            "Review this prediction for cognitive biases:\n"
            "Market: {market_question}\n"
            "Thesis: {thesis}\n"
            "Confidence: {confidence}\n"
            "Check for: anchoring, recency bias, confirmation bias, overconfidence\n"
            "Return JSON: {schema}"
        ),
        "variables": ["market_question", "thesis", "confidence", "schema"],
        "description": "Detect cognitive biases in a prediction thesis.",
    },
    {
        "name": "evaluation_judge",
        "template": (
            "Score this trade thesis on 4 dimensions (0-10 each):\n"
            "Thesis: {thesis}\n"
            "Sources: {sources}\n"
            "Dimensions: groundedness, reasoning_quality, evidence_completeness, "
            "calibration_alignment\n"
            "Return JSON scores with explanations."
        ),
        "variables": ["thesis", "sources"],
        "description": "LLM-as-judge scoring for trade thesis quality.",
    },
    {
        "name": "entity_extraction",
        "template": (
            "Extract entities and relationships from this text:\n"
            "{text}\n"
            "Return JSON array of: {entity_schema}"
        ),
        "variables": ["text", "entity_schema"],
        "description": "Extract named entities and relationships from text.",
    },
    {
        "name": "claim_verification",
        "template": (
            "Verify this claim against the provided sources:\n"
            "Claim: {claim}\n"
            "Sources: {sources}\n"
            "Return: {{verified: bool, confidence: float, supporting_quote: str|null}}"
        ),
        "variables": ["claim", "sources"],
        "description": "Verify a factual claim against source documents.",
    },
    {
        "name": "post_mortem",
        "template": (
            "Analyze this prediction outcome:\n"
            "Original thesis: {thesis}\n"
            "Predicted: {predicted_prob} probability YES\n"
            "Actual outcome: {outcome}\n"
            "Market context: {market_context}\n"
            "Analyze: process quality, luck factor, key lessons"
        ),
        "variables": ["thesis", "predicted_prob", "outcome", "market_context"],
        "description": "Post-mortem analysis of a resolved prediction.",
    },
]


@dataclass
class PromptTemplate:
    """A versioned prompt template."""

    name: str
    version: int
    template: str
    variables: list[str]
    description: str = ""
    is_active: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables."""
        return self.template.format(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


class PromptRegistry:
    """SQLite-backed versioned prompt template registry.

    Stores prompt templates with version history. Only one version per name
    is active at a time.
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
        """Create prompt_templates table and seed default templates if empty."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                template TEXT NOT NULL,
                variables TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                description TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (name, version)
            )
        """)
        conn.commit()

        # Seed if empty
        row = conn.execute("SELECT COUNT(*) as cnt FROM prompt_templates").fetchone()
        if row["cnt"] == 0:
            self._seed_templates()

    def _seed_templates(self) -> None:
        """Insert the default 9 seed templates."""
        for tpl in SEED_TEMPLATES:
            self.register(
                name=tpl["name"],
                template=tpl["template"],
                variables=tpl["variables"],
                description=tpl["description"],
            )
        logger.info("prompts.seeded", count=len(SEED_TEMPLATES))

    def register(
        self,
        name: str,
        template: str,
        variables: list[str],
        description: str = "",
    ) -> PromptTemplate:
        """Register a new version of a prompt template.

        Deactivates previous versions and inserts the new one as active.
        """
        conn = self._get_conn()

        # Get next version number
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) as max_v FROM prompt_templates WHERE name = ?",
            (name,),
        ).fetchone()
        next_version = row["max_v"] + 1

        # Deactivate old versions
        conn.execute(
            "UPDATE prompt_templates SET is_active = 0 WHERE name = ?",
            (name,),
        )

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO prompt_templates
               (name, version, template, variables, created_at, is_active, description)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (name, next_version, template, json.dumps(variables), now, description),
        )
        conn.commit()

        tpl = PromptTemplate(
            name=name,
            version=next_version,
            template=template,
            variables=variables,
            description=description,
            is_active=True,
            created_at=now,
        )
        logger.info("prompts.registered", name=name, version=next_version)
        return tpl

    def get(self, name: str) -> PromptTemplate | None:
        """Get the active version of a named template."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM prompt_templates WHERE name = ? AND is_active = 1",
            (name,),
        ).fetchone()

        if row is None:
            return None

        return PromptTemplate(
            name=row["name"],
            version=row["version"],
            template=row["template"],
            variables=json.loads(row["variables"]),
            description=row["description"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
        )

    def render(self, name: str, **kwargs: Any) -> str:
        """Fetch and render a template by name.

        Raises:
            KeyError: If template not found.
        """
        tpl = self.get(name)
        if tpl is None:
            raise KeyError(f"Prompt template '{name}' not found")
        return tpl.render(**kwargs)

    def list_active(self) -> list[PromptTemplate]:
        """List all active prompt templates."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM prompt_templates WHERE is_active = 1 ORDER BY name"
        ).fetchall()

        return [
            PromptTemplate(
                name=row["name"],
                version=row["version"],
                template=row["template"],
                variables=json.loads(row["variables"]),
                description=row["description"],
                is_active=True,
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_versions(self, name: str) -> list[PromptTemplate]:
        """Get all versions of a named template."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM prompt_templates WHERE name = ? ORDER BY version",
            (name,),
        ).fetchall()

        return [
            PromptTemplate(
                name=row["name"],
                version=row["version"],
                template=row["template"],
                variables=json.loads(row["variables"]),
                description=row["description"],
                is_active=bool(row["is_active"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def set_active_version(self, name: str, version: int) -> None:
        """Set a specific version as active (deactivating others)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE prompt_templates SET is_active = 0 WHERE name = ?",
            (name,),
        )
        conn.execute(
            "UPDATE prompt_templates SET is_active = 1 WHERE name = ? AND version = ?",
            (name, version),
        )
        conn.commit()
        logger.info("prompts.version_activated", name=name, version=version)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
