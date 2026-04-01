"""Training data generation using Claude API for fine-tuning Oracle models."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from oracle.config import settings

logger = structlog.get_logger()

CATEGORIES = {
    "market_summarization": {
        "count": 5000,
        "instruction": (
            "Summarize the following prediction market and its context into a structured "
            "analysis with: market_question, key_factors (list), current_sentiment, "
            "probability_drivers, and risk_factors."
        ),
    },
    "entity_extraction": {
        "count": 3000,
        "instruction": (
            "Extract all named entities and their relationships from the following text. "
            "Return a JSON array of objects with fields: entity, type "
            "(person/organization/event/location/policy), and relationships "
            "(array of {target, relation_type})."
        ),
    },
    "factual_qa": {
        "count": 8000,
        "instruction": (
            "Answer the following question using only the provided context. "
            "Include step-by-step reasoning before your final answer. "
            "If the context is insufficient, state what additional information is needed."
        ),
    },
    "prediction_reasoning": {
        "count": 2000,
        "instruction": (
            "Given the market question and supporting evidence, provide a prediction with: "
            "predicted_probability (0.0-1.0), confidence_level (low/medium/high), "
            "reasoning (3-5 sentences), key_assumptions (list), and risk_factors (list)."
        ),
    },
}


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str = ""
    instruction: str = ""
    input: str = ""
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        return cls(**data)


# --- Prompt templates for generating training data per category ---

_GENERATION_PROMPTS: dict[str, str] = {
    "market_summarization": (
        "Generate {batch_size} diverse prediction market summarization training examples.\n"
        "Each example should have:\n"
        '- "input": a realistic prediction market question with 2-3 paragraphs of context '
        "(news snippets, polling data, market history)\n"
        '- "output": a structured summary with: market_question, key_factors (list of 3-5), '
        "current_sentiment (bullish/bearish/neutral), probability_drivers (list), "
        "risk_factors (list)\n\n"
        "Cover diverse topics: politics, crypto, sports, tech, science, legal, macro-economics.\n"
        "Vary difficulty: {difficulty_mix}.\n\n"
        "Return a JSON array of objects with keys: input, output.\n"
        "Return ONLY valid JSON, no other text."
    ),
    "entity_extraction": (
        "Generate {batch_size} entity extraction training examples from news-like text.\n"
        "Each example should have:\n"
        '- "input": a realistic news paragraph (3-5 sentences) mentioning people, '
        "organizations, events, locations, and policies\n"
        '- "output": a JSON array of entity/relationship triples: '
        '[{{"entity": "...", "type": "person|organization|event|location|policy", '
        '"relationships": [{{"target": "...", "relation_type": "..."}}]}}]\n\n'
        "Cover: geopolitics, finance, tech, legal, sports.\n"
        "Vary difficulty: {difficulty_mix}.\n\n"
        "Return a JSON array of objects with keys: input, output.\n"
        "Return ONLY valid JSON, no other text."
    ),
    "factual_qa": (
        "Generate {batch_size} factual question-answering training examples.\n"
        "Each example should have:\n"
        '- "input": a question followed by a context passage (2-4 paragraphs) that '
        "contains the answer\n"
        '- "output": a step-by-step reasoning followed by the final answer\n\n'
        "Topics: prediction markets, finance, politics, science, crypto, sports.\n"
        "Vary difficulty: {difficulty_mix}.\n\n"
        "Return a JSON array of objects with keys: input, output.\n"
        "Return ONLY valid JSON, no other text."
    ),
    "prediction_reasoning": (
        "Generate {batch_size} prediction reasoning training examples.\n"
        "Each example should have:\n"
        '- "input": a prediction market question with 2-3 evidence items '
        "(news, data points, expert opinions)\n"
        '- "output": a structured prediction with: predicted_probability (0.0-1.0), '
        "confidence_level (low/medium/high), reasoning (3-5 sentences), "
        "key_assumptions (list), risk_factors (list)\n\n"
        "Cover: elections, crypto prices, sports outcomes, policy decisions, tech milestones.\n"
        "Vary difficulty: {difficulty_mix}.\n\n"
        "Return a JSON array of objects with keys: input, output.\n"
        "Return ONLY valid JSON, no other text."
    ),
}

DIFFICULTY_MIX = "30% easy (clear-cut), 50% medium (some ambiguity), 20% hard (conflicting signals)"


class TrainingDataGenerator:
    """Generates training data for fine-tuning using Claude API."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.model = model
        self.examples: list[TrainingExample] = []

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude API for data generation."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_generated(self, raw: str) -> list[dict[str, Any]]:
        """Parse Claude's JSON response, handling markdown code blocks."""
        text = raw.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            logger.warning("training.parse_failed", raw_length=len(raw))
            return []

    def _validate_example(self, raw: dict[str, Any], category: str) -> bool:
        """Validate a generated example has required fields."""
        if not raw.get("input") or not raw.get("output"):
            return False
        if len(str(raw["input"])) < 20:
            return False
        if len(str(raw["output"])) < 20:
            return False
        # Category-specific validation
        if category == "entity_extraction":
            output = raw["output"]
            if isinstance(output, str):
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    return False
        return True

    async def generate_batch(
        self, category: str, count: int, batch_size: int = 10
    ) -> list[TrainingExample]:
        """Generate a batch of training examples for a category.

        Args:
            category: One of the 4 training categories.
            count: Total number of examples to generate.
            batch_size: Examples per API call (max ~10 for quality).

        Returns:
            List of validated TrainingExample objects.
        """
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Must be one of {list(CATEGORIES)}")

        instruction = CATEGORIES[category]["instruction"]
        prompt_template = _GENERATION_PROMPTS[category]
        examples: list[TrainingExample] = []
        attempts = 0
        max_attempts = (count // batch_size) * 3  # Allow 3x retries

        logger.info(
            "training.generate_batch.start",
            category=category,
            target_count=count,
            batch_size=batch_size,
        )

        while len(examples) < count and attempts < max_attempts:
            remaining = count - len(examples)
            current_batch = min(batch_size, remaining)

            prompt = prompt_template.format(
                batch_size=current_batch,
                difficulty_mix=DIFFICULTY_MIX,
            )

            try:
                raw = await self._call_claude(prompt)
                parsed = self._parse_generated(raw)

                for item in parsed:
                    if self._validate_example(item, category):
                        output = item["output"]
                        if isinstance(output, dict | list):
                            output = json.dumps(output)

                        example = TrainingExample(
                            category=category,
                            instruction=instruction,
                            input=str(item["input"]),
                            output=str(output),
                            metadata={
                                "source": "claude_generated",
                                "difficulty": "mixed",
                                "validated": True,
                            },
                        )
                        examples.append(example)

                logger.info(
                    "training.generate_batch.progress",
                    category=category,
                    generated=len(examples),
                    target=count,
                )

            except Exception as e:
                logger.warning(
                    "training.generate_batch.error",
                    category=category,
                    error=str(e),
                    attempt=attempts,
                )

            attempts += 1

        self.examples.extend(examples)
        logger.info(
            "training.generate_batch.complete",
            category=category,
            total=len(examples),
        )
        return examples

    def save_dataset(self, path: str | Path) -> None:
        """Save all generated examples to JSONL format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        logger.info("training.dataset_saved", path=str(path), count=len(self.examples))

    def load_dataset(self, path: str | Path) -> list[TrainingExample]:
        """Load examples from JSONL format."""
        path = Path(path)
        loaded: list[TrainingExample] = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    loaded.append(TrainingExample.from_dict(data))

        self.examples = loaded
        logger.info("training.dataset_loaded", path=str(path), count=len(loaded))
        return loaded

    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        by_category: dict[str, int] = {}
        for ex in self.examples:
            by_category[ex.category] = by_category.get(ex.category, 0) + 1
        return {
            "total": len(self.examples),
            "by_category": by_category,
        }
