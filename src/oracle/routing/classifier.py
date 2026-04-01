"""Model routing classifier — routes queries to local model or Claude based on complexity."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression

from oracle.config import settings

logger = structlog.get_logger()

# Complexity mappings for market categories
CATEGORY_COMPLEXITY: dict[str, int] = {
    "sports": 1,
    "crypto": 2,
    "politics": 3,
    "macro": 4,
    "legal": 5,
}

MULTI_STEP_MARKERS = {"because", "therefore", "given that", "implies", "consequently", "thus"}

DEFAULT_MODEL_PATH = "data/routing/classifier.pkl"


@dataclass
class RoutingDecision:
    """Result of the complexity classifier."""

    model: str  # "local" or "claude"
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)


def extract_features(query: str, context: dict[str, Any] | None = None) -> dict[str, float]:
    """Extract complexity features from a query and optional context.

    Features:
        query_length: Approximate token count (words).
        entity_count: Number of entities detected via spaCy NER.
        has_multi_step_reasoning: Whether reasoning markers are present.
        market_category_complexity: 1-5 scale based on category.
        requires_synthesis: Whether multiple sources are referenced.
        confidence_threshold: How certain the answer needs to be.
    """
    context = context or {}

    # query_length — approximate token count
    query_length = len(query.split())

    # entity_count — try spaCy, fall back to simple heuristic
    entity_count = 0
    try:
        import spacy

        nlp = spacy.blank("en")
        if not nlp.has_pipe("ner"):
            # Use a lightweight approach: count capitalized multi-word sequences
            words = query.split()
            entity_count = sum(1 for w in words if w and w[0].isupper() and w.isalpha())
        else:
            doc = nlp(query)
            entity_count = len(doc.ents)
    except Exception:
        words = query.split()
        entity_count = sum(1 for w in words if w and w[0].isupper() and w.isalpha())

    # has_multi_step_reasoning
    query_lower = query.lower()
    has_multi_step = float(any(marker in query_lower for marker in MULTI_STEP_MARKERS))

    # market_category_complexity
    category = context.get("category", "other")
    category_complexity = CATEGORY_COMPLEXITY.get(category, 3)

    # requires_synthesis
    source_count = len(context.get("sources", []))
    evidence_count = len(context.get("evidence", []))
    requires_synthesis = float(source_count > 1 or evidence_count > 2)

    # confidence_threshold
    confidence_threshold = context.get("confidence_threshold", 0.7)

    return {
        "query_length": float(query_length),
        "entity_count": float(entity_count),
        "has_multi_step_reasoning": has_multi_step,
        "market_category_complexity": float(category_complexity),
        "requires_synthesis": requires_synthesis,
        "confidence_threshold": float(confidence_threshold),
    }


def _features_to_array(features: dict[str, float]) -> list[float]:
    """Convert feature dict to ordered array for sklearn."""
    return [
        features["query_length"],
        features["entity_count"],
        features["has_multi_step_reasoning"],
        features["market_category_complexity"],
        features["requires_synthesis"],
        features["confidence_threshold"],
    ]


def _generate_synthetic_training_data(n: int = 500) -> tuple[list[list[float]], list[int]]:
    """Generate synthetic labeled data for training the routing classifier.

    Label 0 = local, Label 1 = claude.
    Target distribution: ~80% local, ~20% Claude.
    """
    rng = random.Random(42)
    X: list[list[float]] = []
    y: list[int] = []

    for _ in range(n):
        query_length = rng.randint(3, 200)
        entity_count = rng.randint(0, 15)
        has_multi_step = float(rng.random() < 0.25)
        category_complexity = rng.randint(1, 5)
        requires_synthesis = float(rng.random() < 0.3)
        confidence_threshold = round(rng.uniform(0.5, 0.99), 2)

        features = [
            float(query_length),
            float(entity_count),
            has_multi_step,
            float(category_complexity),
            requires_synthesis,
            confidence_threshold,
        ]

        # Complexity score determines routing
        complexity = (
            (query_length / 200) * 0.15
            + (entity_count / 15) * 0.15
            + has_multi_step * 0.25
            + (category_complexity / 5) * 0.2
            + requires_synthesis * 0.15
            + confidence_threshold * 0.1
        )

        # ~80% local: only route to Claude for high complexity
        label = 1 if complexity > 0.55 else 0

        X.append(features)
        y.append(label)

    return X, y


class ComplexityClassifier:
    """Logistic regression classifier for routing queries to local or Claude model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model: LogisticRegression | None = None
        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load existing model or train on synthetic data."""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("routing.classifier_loaded", path=str(self.model_path))
        else:
            self._train_on_synthetic()

    def _train_on_synthetic(self) -> None:
        """Train on synthetic data and save the model."""
        X, y = _generate_synthetic_training_data(500)
        self._model = LogisticRegression(random_state=42, max_iter=1000)
        self._model.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)

        local_pct = (1 - np.mean(y)) * 100
        logger.info(
            "routing.classifier_trained",
            samples=len(y),
            local_pct=round(local_pct, 1),
            path=str(self.model_path),
        )

    def classify(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Classify a query as requiring local model or Claude.

        Args:
            query: The input query text.
            context: Optional context with category, sources, evidence, confidence_threshold.

        Returns:
            RoutingDecision with model choice, confidence, and features.
        """
        features = extract_features(query, context)
        feature_array = np.array([_features_to_array(features)])

        probas = self._model.predict_proba(feature_array)[0]  # type: ignore[union-attr]
        prediction = int(self._model.predict(feature_array)[0])  # type: ignore[union-attr]

        model_choice = "claude" if prediction == 1 else "local"
        confidence = float(max(probas))

        return RoutingDecision(
            model=model_choice,
            confidence=confidence,
            features=features,
        )

    @property
    def routing_stats(self) -> dict[str, Any]:
        """Get routing statistics from synthetic training data distribution."""
        X, y = _generate_synthetic_training_data(500)
        predictions = self._model.predict(X)  # type: ignore[union-attr]
        local_count = int(np.sum(predictions == 0))
        claude_count = int(np.sum(predictions == 1))
        total = len(predictions)
        return {
            "local_pct": round(local_count / total * 100, 1),
            "claude_pct": round(claude_count / total * 100, 1),
            "total_evaluated": total,
        }


class ModelRouter:
    """Routes queries to local model stub or Claude API based on classifier decision."""

    def __init__(self, classifier: ComplexityClassifier | None = None) -> None:
        self.classifier = classifier or ComplexityClassifier()
        self._local_calls = 0
        self._claude_calls = 0

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route a query and return the response.

        Args:
            query: The input query.
            context: Optional context for classification.

        Returns:
            Dict with model used, response, and routing decision.
        """
        decision = self.classifier.classify(query, context)

        if decision.model == "local":
            response = await self._call_local(query)
            self._local_calls += 1
        else:
            response = await self._call_claude(query)
            self._claude_calls += 1

        logger.info(
            "routing.routed",
            model=decision.model,
            confidence=round(decision.confidence, 3),
        )

        return {
            "model": decision.model,
            "response": response,
            "routing": {
                "confidence": decision.confidence,
                "features": decision.features,
            },
        }

    async def _call_local(self, query: str) -> str:
        """Local model stub — returns a simple heuristic response."""
        return f"[local] Processed query ({len(query.split())} tokens): {query[:200]}"

    async def _call_claude(self, query: str) -> str:
        """Call Claude API for complex queries."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning("routing.claude_fallback", error=str(e))
            return await self._call_local(query)

    @property
    def stats(self) -> dict[str, Any]:
        """Runtime routing statistics."""
        total = self._local_calls + self._claude_calls
        return {
            "local_calls": self._local_calls,
            "claude_calls": self._claude_calls,
            "total_calls": total,
            "local_pct": round(self._local_calls / total * 100, 1) if total else 0,
            "claude_pct": round(self._claude_calls / total * 100, 1) if total else 0,
        }
