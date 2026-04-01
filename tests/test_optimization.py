"""Tests for Phase 5 — fine-tuning & cost optimization components."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Model Routing ──────────────────────────────────────────────────────


class TestComplexityClassifier:
    """Test the complexity-based model routing classifier."""

    def test_classifier_initializes(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier

        model_path = str(tmp_path / "classifier.pkl")
        clf = ComplexityClassifier(model_path=model_path)
        assert clf._model is not None
        assert Path(model_path).exists()

    def test_classify_simple_query_routes_local(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier

        clf = ComplexityClassifier(model_path=str(tmp_path / "clf.pkl"))
        decision = clf.classify("What is Bitcoin price?", {"category": "crypto"})
        assert decision.model in ("local", "claude")
        assert 0.0 <= decision.confidence <= 1.0
        assert "query_length" in decision.features

    def test_classify_complex_query_routes_claude(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier

        clf = ComplexityClassifier(model_path=str(tmp_path / "clf.pkl"))
        complex_query = (
            "Given that the Federal Reserve has signaled rate cuts, therefore implying "
            "a shift in monetary policy, and because the ECB is also considering easing, "
            "analyze the implications for crypto markets across multiple jurisdictions "
            "including the regulatory landscape in the EU, US, and Asia"
        )
        decision = clf.classify(
            complex_query,
            {
                "category": "macro",
                "sources": ["fed.gov", "ecb.europa.eu", "reuters.com"],
                "evidence": [{"a": 1}, {"b": 2}, {"c": 3}],
                "confidence_threshold": 0.95,
            },
        )
        # Complex query should route to Claude
        assert decision.model == "claude"

    def test_routing_stats_target_80_local(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier

        clf = ComplexityClassifier(model_path=str(tmp_path / "clf.pkl"))
        stats = clf.routing_stats
        assert stats["local_pct"] >= 60  # At least 60% local
        assert stats["local_pct"] + stats["claude_pct"] == pytest.approx(100, abs=1)

    def test_extract_features(self):
        from oracle.routing.classifier import extract_features

        features = extract_features(
            "Will Bitcoin reach 100k because of ETF inflows?",
            {"category": "crypto", "sources": ["a", "b"], "confidence_threshold": 0.8},
        )
        assert features["query_length"] > 0
        assert features["has_multi_step_reasoning"] == 1.0  # "because"
        assert features["market_category_complexity"] == 2.0  # crypto
        assert features["requires_synthesis"] == 1.0  # 2 sources
        assert features["confidence_threshold"] == 0.8

    def test_load_existing_model(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier

        model_path = str(tmp_path / "clf.pkl")
        clf1 = ComplexityClassifier(model_path=model_path)
        # Load again — should use existing file
        clf2 = ComplexityClassifier(model_path=model_path)
        decision1 = clf1.classify("test query")
        decision2 = clf2.classify("test query")
        assert decision1.model == decision2.model


class TestModelRouter:
    """Test the model router."""

    @pytest.mark.asyncio
    async def test_router_tracks_stats(self, tmp_path):
        from oracle.routing.classifier import ComplexityClassifier, ModelRouter

        clf = ComplexityClassifier(model_path=str(tmp_path / "clf.pkl"))
        router = ModelRouter(classifier=clf)
        result = await router.route("Simple question")
        assert result["model"] in ("local", "claude")
        assert router.stats["total_calls"] == 1


# ── Semantic Cache ─────────────────────────────────────────────────────


class TestSemanticCache:
    """Test semantic cache with mocked Qdrant."""

    def _make_cache(self):
        from oracle.cache.semantic_cache import SemanticCache

        cache = SemanticCache(ttl_seconds=3600)
        cache._client = MagicMock()
        cache._embedder = MagicMock()
        cache._embedder.embed_query.return_value = [0.1] * 1024
        return cache

    def test_cache_miss(self):
        cache = self._make_cache()
        cache._client.search.return_value = []
        result = cache.lookup("test query")
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_cache_hit(self):
        cache = self._make_cache()

        mock_point = MagicMock()
        mock_point.score = 0.98
        mock_point.payload = {
            "query": "test query",
            "result": "cached answer",
            "entity_ids": ["e1"],
            "market_ids": ["m1"],
            "timestamp": time.time(),
        }
        cache._client.search.return_value = [mock_point]

        result = cache.lookup("test query")
        assert result is not None
        assert result.result == "cached answer"
        assert result.score == 0.98
        assert cache.stats.hits == 1

    def test_cache_store(self):
        cache = self._make_cache()
        point_id = cache.store("query", "result", entity_ids=["e1"], market_ids=["m1"])
        assert point_id  # non-empty string
        cache._client.upsert.assert_called_once()

    def test_ttl_expiry(self):
        """Expired entries should not be returned (filter applied in Qdrant query)."""
        cache = self._make_cache()
        # Search returns empty because of TTL filter
        cache._client.search.return_value = []
        result = cache.lookup("expired query")
        assert result is None
        assert cache.stats.misses == 1

    def test_invalidate_by_entity(self):
        cache = self._make_cache()
        cache.invalidate_by_entity("entity-123")
        cache._client.delete.assert_called_once()

    def test_invalidate_by_market(self):
        cache = self._make_cache()
        cache.invalidate_by_market("market-456")
        cache._client.delete.assert_called_once()

    def test_cache_stats(self):
        from oracle.cache.semantic_cache import CacheStats

        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == pytest.approx(0.7)
        assert stats.total_queries == 10
        d = stats.to_dict()
        assert d["hit_rate"] == 0.7

    def test_get_stats_with_entries(self):
        cache = self._make_cache()
        mock_info = MagicMock()
        mock_info.points_count = 42
        cache._client.get_collection.return_value = mock_info
        stats = cache.get_stats()
        assert stats["total_entries"] == 42


# ── Prompt Registry ────────────────────────────────────────────────────


class TestPromptRegistry:
    """Test the SQLite-backed prompt registry."""

    def _make_registry(self, tmp_path):
        from oracle.prompts.registry import PromptRegistry

        db_path = str(tmp_path / "test_prompts.db")
        registry = PromptRegistry(db_path=db_path)
        registry.initialize()
        return registry

    def test_seed_templates(self, tmp_path):
        registry = self._make_registry(tmp_path)
        templates = registry.list_active()
        assert len(templates) == 9
        names = {t.name for t in templates}
        assert "research_plan" in names
        assert "post_mortem" in names

    def test_register_new_version(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register(
            name="research_plan",
            template="New template: {market_question}",
            variables=["market_question"],
            description="Updated research plan",
        )
        tpl = registry.get("research_plan")
        assert tpl is not None
        assert tpl.version == 2
        assert "New template" in tpl.template

    def test_render(self, tmp_path):
        registry = self._make_registry(tmp_path)
        rendered = registry.render(
            "research_plan",
            market_question="Will BTC hit 100k?",
            context="Recent ETF approvals",
        )
        assert "Will BTC hit 100k?" in rendered
        assert "Recent ETF approvals" in rendered

    def test_render_missing_template_raises(self, tmp_path):
        registry = self._make_registry(tmp_path)
        with pytest.raises(KeyError):
            registry.render("nonexistent_template")

    def test_version_history(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("research_plan", "v2: {market_question}", ["market_question"])
        registry.register("research_plan", "v3: {market_question}", ["market_question"])
        versions = registry.get_versions("research_plan")
        assert len(versions) == 3
        assert versions[0].version == 1
        assert versions[2].version == 3
        # Only latest is active
        active = [v for v in versions if v.is_active]
        assert len(active) == 1
        assert active[0].version == 3

    def test_set_active_version(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("research_plan", "v2: {market_question}", ["market_question"])
        registry.set_active_version("research_plan", 1)
        tpl = registry.get("research_plan")
        assert tpl is not None
        assert tpl.version == 1


# ── Token Budgeting ────────────────────────────────────────────────────


class TestTokenBudget:
    """Test token counting and trimming strategies."""

    def test_count_tokens(self):
        from oracle.utils.token_budget import count_tokens

        count = count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_within_budget_no_change(self):
        from oracle.utils.token_budget import trim_to_budget

        texts = ["Hello", "World"]
        result = trim_to_budget(texts, budget=100)
        assert result == texts

    def test_truncate_last_strategy(self):
        from oracle.utils.token_budget import count_tokens, trim_to_budget

        short = "Short text."
        long_text = "This is a much longer text that contains many tokens. " * 20
        texts = [short, long_text]
        result = trim_to_budget(texts, budget=50, strategy="truncate_last")
        # First item should be preserved, last trimmed
        assert result[0] == short
        total = sum(count_tokens(t) for t in result)
        assert total <= 50

    def test_proportional_strategy(self):
        from oracle.utils.token_budget import count_tokens, trim_to_budget

        texts = ["A " * 50, "B " * 50]
        result = trim_to_budget(texts, budget=50, strategy="proportional")
        total = sum(count_tokens(t) for t in result)
        assert total <= 50

    def test_priority_strategy(self):
        from oracle.utils.token_budget import count_tokens, trim_to_budget

        texts = ["Important content. " * 10, "Less important. " * 10, "Least important. " * 10]
        weights = [3.0, 2.0, 1.0]
        result = trim_to_budget(
            texts, budget=40, strategy="priority", weights=weights
        )
        # Highest priority should be most preserved
        assert count_tokens(result[0]) >= count_tokens(result[2])

    def test_wrap_with_budget(self):
        from oracle.utils.token_budget import count_tokens, wrap_with_budget

        parts = {
            "system": "You are an analyst.",
            "context": "Some context " * 50,
            "question": "What is the answer?",
        }
        result = wrap_with_budget(parts, budget=100)
        assert count_tokens(result) <= 100
        assert "You are an analyst." in result

    def test_budget_aware_decorator(self):
        from oracle.utils.token_budget import budget_aware

        @budget_aware(max_tokens=50)
        def process(text: str = "") -> str:
            return text

        long_text = "word " * 200
        result = process(text=long_text)
        # The text kwarg should have been trimmed
        assert isinstance(result, str)

    def test_empty_input(self):
        from oracle.utils.token_budget import trim_to_budget

        assert trim_to_budget([], budget=100) == []


# ── A/B Testing ────────────────────────────────────────────────────────


class TestABTesting:
    """Test the A/B testing framework."""

    def _make_manager(self, tmp_path):
        from oracle.prompts.ab_testing import ABTestManager
        from oracle.prompts.registry import PromptRegistry

        db_path = str(tmp_path / "test_ab.db")
        # Initialize registry first (for promote_winner)
        registry = PromptRegistry(db_path=db_path)
        registry.initialize()

        manager = ABTestManager(db_path=db_path)
        manager.initialize()
        return manager

    def test_create_test(self, tmp_path):
        manager = self._make_manager(tmp_path)
        test_id = manager.create_test("research_plan", 1, 2)
        assert test_id > 0

    def test_assign_variant_distribution(self, tmp_path):
        manager = self._make_manager(tmp_path)
        test_id = manager.create_test("research_plan", 1, 2)

        counts = {"A": 0, "B": 0}
        for i in range(100):
            variant = manager.assign_variant(test_id, f"pred_{i}")
            counts[variant] += 1

        # Should be roughly 50/50 (allow wide margin for randomness)
        assert counts["A"] > 20
        assert counts["B"] > 20

    def test_inconclusive_with_few_samples(self, tmp_path):
        manager = self._make_manager(tmp_path)
        test_id = manager.create_test("research_plan", 1, 2)

        # Add only a few samples
        for i in range(5):
            variant = manager.assign_variant(test_id, f"pred_{i}")
            manager.record_result(test_id, f"pred_{i}", 0.3 if variant == "A" else 0.5)

        result = manager.analyze(test_id)
        assert result.winner == "inconclusive"
        assert result.p_value == 1.0

    def test_winner_detection(self, tmp_path):
        import random

        manager = self._make_manager(tmp_path)
        test_id = manager.create_test("research_plan", 1, 2)

        rng = random.Random(42)
        # Variant A clearly better (lower Brier score)
        for i in range(60):
            pid = f"pred_{i}"
            # Force alternating assignment for balanced test
            variant = "A" if i % 2 == 0 else "B"
            # Override random assignment
            conn = manager._get_conn()
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO ab_assignments (test_id, prediction_id, variant, assigned_at) "
                "VALUES (?, ?, ?, ?)",
                (test_id, pid, variant, now),
            )
            conn.commit()

            if variant == "A":
                value = 0.1 + rng.uniform(-0.02, 0.02)  # Low Brier = good
            else:
                value = 0.5 + rng.uniform(-0.02, 0.02)  # High Brier = bad
            manager.record_result(test_id, pid, value)

        result = manager.analyze(test_id)
        assert result.winner == "A"
        assert result.p_value < 0.05
        assert result.sample_sizes["A"] >= 30
        assert result.sample_sizes["B"] >= 30

    def test_promote_winner(self, tmp_path):
        import random

        from oracle.prompts.registry import PromptRegistry

        manager = self._make_manager(tmp_path)
        db_path = str(tmp_path / "test_ab.db")

        # Register v2 of research_plan
        registry = PromptRegistry(db_path=db_path)
        registry.initialize()
        registry.register("research_plan", "v2: {market_question}", ["market_question"])

        test_id = manager.create_test("research_plan", 1, 2)

        rng = random.Random(42)
        conn = manager._get_conn()
        from datetime import datetime, timezone

        for i in range(60):
            pid = f"pred_{i}"
            variant = "A" if i % 2 == 0 else "B"
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO ab_assignments (test_id, prediction_id, variant, assigned_at) "
                "VALUES (?, ?, ?, ?)",
                (test_id, pid, variant, now),
            )
            conn.commit()
            value = 0.1 + rng.uniform(-0.02, 0.02) if variant == "A" else 0.5 + rng.uniform(-0.02, 0.02)
            manager.record_result(test_id, pid, value)

        winner = manager.promote_winner(test_id)
        assert winner == "A"

        # Check registry was updated
        tpl = registry.get("research_plan")
        assert tpl is not None
        assert tpl.version == 1  # A = version 1

    def test_list_active_tests(self, tmp_path):
        manager = self._make_manager(tmp_path)
        manager.create_test("test1", 1, 2)
        manager.create_test("test2", 1, 3)
        active = manager.list_active_tests()
        assert len(active) == 2


# ── Training Data Generator ───────────────────────────────────────────


class TestTrainingDataGenerator:
    """Test training data generation and JSONL I/O."""

    def test_training_example_serialization(self):
        from oracle.training.data_generator import TrainingExample

        ex = TrainingExample(
            category="factual_qa",
            instruction="Answer the question.",
            input="What is 2+2?",
            output="4",
            metadata={"source": "test", "difficulty": "easy", "validated": True},
        )
        d = ex.to_dict()
        assert d["category"] == "factual_qa"
        restored = TrainingExample.from_dict(d)
        assert restored.input == "What is 2+2?"

    def test_save_load_dataset(self, tmp_path):
        from oracle.training.data_generator import TrainingDataGenerator, TrainingExample

        gen = TrainingDataGenerator()
        gen.examples = [
            TrainingExample(
                category="factual_qa",
                instruction="test",
                input=f"input_{i}",
                output=f"output_{i}",
                metadata={"source": "test", "difficulty": "easy", "validated": True},
            )
            for i in range(10)
        ]

        path = tmp_path / "test_data.jsonl"
        gen.save_dataset(path)
        assert path.exists()

        gen2 = TrainingDataGenerator()
        loaded = gen2.load_dataset(path)
        assert len(loaded) == 10
        assert loaded[0].input == "input_0"

    def test_stats(self):
        from oracle.training.data_generator import TrainingDataGenerator, TrainingExample

        gen = TrainingDataGenerator()
        gen.examples = [
            TrainingExample(category="factual_qa", instruction="", input="a", output="b"),
            TrainingExample(category="factual_qa", instruction="", input="c", output="d"),
            TrainingExample(category="entity_extraction", instruction="", input="e", output="f"),
        ]
        stats = gen.stats()
        assert stats["total"] == 3
        assert stats["by_category"]["factual_qa"] == 2
        assert stats["by_category"]["entity_extraction"] == 1

    def test_categories_defined(self):
        from oracle.training.data_generator import CATEGORIES

        assert len(CATEGORIES) == 4
        total = sum(c["count"] for c in CATEGORIES.values())
        assert total == 18000

    def test_validate_example(self):
        from oracle.training.data_generator import TrainingDataGenerator

        gen = TrainingDataGenerator()
        assert gen._validate_example(
            {"input": "A valid input with enough text", "output": "A valid output with enough text"},
            "factual_qa",
        )
        assert not gen._validate_example({"input": "", "output": "something"}, "factual_qa")
        assert not gen._validate_example({"input": "short", "output": "x"}, "factual_qa")
