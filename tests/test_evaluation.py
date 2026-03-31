"""Tests for Phase 4 evaluation pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from oracle.evaluation.calibration import CalibrationMonitor, _assign_bucket
from oracle.evaluation.gates import TradeGate
from oracle.evaluation.hallucination import HallucinationDetector, ClaimVerification
from oracle.evaluation.judge import EvaluationJudge, EvaluationResult
from oracle.evaluation.post_mortem import PostMortemGenerator
from oracle.evaluation.post_resolution import PostResolutionEvaluator, _assign_bucket as pr_bucket


# --- LLM Judge Tests ---


class TestEvaluationJudge:
    def test_parse_response_passing(self):
        """Judge should pass when all scores meet thresholds."""
        judge = EvaluationJudge(api_key="test")
        response_json = json.dumps({
            "groundedness": {"score": 8, "explanation": "Well grounded"},
            "reasoning_quality": {"score": 7, "explanation": "Solid reasoning"},
            "evidence_completeness": {"score": 6, "explanation": "Adequate evidence"},
            "calibration_alignment": {"score": 7, "explanation": "Well calibrated"},
        })
        result = judge._parse_response(response_json)
        assert result.passed is True
        assert result.scores["groundedness"] == 8
        assert result.scores["reasoning_quality"] == 7
        assert result.scores["evidence_completeness"] == 6
        assert result.overall_quality == 7.0

    def test_parse_response_failing_groundedness(self):
        """Judge should fail when groundedness < 7."""
        judge = EvaluationJudge(api_key="test")
        response_json = json.dumps({
            "groundedness": {"score": 5, "explanation": "Poorly grounded"},
            "reasoning_quality": {"score": 8, "explanation": "Good"},
            "evidence_completeness": {"score": 7, "explanation": "Good"},
            "calibration_alignment": {"score": 6, "explanation": "OK"},
        })
        result = judge._parse_response(response_json)
        assert result.passed is False

    def test_parse_response_failing_reasoning(self):
        """Judge should fail when reasoning_quality < 6."""
        judge = EvaluationJudge(api_key="test")
        response_json = json.dumps({
            "groundedness": {"score": 8, "explanation": "Good"},
            "reasoning_quality": {"score": 4, "explanation": "Weak"},
            "evidence_completeness": {"score": 7, "explanation": "Good"},
            "calibration_alignment": {"score": 6, "explanation": "OK"},
        })
        result = judge._parse_response(response_json)
        assert result.passed is False

    def test_parse_response_failing_evidence(self):
        """Judge should fail when evidence_completeness < 5."""
        judge = EvaluationJudge(api_key="test")
        response_json = json.dumps({
            "groundedness": {"score": 8, "explanation": "Good"},
            "reasoning_quality": {"score": 7, "explanation": "Good"},
            "evidence_completeness": {"score": 3, "explanation": "Lacking"},
            "calibration_alignment": {"score": 6, "explanation": "OK"},
        })
        result = judge._parse_response(response_json)
        assert result.passed is False

    def test_parse_response_with_code_block(self):
        """Should handle JSON wrapped in markdown code blocks."""
        judge = EvaluationJudge(api_key="test")
        response = '```json\n' + json.dumps({
            "groundedness": {"score": 9, "explanation": "Excellent"},
            "reasoning_quality": {"score": 8, "explanation": "Strong"},
            "evidence_completeness": {"score": 7, "explanation": "Good"},
            "calibration_alignment": {"score": 8, "explanation": "Great"},
        }) + '\n```'
        result = judge._parse_response(response)
        assert result.passed is True
        assert result.scores["groundedness"] == 9

    def test_heuristic_with_many_sources(self):
        """Heuristic should pass with enough sources."""
        judge = EvaluationJudge(api_key="")
        result = judge._evaluate_heuristic(
            "A long thesis about prediction markets that contains many words " * 10,
            ["source 1", "source 2", "source 3", "source 4"],
        )
        assert result.scores["groundedness"] == 8
        assert result.scores["evidence_completeness"] == 8

    def test_heuristic_with_few_sources(self):
        """Heuristic should fail with too few sources."""
        judge = EvaluationJudge(api_key="")
        result = judge._evaluate_heuristic("Short thesis.", ["one source"])
        assert result.scores["groundedness"] == 2
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_evaluate_falls_back_to_heuristic(self):
        """When no API key, should use heuristic."""
        judge = EvaluationJudge(api_key="")
        result = await judge.evaluate("Thesis text.", ["source 1", "source 2", "source 3", "source 4"])
        assert isinstance(result, EvaluationResult)
        assert "groundedness" in result.scores


# --- Hallucination Detection Tests ---


class TestHallucinationDetector:
    def test_extract_claims_heuristic(self):
        """Heuristic extraction should split on sentences."""
        detector = HallucinationDetector(api_key="")
        claims = detector._extract_claims_heuristic(
            "The market is trading at 0.65. Recent polls show a 10-point lead. "
            "Historical data supports this trend."
        )
        assert len(claims) >= 2
        assert all(len(c.split()) >= 4 for c in claims)

    def test_verify_claim_grounded(self):
        """Claim matching a source should be verified."""
        detector = HallucinationDetector(api_key="")

        # Mock embedding service to return high similarity
        mock_service = MagicMock()
        # Claim embedding
        mock_service.embed.side_effect = [
            [[1.0, 0.0, 0.0]],  # First call: claim
            [[0.98, 0.1, 0.0], [0.1, 0.9, 0.0]],  # Second call: sources
        ]
        detector._embedding_service = mock_service

        result = detector.verify_claim(
            "The market price is 0.65",
            ["The current market price stands at 0.65", "Unrelated source text"],
        )
        # With mocked embeddings: dot product of [1,0,0] and [0.98,0.1,0] ≈ 0.98
        assert result.verified is True
        assert result.confidence > 0.75

    def test_verify_claim_ungrounded(self):
        """Claim not matching any source should be unverified."""
        detector = HallucinationDetector(api_key="")

        mock_service = MagicMock()
        mock_service.embed.side_effect = [
            [[1.0, 0.0, 0.0]],  # Claim
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Sources (orthogonal)
        ]
        detector._embedding_service = mock_service

        result = detector.verify_claim(
            "A completely fabricated claim",
            ["Actual source about something else", "Another unrelated source"],
        )
        assert result.verified is False
        assert result.confidence < 0.75

    def test_verify_claim_no_sources(self):
        """Claim with no sources should be unverified."""
        detector = HallucinationDetector(api_key="")
        result = detector.verify_claim("Any claim", [])
        assert result.verified is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_detect_full_pipeline(self):
        """Full detection pipeline with mixed grounded/ungrounded claims."""
        detector = HallucinationDetector(api_key="")

        # Mock extract_claims to return known claims
        detector.extract_claims = AsyncMock(return_value=[
            "Claim that matches source",
            "Fabricated claim with no basis",
        ])

        # Mock verify_claim to return controlled results
        original_verify = detector.verify_claim
        call_idx = [0]
        verifications = [
            ClaimVerification(
                claim="Claim that matches source",
                verified=True, confidence=0.85, matching_source="source 1",
            ),
            ClaimVerification(
                claim="Fabricated claim with no basis",
                verified=False, confidence=0.3, matching_source=None,
            ),
        ]
        def mock_verify(claim, sources):
            result = verifications[call_idx[0]]
            call_idx[0] += 1
            return result
        detector.verify_claim = mock_verify

        result = await detector.detect("Thesis text", ["source 1", "source 2"])

        assert len(result.grounded_claims) == 1
        assert len(result.ungrounded_claims) == 1
        assert result.hallucination_rate == pytest.approx(0.5)


# --- Brier Score Tests ---


class TestPostResolutionEvaluator:
    @pytest.mark.asyncio
    async def test_brier_score_perfect_prediction(self, tmp_path):
        """Brier score should be 0.0 for a perfect prediction."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        result = await evaluator.evaluate_prediction(
            trade_id="t1", market_id="m1",
            predicted_prob=1.0, predicted_direction="yes",
            actual_outcome=True, actual_final_price=0.95,
        )
        assert result.brier_score == pytest.approx(0.0)
        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_brier_score_worst_prediction(self, tmp_path):
        """Brier score should be 1.0 for completely wrong prediction."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        result = await evaluator.evaluate_prediction(
            trade_id="t2", market_id="m2",
            predicted_prob=1.0, predicted_direction="yes",
            actual_outcome=False, actual_final_price=0.05,
        )
        assert result.brier_score == pytest.approx(1.0)
        assert result.is_correct is False

    @pytest.mark.asyncio
    async def test_brier_score_moderate(self, tmp_path):
        """Brier score for a 0.7 prediction on YES outcome should be 0.09."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        result = await evaluator.evaluate_prediction(
            trade_id="t3", market_id="m3",
            predicted_prob=0.7, predicted_direction="yes",
            actual_outcome=True, actual_final_price=0.95,
        )
        assert result.brier_score == pytest.approx(0.09, abs=0.001)
        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_alpha_detection(self, tmp_path):
        """Should detect alpha when diverging >10% from consensus and correct."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        result = await evaluator.evaluate_prediction(
            trade_id="t4", market_id="m4",
            predicted_prob=0.8, predicted_direction="yes",
            actual_outcome=True, actual_final_price=0.95,
            market_consensus=0.55,
        )
        assert result.generated_alpha is True
        assert result.divergence_from_market > 0.10

    @pytest.mark.asyncio
    async def test_no_alpha_when_wrong(self, tmp_path):
        """Should not detect alpha when prediction is wrong."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        result = await evaluator.evaluate_prediction(
            trade_id="t5", market_id="m5",
            predicted_prob=0.8, predicted_direction="yes",
            actual_outcome=False, actual_final_price=0.05,
            market_consensus=0.55,
        )
        assert result.generated_alpha is False

    @pytest.mark.asyncio
    async def test_aggregate_stats(self, tmp_path):
        """Aggregate stats should compute across all predictions."""
        db_path = str(tmp_path / "test.db")
        evaluator = PostResolutionEvaluator(db_path=db_path)
        await evaluator.initialize()

        # Add several predictions
        await evaluator.evaluate_prediction(
            trade_id="t1", market_id="m1",
            predicted_prob=0.8, predicted_direction="yes",
            actual_outcome=True, actual_final_price=0.95,
            category="politics",
        )
        await evaluator.evaluate_prediction(
            trade_id="t2", market_id="m2",
            predicted_prob=0.6, predicted_direction="yes",
            actual_outcome=False, actual_final_price=0.1,
            category="politics",
        )

        stats = await evaluator.aggregate_stats()
        assert stats["total_predictions"] == 2
        assert 0.0 <= stats["overall_accuracy"] <= 1.0
        assert stats["brier_score"] > 0
        assert "politics" in stats["by_category"]


# --- Calibration Bucket Tests ---


class TestCalibrationMonitor:
    def test_assign_bucket_50_60(self):
        assert _assign_bucket(0.55) == "50-60%"
        assert _assign_bucket(0.50) == "50-60%"
        assert _assign_bucket(0.59) == "50-60%"

    def test_assign_bucket_60_70(self):
        assert _assign_bucket(0.60) == "60-70%"
        assert _assign_bucket(0.65) == "60-70%"

    def test_assign_bucket_70_80(self):
        assert _assign_bucket(0.70) == "70-80%"
        assert _assign_bucket(0.79) == "70-80%"

    def test_assign_bucket_80_90(self):
        assert _assign_bucket(0.80) == "80-90%"
        assert _assign_bucket(0.89) == "80-90%"

    def test_assign_bucket_90_100(self):
        assert _assign_bucket(0.90) == "90-100%"
        assert _assign_bucket(0.99) == "90-100%"
        assert _assign_bucket(1.0) == "90-100%"

    @pytest.mark.asyncio
    async def test_record_and_compute(self, tmp_path):
        """Record predictions + outcomes and compute calibration."""
        db_path = str(tmp_path / "test.db")
        monitor = CalibrationMonitor(db_path=db_path)
        await monitor.initialize()

        # Record predictions in the 70-80% bucket
        await monitor.record_prediction("m1", 0.75)
        await monitor.record_prediction("m2", 0.72)
        await monitor.record_prediction("m3", 0.78)

        # Resolve: 2 out of 3 correct → actual rate = 0.667
        await monitor.record_outcome("m1", True)
        await monitor.record_outcome("m2", True)
        await monitor.record_outcome("m3", False)

        cal = await monitor.compute_calibration()
        bucket_70 = next(b for b in cal.buckets if b.range == "70-80%")
        assert bucket_70.count == 3
        assert bucket_70.actual_rate == pytest.approx(2 / 3, abs=0.01)
        assert bucket_70.predicted_avg == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_chart_data_format(self, tmp_path):
        """Chart data should have expected keys."""
        db_path = str(tmp_path / "test.db")
        monitor = CalibrationMonitor(db_path=db_path)
        await monitor.initialize()

        data = await monitor.get_calibration_chart_data()
        assert "labels" in data
        assert "predicted_avg" in data
        assert "actual_rate" in data
        assert "ideal" in data
        assert "counts" in data
        assert "calibration_error" in data
        assert "is_well_calibrated" in data
        assert len(data["labels"]) == 5


# --- Gate Blocking Tests ---


class TestTradeGate:
    @pytest.mark.asyncio
    async def test_gate_blocks_low_groundedness(self):
        """Gate should block when groundedness < 7."""
        gate = TradeGate(api_key="")
        # With empty API key, heuristic kicks in. 1 source → groundedness=2
        result = await gate.evaluate_trade_proposal("Short thesis", ["one source"])
        assert result.approved is False
        assert any("Groundedness" in r for r in result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_gate_blocks_low_reasoning(self):
        """Gate should block when reasoning_quality < 6."""
        gate = TradeGate(api_key="")
        # Short thesis → low reasoning score
        result = await gate.evaluate_trade_proposal("Short.", ["s1", "s2", "s3", "s4"])
        assert result.approved is False
        assert any("Reasoning" in r for r in result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_gate_blocks_low_evidence(self):
        """Gate should block when evidence_completeness < 5."""
        gate = TradeGate(api_key="")
        result = await gate.evaluate_trade_proposal("Some thesis text", ["only one source"])
        assert result.approved is False
        assert any("Evidence" in r or "Groundedness" in r for r in result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_gate_blocks_many_ungrounded_claims(self):
        """Gate should block when >1 ungrounded claims."""
        gate = TradeGate(api_key="")

        # Mock the detector to return many ungrounded claims
        from oracle.evaluation.hallucination import HallucinationResult
        mock_result = HallucinationResult(
            ungrounded_claims=[
                ClaimVerification(claim="fake1", verified=False, confidence=0.2),
                ClaimVerification(claim="fake2", verified=False, confidence=0.1),
                ClaimVerification(claim="fake3", verified=False, confidence=0.15),
            ],
            hallucination_rate=1.0,
        )
        gate._detector.detect = AsyncMock(return_value=mock_result)

        # Also mock judge to pass so we only test hallucination blocking
        from oracle.evaluation.judge import EvaluationResult
        mock_eval = EvaluationResult(
            passed=True,
            scores={"groundedness": 8, "reasoning_quality": 8,
                    "evidence_completeness": 8, "calibration_alignment": 8},
            overall_quality=8.0,
        )
        gate._judge.evaluate = AsyncMock(return_value=mock_eval)

        result = await gate.evaluate_trade_proposal("Thesis", ["source"])
        assert result.approved is False
        assert any("Ungrounded" in r for r in result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_gate_approves_good_trade(self):
        """Gate should approve when all checks pass."""
        gate = TradeGate(api_key="")

        from oracle.evaluation.hallucination import HallucinationResult
        from oracle.evaluation.judge import EvaluationResult

        mock_eval = EvaluationResult(
            passed=True,
            scores={"groundedness": 9, "reasoning_quality": 8,
                    "evidence_completeness": 7, "calibration_alignment": 8},
            overall_quality=8.0,
        )
        gate._judge.evaluate = AsyncMock(return_value=mock_eval)

        mock_halluc = HallucinationResult(
            grounded_claims=[
                ClaimVerification(claim="good1", verified=True, confidence=0.9),
            ],
            ungrounded_claims=[],
            hallucination_rate=0.0,
        )
        gate._detector.detect = AsyncMock(return_value=mock_halluc)

        result = await gate.evaluate_trade_proposal("Good thesis", ["source 1"])
        assert result.approved is True
        assert result.blocking_reasons == []


# --- Post-Mortem Tests ---


class TestPostMortemGenerator:
    @pytest.mark.asyncio
    async def test_heuristic_correct_with_evidence(self, tmp_path):
        """Correct prediction with good evidence → good process."""
        db_path = str(tmp_path / "test.db")
        gen = PostMortemGenerator(db_path=db_path, api_key="")
        await gen.initialize()

        pm = await gen.generate(
            trade_id="t1",
            original_thesis="A detailed thesis with many words and evidence to support the claim " * 8,
            actual_outcome=True,
        )
        assert pm.was_correct is True
        assert pm.process_quality == "good"
        assert pm.good_luck_factor < 0.5

    @pytest.mark.asyncio
    async def test_heuristic_correct_without_evidence(self, tmp_path):
        """Correct prediction with thin evidence → bad process, got lucky."""
        db_path = str(tmp_path / "test.db")
        gen = PostMortemGenerator(db_path=db_path, api_key="")
        await gen.initialize()

        pm = await gen.generate(
            trade_id="t2",
            original_thesis="Short thesis",
            actual_outcome=True,
        )
        assert pm.was_correct is True
        assert pm.process_quality == "bad"
        assert pm.good_luck_factor > 0.5

    @pytest.mark.asyncio
    async def test_retrieve_post_mortem(self, tmp_path):
        """Should be able to retrieve a stored post-mortem."""
        db_path = str(tmp_path / "test.db")
        gen = PostMortemGenerator(db_path=db_path, api_key="")
        await gen.initialize()

        await gen.generate(
            trade_id="t1",
            original_thesis="A detailed thesis with many words and evidence to support the claim " * 8,
            actual_outcome=True,
        )

        retrieved = await gen.get_post_mortem("t1")
        assert retrieved is not None
        assert retrieved.trade_id == "t1"
        assert retrieved.was_correct is True

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, tmp_path):
        """Should return None for non-existent trade_id."""
        db_path = str(tmp_path / "test.db")
        gen = PostMortemGenerator(db_path=db_path, api_key="")
        await gen.initialize()

        result = await gen.get_post_mortem("nonexistent")
        assert result is None
