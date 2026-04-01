"""Backtest metrics aggregator — computes all performance metrics from prediction records."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionRecord:
    """Single resolved prediction for backtest analysis."""

    trade_id: str
    predicted_prob: float        # Oracle's yes-probability
    market_prob_at_entry: float  # Market consensus at time of entry
    actual_outcome: bool         # True = resolved YES
    is_correct: bool
    brier_score: float
    confidence_bucket: str
    category: str = "other"
    latency_seconds: float = 0.0
    hallucination_flagged: bool = False  # True if ≥1 claim failed grounding
    bias_flagged: bool = False           # True if reflection detected a bias
    judge_scores_run1: dict[str, int] = field(default_factory=dict)
    judge_scores_run2: dict[str, int] = field(default_factory=dict)


@dataclass
class BacktestReport:
    """Full backtest report with all performance metrics."""

    # Volume
    markets_evaluated: int = 0
    markets_resolved: int = 0
    resolution_rate: float = 0.0

    # Core calibration
    brier_score: float = 0.0
    log_loss: float = 0.0
    overall_accuracy: float = 0.0
    calibration_error: float = 0.0   # Mean absolute deviation from diagonal
    calibration_curve: list[dict] = field(default_factory=list)

    # Edge / alpha
    ev_per_trade: float = 0.0            # avg (predicted_prob - market_prob)
    market_adjusted_return: float = 0.0  # correct_rate - avg_market_prob
    alpha_rate: float = 0.0              # % diverged >10% AND correct

    # Hit rate by confidence tier
    hit_rate_by_tier: dict[str, dict] = field(default_factory=dict)

    # Pipeline quality
    judge_consistency: float = 0.0       # 0–1 (1 = perfectly consistent across runs)
    bias_detection_recall: float = 0.0   # of flagged: % that actually resolved against market
    hallucination_catch_rate: float = 0.0

    # Latency
    latency_p50: float = 0.0
    latency_p95: float = 0.0

    # By category
    by_category: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume": {
                "markets_evaluated": self.markets_evaluated,
                "markets_resolved": self.markets_resolved,
                "resolution_rate": round(self.resolution_rate, 4),
            },
            "calibration": {
                "brier_score": round(self.brier_score, 4),
                "log_loss": round(self.log_loss, 4),
                "overall_accuracy": round(self.overall_accuracy, 4),
                "calibration_error": round(self.calibration_error, 4),
                "calibration_curve": self.calibration_curve,
            },
            "edge": {
                "ev_per_trade": round(self.ev_per_trade, 4),
                "market_adjusted_return": round(self.market_adjusted_return, 4),
                "alpha_rate": round(self.alpha_rate, 4),
            },
            "hit_rate_by_tier": self.hit_rate_by_tier,
            "pipeline": {
                "judge_consistency": round(self.judge_consistency, 4),
                "bias_detection_recall": round(self.bias_detection_recall, 4),
                "hallucination_catch_rate": round(self.hallucination_catch_rate, 4),
                "latency_p50_seconds": round(self.latency_p50, 3),
                "latency_p95_seconds": round(self.latency_p95, 3),
            },
            "by_category": self.by_category,
        }


_BUCKETS = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
_BUCKET_RANGES = [
    (0.50, 0.60, 0.55),
    (0.60, 0.70, 0.65),
    (0.70, 0.80, 0.75),
    (0.80, 0.90, 0.85),
    (0.90, 1.01, 0.95),
]


def _clip(p: float, eps: float = 1e-7) -> float:
    return max(eps, min(1 - eps, p))


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


class BacktestAggregator:
    """Computes all backtest metrics from a list of PredictionRecord objects."""

    def compute(self, records: list[PredictionRecord]) -> BacktestReport:
        if not records:
            return BacktestReport()

        report = BacktestReport()
        report.markets_evaluated = len(records)
        report.markets_resolved = len(records)
        report.resolution_rate = 1.0  # All passed-in records are resolved

        # --- Core calibration ---
        report.brier_score = _mean([r.brier_score for r in records])

        log_losses = []
        for r in records:
            p = _clip(r.predicted_prob)
            y = 1.0 if r.actual_outcome else 0.0
            log_losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
        report.log_loss = _mean(log_losses)

        report.overall_accuracy = _mean([1.0 if r.is_correct else 0.0 for r in records])

        # --- Calibration curve ---
        curve = []
        deviations = []
        for label, (lo, hi, midpoint) in zip(_BUCKETS, _BUCKET_RANGES):
            bucket = [r for r in records if lo <= r.predicted_prob < hi]
            if not bucket:
                curve.append({"bucket": label, "predicted_avg": midpoint, "actual_rate": None, "count": 0})
                continue
            predicted_avg = _mean([r.predicted_prob for r in bucket])
            actual_rate = _mean([1.0 if r.actual_outcome else 0.0 for r in bucket])
            deviations.append(abs(predicted_avg - actual_rate))
            curve.append({
                "bucket": label,
                "predicted_avg": round(predicted_avg, 4),
                "actual_rate": round(actual_rate, 4),
                "count": len(bucket),
            })
        report.calibration_curve = curve
        report.calibration_error = _mean(deviations)

        # --- Edge / alpha ---
        evs = [r.predicted_prob - r.market_prob_at_entry for r in records]
        report.ev_per_trade = _mean(evs)

        correct_rate = _mean([1.0 if r.is_correct else 0.0 for r in records])
        avg_market_prob = _mean([r.market_prob_at_entry for r in records])
        report.market_adjusted_return = correct_rate - avg_market_prob

        alpha_count = sum(
            1 for r in records
            if abs(r.predicted_prob - r.market_prob_at_entry) > 0.10 and r.is_correct
        )
        report.alpha_rate = alpha_count / len(records)

        # --- Hit rate by confidence tier ---
        for label, (lo, hi, _) in zip(_BUCKETS, _BUCKET_RANGES):
            bucket = [r for r in records if lo <= r.predicted_prob < hi]
            if not bucket:
                report.hit_rate_by_tier[label] = {"count": 0, "accuracy": None}
                continue
            acc = _mean([1.0 if r.is_correct else 0.0 for r in bucket])
            report.hit_rate_by_tier[label] = {"count": len(bucket), "accuracy": round(acc, 4)}

        # --- Judge consistency (requires 2 runs of judge per record) ---
        consistency_scores = []
        for r in records:
            if r.judge_scores_run1 and r.judge_scores_run2:
                dims = set(r.judge_scores_run1) & set(r.judge_scores_run2)
                if dims:
                    avg_delta = _mean([abs(r.judge_scores_run1[d] - r.judge_scores_run2[d]) for d in dims])
                    consistency_scores.append(1.0 - avg_delta / 10.0)
        report.judge_consistency = _mean(consistency_scores)

        # --- Bias detection recall ---
        # Of all bias-flagged predictions, what % actually resolved against the market-implied direction?
        flagged = [r for r in records if r.bias_flagged]
        if flagged:
            market_was_wrong = [
                r for r in flagged
                if (r.market_prob_at_entry >= 0.5) != r.actual_outcome
            ]
            report.bias_detection_recall = len(market_was_wrong) / len(flagged)

        # --- Hallucination catch rate ---
        report.hallucination_catch_rate = _mean([1.0 if r.hallucination_flagged else 0.0 for r in records])

        # --- Latency percentiles ---
        latencies = sorted(r.latency_seconds for r in records if r.latency_seconds > 0)
        if latencies:
            p50_idx = max(0, int(len(latencies) * 0.50) - 1)
            p95_idx = max(0, int(len(latencies) * 0.95) - 1)
            report.latency_p50 = latencies[p50_idx]
            report.latency_p95 = latencies[p95_idx]

        # --- By category ---
        categories: dict[str, list[PredictionRecord]] = {}
        for r in records:
            categories.setdefault(r.category, []).append(r)

        for cat, cat_records in categories.items():
            report.by_category[cat] = {
                "count": len(cat_records),
                "accuracy": round(_mean([1.0 if r.is_correct else 0.0 for r in cat_records]), 4),
                "brier_score": round(_mean([r.brier_score for r in cat_records]), 4),
                "ev_per_trade": round(_mean([r.predicted_prob - r.market_prob_at_entry for r in cat_records]), 4),
            }

        return report

    def print_summary(self, report: BacktestReport) -> None:
        """Print a human-readable summary to stdout."""
        d = report.to_dict()
        print("\n" + "=" * 60)
        print("ORACLE BACKTEST RESULTS")
        print("=" * 60)

        v = d["volume"]
        print(f"\nVolume")
        print(f"  Markets evaluated : {v['markets_evaluated']}")
        print(f"  Resolution rate   : {v['resolution_rate']:.1%}")

        c = d["calibration"]
        print(f"\nCalibration")
        print(f"  Brier score       : {c['brier_score']:.4f}  (random=0.25, good<0.15)")
        print(f"  Log loss          : {c['log_loss']:.4f}")
        print(f"  Accuracy          : {c['overall_accuracy']:.1%}")
        print(f"  Calibration error : {c['calibration_error']:.4f}  (0=perfect)")

        print(f"\n  Calibration curve:")
        print(f"  {'Bucket':<12} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        for b in c["calibration_curve"]:
            actual = f"{b['actual_rate']:.3f}" if b["actual_rate"] is not None else "  n/a"
            print(f"  {b['bucket']:<12} {b['predicted_avg']:>10.3f} {actual:>10} {b['count']:>8}")

        e = d["edge"]
        print(f"\nEdge")
        print(f"  EV per trade           : {e['ev_per_trade']:+.4f}")
        print(f"  Market-adjusted return : {e['market_adjusted_return']:+.4f}")
        print(f"  Alpha rate             : {e['alpha_rate']:.1%}  (diverged >10% AND correct)")

        print(f"\nHit rate by confidence tier:")
        for tier, v2 in d["hit_rate_by_tier"].items():
            acc = f"{v2['accuracy']:.1%}" if v2["accuracy"] is not None else "n/a"
            print(f"  {tier:<10} {acc:>8}  (n={v2['count']})")

        p = d["pipeline"]
        print(f"\nPipeline quality")
        print(f"  Judge consistency      : {p['judge_consistency']:.1%}")
        print(f"  Bias detection recall  : {p['bias_detection_recall']:.1%}")
        print(f"  Hallucination catch    : {p['hallucination_catch_rate']:.1%}")
        print(f"  Latency p50            : {p['latency_p50_seconds']:.1f}s")
        print(f"  Latency p95            : {p['latency_p95_seconds']:.1f}s")

        if d["by_category"]:
            print(f"\nBy category:")
            print(f"  {'Category':<12} {'Count':>6} {'Accuracy':>10} {'Brier':>8} {'EV':>8}")
            for cat, stats in d["by_category"].items():
                print(
                    f"  {cat:<12} {stats['count']:>6} "
                    f"{stats['accuracy']:>10.1%} {stats['brier_score']:>8.4f} "
                    f"{stats['ev_per_trade']:>+8.4f}"
                )
        print("=" * 60 + "\n")
