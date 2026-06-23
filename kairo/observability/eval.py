"""
Kairo Eval + Monitoring — Regression Detection + Drift Alerts.

Uses the tracing library's eval framework with Kairo-specific additions:
  - Per-call scoring: grounded_rate, false_refusal_rate, wrong_bbox_rate
  - Regression detection: rolling window (last 100 extractions)
  - Drift detection: avg confidence per field per pack, alert on drops > 10%
  - Export: GET /api/eval/report -> JSON eval report
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kairo.observability.trace import _trace_store, GroundingTrace

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Metrics from evaluating a batch of extractions."""
    grounded_rate: float = 0.0
    false_refusal_rate: float = 0.0
    wrong_bbox_rate: float = 0.0
    total_extractions: int = 0
    grounded_count: int = 0
    refused_count: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded_rate": round(self.grounded_rate, 2),
            "false_refusal_rate": round(self.false_refusal_rate, 2),
            "wrong_bbox_rate": round(self.wrong_bbox_rate, 2),
            "total_extractions": self.total_extractions,
            "grounded_count": self.grounded_count,
            "refused_count": self.refused_count,
            "timestamp": self.timestamp,
        }


@dataclass
class RegressionAlert:
    """A regression alert when metrics drop below thresholds."""
    alert_type: str  # "warning", "critical"
    metric: str  # "grounded_rate", "false_refusal_rate"
    current_value: float
    threshold: float
    message: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "metric": self.metric,
            "current_value": round(self.current_value, 2),
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp,
        }


# Rolling window for regression detection
_rolling_window: deque = deque(maxlen=100)
# Baseline confidence per field per pack
_confidence_baselines: dict[str, float] = {}
# Current confidence tracking
_confidence_history: dict[str, list[float]] = defaultdict(list)
# Active alerts
_active_alerts: list[RegressionAlert] = []


def score_extraction(trace: GroundingTrace) -> EvalMetrics:
    """Score a single grounding trace.

    Args:
        trace: A GroundingTrace from the trace store.

    Returns:
        EvalMetrics for this extraction.
    """
    total = 1
    grounded = 1 if trace.final_decision == "grounded" else 0
    refused = 1 if trace.final_decision == "refused" else 0
    wrong_bbox = 1 if (trace.final_decision == "grounded" and trace.final_bbox is None) else 0

    return EvalMetrics(
        grounded_rate=grounded / total * 100,
        false_refusal_rate=refused / total * 100,
        wrong_bbox_rate=wrong_bbox / total * 100,
        total_extractions=total,
        grounded_count=grounded,
        refused_count=refused,
        timestamp=trace.timestamp,
    )


def update_rolling_window(metrics: EvalMetrics) -> None:
    """Add metrics to the rolling window for regression detection."""
    _rolling_window.append(metrics)


def detect_regression() -> list[RegressionAlert]:
    """Detect regressions in the rolling window.

    - grounded_rate < 95% -> warning, < 90% -> critical
    - false_refusal_rate > 5% -> warning, > 10% -> critical
    """
    alerts: list[RegressionAlert] = []
    now = datetime.now().isoformat()

    if len(_rolling_window) < 10:
        return alerts  # not enough data

    # Calculate aggregate metrics from rolling window
    total = len(_rolling_window)
    grounded_count = sum(m.grounded_count for m in _rolling_window)
    refused_count = sum(m.refused_count for m in _rolling_window)
    grounded_rate = grounded_count / total * 100
    false_refusal_rate = refused_count / total * 100

    # Grounded rate regression
    if grounded_rate < 90:
        alerts.append(RegressionAlert(
            alert_type="critical",
            metric="grounded_rate",
            current_value=grounded_rate,
            threshold=90.0,
            message=f"CRITICAL: grounded_rate dropped to {grounded_rate:.1f}% (threshold: 90%)",
            timestamp=now,
        ))
    elif grounded_rate < 95:
        alerts.append(RegressionAlert(
            alert_type="warning",
            metric="grounded_rate",
            current_value=grounded_rate,
            threshold=95.0,
            message=f"WARNING: grounded_rate at {grounded_rate:.1f}% (threshold: 95%)",
            timestamp=now,
        ))

    # False refusal rate regression
    if false_refusal_rate > 10:
        alerts.append(RegressionAlert(
            alert_type="critical",
            metric="false_refusal_rate",
            current_value=false_refusal_rate,
            threshold=10.0,
            message=f"CRITICAL: false_refusal_rate at {false_refusal_rate:.1f}% (threshold: 10%)",
            timestamp=now,
        ))
    elif false_refusal_rate > 5:
        alerts.append(RegressionAlert(
            alert_type="warning",
            metric="false_refusal_rate",
            current_value=false_refusal_rate,
            threshold=5.0,
            message=f"WARNING: false_refusal_rate at {false_refusal_rate:.1f}% (threshold: 5%)",
            timestamp=now,
        ))

    return alerts


def detect_drift(field_name: str, confidence: float, pack_name: str = "") -> RegressionAlert | None:
    """Detect confidence drift for a specific field.

    If avg confidence drops > 10% from baseline, return a drift alert.
    """
    key = f"{pack_name}:{field_name}" if pack_name else field_name
    _confidence_history[key].append(confidence)

    # Keep only last 50 measurements
    if len(_confidence_history[key]) > 50:
        _confidence_history[key] = _confidence_history[key][-50:]

    # Set baseline after first 10 measurements
    if key not in _confidence_baselines and len(_confidence_history[key]) >= 10:
        baseline = sum(_confidence_history[key][:10]) / 10
        _confidence_baselines[key] = baseline
        return None

    # Check for drift
    if key in _confidence_baselines and len(_confidence_history[key]) >= 20:
        baseline = _confidence_baselines[key]
        recent = sum(_confidence_history[key][-10:]) / 10
        if baseline > 0 and (baseline - recent) / baseline > 0.10:
            return RegressionAlert(
                alert_type="warning",
                metric=f"confidence_drift:{key}",
                current_value=recent,
                threshold=baseline,
                message=f"DRIFT: {key} confidence dropped from {baseline:.2f} to {recent:.2f} (>10% drop)",
                timestamp=datetime.now().isoformat(),
            )

    return None


def get_eval_report() -> dict[str, Any]:
    """Generate a full eval report for GET /api/eval/report."""
    # Score all traces in the store
    all_metrics = [score_extraction(t) for t in _trace_store]

    # Aggregate
    total = len(all_metrics)
    if total == 0:
        return {
            "total_extractions": 0,
            "grounded_rate": 0.0,
            "false_refusal_rate": 0.0,
            "regression_alerts": [],
            "drift_alerts": [],
            "rolling_window_size": len(_rolling_window),
        }

    grounded = sum(m.grounded_count for m in all_metrics)
    refused = sum(m.refused_count for m in all_metrics)

    # Detect regressions
    regression_alerts = detect_regression()
    _active_alerts.clear()
    _active_alerts.extend(regression_alerts)

    return {
        "total_extractions": total,
        "grounded_rate": round(grounded / total * 100, 2),
        "false_refusal_rate": round(refused / total * 100, 2),
        "wrong_bbox_rate": round(sum(m.wrong_bbox_rate for m in all_metrics) / total, 2),
        "regression_alerts": [a.to_dict() for a in regression_alerts],
        "drift_alerts": [],
        "rolling_window_size": len(_rolling_window),
        "confidence_baselines": dict(_confidence_baselines),
        "timestamp": datetime.now().isoformat(),
    }


def reset_eval_state() -> None:
    """Reset all eval state (for testing)."""
    _rolling_window.clear()
    _confidence_baselines.clear()
    _confidence_history.clear()
    _active_alerts.clear()