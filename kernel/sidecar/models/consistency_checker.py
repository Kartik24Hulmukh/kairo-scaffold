"""F2 — Consistency Scoring for structured model outputs.

Reference: STED / Consistency Scoring (arXiv:2512.23712, Amazon Science).
Motivation: Cleanlab research shows that schema-valid outputs can be
semantically wrong. The constrained decoder (B4) guarantees JSON Schema
conformance but NOT semantic correctness. A model under constrained
decoding may produce valid-but-wrong values consistently OR inconsistently.

Consistency Scoring quantifies: given prompt P and schema S, if we sample
the decoder N times, how often do the field values agree?

Algorithm (field-level agreement):
    1. Sample the decoder N times at temperature T (default N=5, T=0.7).
    2. For each field in the schema's required properties:
       a. Collect [v1, v2, ..., vN] from N samples.
       b. Normalize each value to a canonical key (str, rounded float, bool).
       c. Count the mode: the most frequent normalized value.
       d. field_consistency = mode_count / N   ∈ [0.0, 1.0]
    3. object_consistency = mean(field_consistencies over required fields).
    4. low_consistency_fields = [f for f, s in per_field if s < LOW_THRESHOLD]

Integration with STEDGate (B4/D4 pipeline):
    After ConstrainedDecoder.decode_structured(), call:
        result = ConsistencyChecker().score(prompt, schema, obj, n=5)
        sted_result = STEDGate().validate(obj, schema, consistency=result)
    STEDGate surfaces consistency_score and low_consistency_fields in the
    returned envelope so callers can down-rank or block low-consistency fields.

Catch known constrained-decoding failure mode:
    When the grammar-constrained beam forces a semantically wrong token (e.g.,
    total_amount=0 because the model's probability mass was on "0" matching the
    numeric grammar), the same wrong value appears in most samples → high
    within-sample consistency but the caller can cross-check against anchors.
    When values vary across samples (ambiguous prompt or multi-valid answers),
    the field consistency is low → confidence is multiplied by the consistency
    score as a down-ranking signal.

Gate:
    pytest kernel/tests/test_consistency_checker.py -v
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of samples for consistency scoring.
DEFAULT_N: int = int(os.environ.get("KAIRO_CONSISTENCY_N", "5"))

#: Default sampling temperature. STED paper recommends 0.7.
DEFAULT_TEMPERATURE: float = float(os.environ.get("KAIRO_CONSISTENCY_TEMP", "0.7"))

#: Fields with agreement ratio below this threshold are flagged as low-confidence.
LOW_CONSISTENCY_THRESHOLD: float = float(
    os.environ.get("KAIRO_CONSISTENCY_LOW_THRESHOLD", "0.60")
)

#: Confidence is multiplied by this factor for low-consistency fields.
LOW_CONSISTENCY_PENALTY: float = float(
    os.environ.get("KAIRO_CONSISTENCY_PENALTY", "0.5")
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyResult:
    """Result of a consistency scoring run.

    Attributes:
        score: Mean field-agreement ratio across required fields, ∈ [0.0, 1.0].
            1.0 = all samples agreed on all fields (maximally consistent).
            0.0 = no two samples agreed on any field (maximally inconsistent).
        per_field: Mapping of field_name → agreement_ratio ∈ [0.0, 1.0].
        low_consistency_fields: Fields whose agreement ratio < LOW_CONSISTENCY_THRESHOLD.
        n_samples: Number of samples used.
        temperature: Sampling temperature used.
        sampled_values: Per-field list of sampled raw values (for audit/debug).
    """

    score: float
    per_field: dict[str, float]
    low_consistency_fields: list[str]
    n_samples: int
    temperature: float
    sampled_values: dict[str, list[Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical value normalisation (for agreement comparison)
# ---------------------------------------------------------------------------


def _canonical_key(value: Any) -> str:
    """Convert a field value to a canonical string key for equality comparison.

    Rules:
    - None           → "null"
    - bool           → "true" / "false"
    - int            → str(int)
    - float          → str(round(float, 2))  (round to 2dp to absorb noise)
    - str            → lowercased + stripped
    - list / dict    → sorted str(repr) (deep structural)
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "nan_or_inf"
        return str(round(value, 2))
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, dict)):
        return repr(sorted(str(value)))
    return str(value)


# ---------------------------------------------------------------------------
# ConsistencyChecker
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """Score the field-level consistency of a structured decoder.

    Usage (normal, with ConstrainedDecoder)::

        checker = ConsistencyChecker()
        result = checker.score(prompt, schema, reference_obj, n=5)
        # result.score ∈ [0.0, 1.0]
        # result.low_consistency_fields = ["total_amount"] if values vary

    Usage (with custom sampler, for testing)::

        def my_sampler(prompt, schema):
            return {"name": "Kairo", "score": 0.9}

        result = checker.score(prompt, schema, ref, n=5, sampler=my_sampler)

    The sampler callable has signature: (prompt: str, schema: dict) → dict.
    If None, uses ConstrainedDecoder (offline fallback = schema-guided fuzzer).
    """

    def __init__(
        self,
        low_threshold: float = LOW_CONSISTENCY_THRESHOLD,
        penalty: float = LOW_CONSISTENCY_PENALTY,
    ) -> None:
        self._low_threshold = low_threshold
        self._penalty = penalty

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(
        self,
        prompt: str,
        schema: dict,
        reference_obj: dict,
        n: int = DEFAULT_N,
        temperature: float = DEFAULT_TEMPERATURE,
        sampler: Optional[Callable[[str, dict], dict]] = None,
    ) -> ConsistencyResult:
        """Score field-level consistency over N samples.

        Args:
            prompt: The generation prompt (same prompt used to produce reference_obj).
            schema: JSON Schema (draft-07) the output must conform to.
            reference_obj: The candidate output to evaluate consistency for.
                           Used to identify which fields to check.
            n: Number of samples to draw (default: 5).
            temperature: Sampling temperature hint (forwarded to sampler if supported).
            sampler: Optional callable (prompt, schema) → dict.
                     Defaults to ConstrainedDecoder (offline-safe).

        Returns:
            ConsistencyResult with .score, .per_field, .low_consistency_fields.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        effective_sampler = sampler or self._default_sampler()

        # Identify fields to score (required fields from schema, fall back to ref keys)
        fields_to_score = self._fields_to_score(schema, reference_obj)

        if not fields_to_score:
            # No scoreable fields — return perfect consistency (nothing to disagree on)
            return ConsistencyResult(
                score=1.0,
                per_field={},
                low_consistency_fields=[],
                n_samples=n,
                temperature=temperature,
            )

        # Draw N samples
        samples: list[dict] = []
        for _ in range(n):
            try:
                obj = effective_sampler(prompt, schema)
                if not isinstance(obj, dict):
                    obj = {}
            except Exception:
                obj = {}
            samples.append(obj)

        # Compute per-field agreement
        per_field: dict[str, float] = {}
        sampled_values: dict[str, list[Any]] = {}

        for fname in fields_to_score:
            vals = [s.get(fname) for s in samples]
            sampled_values[fname] = vals
            keys = [_canonical_key(v) for v in vals]
            if n == 0:
                per_field[fname] = 0.0
                continue
            mode_count = Counter(keys).most_common(1)[0][1]
            per_field[fname] = mode_count / n

        # Object-level consistency = mean of field scores
        if per_field:
            obj_score = sum(per_field.values()) / len(per_field)
        else:
            obj_score = 1.0

        obj_score = max(0.0, min(1.0, obj_score))

        low_consistency_fields = [
            f for f, s in per_field.items() if s < self._low_threshold
        ]

        return ConsistencyResult(
            score=obj_score,
            per_field=per_field,
            low_consistency_fields=low_consistency_fields,
            n_samples=n,
            temperature=temperature,
            sampled_values=sampled_values,
        )

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def adjust_confidence(
        self,
        field_name: str,
        original_confidence: float,
        consistency_result: ConsistencyResult,
    ) -> float:
        """Multiply confidence by penalty if field has low consistency.

        Args:
            field_name: The field whose confidence to adjust.
            original_confidence: Original confidence ∈ [0.0, 1.0].
            consistency_result: The ConsistencyResult for this object.

        Returns:
            Adjusted confidence ∈ [0.0, 1.0].
        """
        field_score = consistency_result.per_field.get(field_name)
        if field_score is None:
            return original_confidence
        if field_score < self._low_threshold:
            adjusted = original_confidence * self._penalty
        else:
            adjusted = original_confidence
        return max(0.0, min(1.0, adjusted))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fields_to_score(schema: dict, reference_obj: dict) -> list[str]:
        """Return list of field names to score.

        Priority:
        1. schema['required'] — guaranteed fields
        2. schema['properties'].keys() — declared optional fields
        3. reference_obj.keys() — fallback if no schema properties
        Scalar types only (not nested objects/arrays at depth>0 by default).
        """
        required = list(schema.get("required", []))
        properties = schema.get("properties", {})

        candidates: list[str] = []
        # Required fields first (most important to score)
        for f in required:
            if f not in candidates:
                candidates.append(f)
        # Then declared optional properties
        for f in properties:
            if f not in candidates:
                candidates.append(f)
        # Fallback to reference object keys
        if not candidates:
            candidates = list(reference_obj.keys())

        return candidates

    @staticmethod
    def _default_sampler() -> Callable[[str, dict], dict]:
        """Return the default sampler: ConstrainedDecoder (offline-safe)."""
        from kernel.sidecar.models.constrained_decoding import (
            ConstrainedDecoder,
        )

        decoder = ConstrainedDecoder()

        def _sample(prompt: str, schema: dict) -> dict:
            _scratchpad, obj = decoder.decode_structured(prompt, schema)
            return obj

        return _sample
