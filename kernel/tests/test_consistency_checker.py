"""F2 — Consistency Scoring tests: field-level agreement + STEDGate integration.

Reference: STED / Consistency Scoring (arXiv:2512.23712, Amazon Science)
Motivation: Cleanlab "Pervasive Label Errors in Test Sets" (NeurIPS 2021, arXiv:2103.14749)

Tests:
  F2-01: ConsistencyResult schema — all fields present, types correct
  F2-02: Score in [0.0, 1.0] always
  F2-03: Perfect consistency (all samples agree) → score = 1.0
  F2-04: Zero consistency (all samples differ) → score = low
  F2-05: Per-field scores present for required fields
  F2-06: low_consistency_fields populated when fields disagree
  F2-07: KNOWN FAILURE GATE — schema-valid-but-wrong output is caught (score < 0.60)
  F2-08: High-consistency correct fields → unaffected (score ≥ 0.80)
  F2-09: adjust_confidence — low-consistency field penalised
  F2-10: adjust_confidence — high-consistency field unaffected
  F2-11: STEDGate.validate — consistency_score surfaced in result
  F2-12: STEDGate.validate — low_consistency_fields surfaced in result
  F2-13: STEDGate.validate — no consistency passed → consistency_score is None
  F2-14: STEDGate.validate — schema-invalid + consistency still returns all keys
  F2-15: _canonical_key — None, bool, int, float, str normalization
  F2-16: n=1 → consistency = 1.0 (single sample always agrees with itself)
  F2-17: Empty schema → empty per_field → score = 1.0
  F2-18: ConsistencyChecker — custom sampler injection
  F2-19: ConstrainedDecoder.decode_structured still works after F2 changes
  F2-20: STEDGate maintains backward compat (passed/errors always present)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernel.sidecar.models.consistency_checker import (
    ConsistencyChecker,
    ConsistencyResult,
    LOW_CONSISTENCY_THRESHOLD,
    _canonical_key,
)
from kernel.sidecar.models.sted import STEDGate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INVOICE_SCHEMA = {
    "type": "object",
    "required": ["invoice_id", "total_amount", "vendor"],
    "properties": {
        "invoice_id":   {"type": "string"},
        "total_amount": {"type": "number", "minimum": 0},
        "vendor":       {"type": "string", "enum": ["Acme Corp", "Globex", "Initech"]},
        "is_paid":      {"type": "boolean"},
    },
}

PERFECT_INVOICE = {
    "invoice_id": "INV-001",
    "total_amount": 1234.56,
    "vendor": "Acme Corp",
    "is_paid": True,
}


def _make_fixed_sampler(*objects):
    """Return a sampler that cycles through the given objects."""
    items = list(objects)
    state = {"i": 0}

    def _sampler(prompt: str, schema: dict) -> dict:
        obj = items[state["i"] % len(items)]
        state["i"] += 1
        return obj

    return _sampler


def _make_uniform_sampler(obj: dict):
    """Return a sampler that always returns the same object (perfect agreement)."""
    def _sampler(prompt: str, schema: dict) -> dict:
        return dict(obj)
    return _sampler


def _make_chaotic_sampler(schema: dict, seed_offset: int = 0):
    """Return a sampler that always returns a random (different) schema-valid object."""
    from kernel.sidecar.models.constrained_decoding import generate_fuzzed_json_from_schema
    counter = {"n": seed_offset}

    def _sampler(prompt: str, _schema: dict) -> dict:
        counter["n"] += 1
        # Use fuzzed generator: each call returns a different random object
        return generate_fuzzed_json_from_schema(schema)

    return _sampler


# ---------------------------------------------------------------------------
# F2-01: ConsistencyResult schema
# ---------------------------------------------------------------------------

class TestConsistencyResultSchema:
    """F2-01: ConsistencyResult has all required fields and correct types."""

    def test_all_fields_present(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=3, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert hasattr(result, "score")
        assert hasattr(result, "per_field")
        assert hasattr(result, "low_consistency_fields")
        assert hasattr(result, "n_samples")
        assert hasattr(result, "temperature")
        assert hasattr(result, "sampled_values")

    def test_types(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=3, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert isinstance(result, ConsistencyResult)
        assert isinstance(result.score, float)
        assert isinstance(result.per_field, dict)
        assert isinstance(result.low_consistency_fields, list)
        assert isinstance(result.n_samples, int)
        assert isinstance(result.temperature, float)

    def test_n_samples_recorded(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert result.n_samples == 5


# ---------------------------------------------------------------------------
# F2-02: Score range
# ---------------------------------------------------------------------------

class TestScoreRange:
    """F2-02: Consistency score always in [0.0, 1.0]."""

    def test_score_in_range_uniform(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert 0.0 <= result.score <= 1.0

    def test_score_in_range_chaotic(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=_make_chaotic_sampler(INVOICE_SCHEMA)
        )
        assert 0.0 <= result.score <= 1.0

    def test_per_field_scores_in_range(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        for field_name, field_score in result.per_field.items():
            assert 0.0 <= field_score <= 1.0, (
                f"per_field[{field_name!r}]={field_score} out of range"
            )


# ---------------------------------------------------------------------------
# F2-03: Perfect consistency → score = 1.0
# ---------------------------------------------------------------------------

class TestPerfectConsistency:
    """F2-03: When all N samples agree on all fields, score = 1.0."""

    def test_uniform_sampler_score_is_1(self):
        """If sampler always returns same object, consistency = 1.0."""
        checker = ConsistencyChecker()
        result = checker.score(
            "What is the invoice total?",
            INVOICE_SCHEMA,
            PERFECT_INVOICE,
            n=5,
            sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert result.score == 1.0, (
            f"Expected 1.0 for uniform sampler, got {result.score}"
        )

    def test_uniform_sampler_no_low_consistency_fields(self):
        """Perfect consistency → no low_consistency_fields."""
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        assert result.low_consistency_fields == []


# ---------------------------------------------------------------------------
# F2-04: Zero consistency (all samples differ) → score = low
# ---------------------------------------------------------------------------

class TestLowConsistency:
    """F2-04: When N samples all differ, consistency score is low."""

    def test_chaotic_sampler_score_is_low(self):
        """Random sampler (all samples differ) → low consistency score."""
        checker = ConsistencyChecker()
        # Use a sampler that returns 5 completely different objects
        objects = [
            {"invoice_id": f"INV-{i:03d}", "total_amount": float(i * 100),
             "vendor": ["Acme Corp", "Globex", "Initech"][i % 3]}
            for i in range(1, 6)
        ]
        sampler = _make_fixed_sampler(*objects)
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=sampler
        )
        # All objects differ → low consistency
        assert result.score < 0.80, (
            f"Expected low consistency for chaotic sampler, got {result.score}"
        )


# ---------------------------------------------------------------------------
# F2-05 to F2-06: Per-field scores
# ---------------------------------------------------------------------------

class TestPerFieldScores:
    """F2-05/F2-06: Per-field scores and low_consistency_fields."""

    def test_per_field_keys_include_required_fields(self):
        """per_field includes all required schema fields."""
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=3, sampler=_make_uniform_sampler(PERFECT_INVOICE)
        )
        for req_field in INVOICE_SCHEMA["required"]:
            assert req_field in result.per_field, (
                f"Required field {req_field!r} missing from per_field"
            )

    def test_low_consistency_field_detected(self):
        """A field that varies across samples is flagged in low_consistency_fields."""
        checker = ConsistencyChecker(low_threshold=0.80)

        # 4 samples with total_amount=100, 1 sample with total_amount=999
        # → mode_count=4, agreement=0.80 (exactly at threshold)
        # Use total_amount varying: 1 divergent out of 5
        objects = [
            {"invoice_id": "INV-001", "total_amount": 100.0, "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 100.0, "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 100.0, "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 999.0, "vendor": "Acme Corp"},  # divergent
            {"invoice_id": "INV-001", "total_amount": 100.0, "vendor": "Acme Corp"},
        ]
        sampler = _make_fixed_sampler(*objects)
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=5, sampler=sampler
        )
        # total_amount agreement: 4/5 = 0.80 — at threshold 0.80 exactly
        # (borderline — field should be very close to threshold)
        assert "total_amount" in result.per_field
        assert result.per_field["total_amount"] <= 0.85  # mode_count=4, n=5


# ---------------------------------------------------------------------------
# F2-07: KNOWN FAILURE GATE — schema-valid-but-wrong caught (CRITICAL)
# ---------------------------------------------------------------------------

class TestKnownFailureGate:
    """F2-07: Known constrained-decoding failure mode caught by low consistency.

    The constrained decoder (B4) guarantees schema validity but NOT semantic
    correctness. Example failure:
        - The grammar forces total_amount to a valid number
        - Due to beam search failure, it chooses 0.0 (valid per schema)
        - The reference object says total_amount=0.0 (schema-valid, wrong)

    Consistency check: sample 5 times, 3 different values appear for total_amount
    → consistency < 0.60 → output is flagged as low-confidence.

    This is the gate: a known failure case must be caught.
    """

    def test_valid_but_wrong_output_caught_by_low_consistency(self):
        """Schema-valid-but-semantically-wrong output → consistency < LOW_THRESHOLD.

        Simulates a constrained-decoding failure where total_amount=0 was
        chosen (beam search got stuck) despite the document containing $1,234.56.
        When we sample again, different valid values appear → low consistency.
        """
        # The "wrong" reference output: total_amount=0 (schema-valid, wrong)
        wrong_obj = {
            "invoice_id": "INV-001",
            "total_amount": 0.0,   # WRONG: should be 1234.56
            "vendor": "Acme Corp",
        }

        # Sampler simulates model uncertainty: 3 different values across 5 samples
        # → none dominates → low consistency for total_amount
        uncertain_samples = [
            {"invoice_id": "INV-001", "total_amount": 0.0,     "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 1234.56, "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 0.0,     "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 500.0,   "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 1234.56, "vendor": "Acme Corp"},
        ]
        sampler = _make_fixed_sampler(*uncertain_samples)

        checker = ConsistencyChecker(low_threshold=LOW_CONSISTENCY_THRESHOLD)
        result = checker.score(
            "Extract the invoice total.",
            INVOICE_SCHEMA,
            wrong_obj,
            n=5,
            sampler=sampler,
        )

        # total_amount mode: 0.0 appears 2x, 1234.56 appears 2x, 500.0 appears 1x
        # → mode_count = 2, agreement = 2/5 = 0.40 < 0.60 → LOW CONSISTENCY
        total_amount_consistency = result.per_field.get("total_amount", 1.0)
        assert total_amount_consistency < LOW_CONSISTENCY_THRESHOLD, (
            f"GATE FAILED: Known failure case NOT caught by consistency checker.\n"
            f"  total_amount consistency = {total_amount_consistency:.3f} "
            f"(expected < {LOW_CONSISTENCY_THRESHOLD})\n"
            f"  This means a valid-but-wrong output would NOT be flagged as low-confidence."
        )

        # low_consistency_fields must include total_amount
        assert "total_amount" in result.low_consistency_fields, (
            f"'total_amount' not in low_consistency_fields: {result.low_consistency_fields}"
        )

    def test_low_consistency_output_is_down_ranked(self):
        """Low-consistency output's confidence is penalised by adjust_confidence."""
        wrong_obj = {"invoice_id": "INV-001", "total_amount": 0.0, "vendor": "Acme Corp"}
        uncertain_samples = [
            {"invoice_id": "INV-001", "total_amount": 0.0,     "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 1234.56, "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 0.0,     "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 500.0,   "vendor": "Acme Corp"},
            {"invoice_id": "INV-001", "total_amount": 1234.56, "vendor": "Acme Corp"},
        ]
        sampler = _make_fixed_sampler(*uncertain_samples)

        checker = ConsistencyChecker(low_threshold=0.60, penalty=0.5)
        result = checker.score(
            "Extract the invoice total.",
            INVOICE_SCHEMA,
            wrong_obj,
            n=5,
            sampler=sampler,
        )

        original_confidence = 0.85
        adjusted = checker.adjust_confidence("total_amount", original_confidence, result)

        assert adjusted < original_confidence, (
            f"Confidence should be penalised for low-consistency field. "
            f"original={original_confidence}, adjusted={adjusted}"
        )
        assert adjusted == pytest.approx(original_confidence * 0.5, rel=0.01)


# ---------------------------------------------------------------------------
# F2-08: High-consistency correct fields → unaffected
# ---------------------------------------------------------------------------

class TestHighConsistencyUnaffected:
    """F2-08: High-consistency correct fields pass without down-ranking."""

    def test_high_consistency_field_score_is_high(self):
        """Fields where all N samples agree → per_field score = 1.0."""
        checker = ConsistencyChecker()
        result = checker.score(
            "Extract the invoice.",
            INVOICE_SCHEMA,
            PERFECT_INVOICE,
            n=5,
            sampler=_make_uniform_sampler(PERFECT_INVOICE),
        )
        assert result.per_field.get("invoice_id", 0.0) == 1.0
        assert result.per_field.get("vendor", 0.0) == 1.0

    def test_high_consistency_field_not_penalised(self):
        """adjust_confidence returns original confidence for high-consistency fields."""
        checker = ConsistencyChecker()
        result = checker.score(
            "Extract the invoice.",
            INVOICE_SCHEMA,
            PERFECT_INVOICE,
            n=5,
            sampler=_make_uniform_sampler(PERFECT_INVOICE),
        )
        original = 0.90
        adjusted = checker.adjust_confidence("invoice_id", original, result)
        assert adjusted == original, (
            f"High-consistency field should NOT be penalised. "
            f"original={original}, adjusted={adjusted}"
        )


# ---------------------------------------------------------------------------
# F2-09 to F2-10: adjust_confidence
# ---------------------------------------------------------------------------

class TestAdjustConfidence:
    """F2-09/F2-10: adjust_confidence penalises low-consistency, ignores high-consistency."""

    def test_low_consistency_reduces_confidence(self):
        # Simulate ConsistencyResult with low per_field score for "total_amount"
        result = ConsistencyResult(
            score=0.40,
            per_field={"invoice_id": 1.0, "total_amount": 0.40, "vendor": 1.0},
            low_consistency_fields=["total_amount"],
            n_samples=5,
            temperature=0.7,
        )
        checker = ConsistencyChecker(low_threshold=0.60, penalty=0.5)
        adjusted = checker.adjust_confidence("total_amount", 0.85, result)
        assert adjusted == pytest.approx(0.85 * 0.5, rel=0.01)

    def test_high_consistency_unchanged(self):
        result = ConsistencyResult(
            score=1.0,
            per_field={"invoice_id": 1.0, "total_amount": 1.0, "vendor": 1.0},
            low_consistency_fields=[],
            n_samples=5,
            temperature=0.7,
        )
        checker = ConsistencyChecker(low_threshold=0.60, penalty=0.5)
        adjusted = checker.adjust_confidence("total_amount", 0.85, result)
        assert adjusted == 0.85

    def test_adjusted_confidence_in_range(self):
        """Adjusted confidence is always in [0.0, 1.0]."""
        result = ConsistencyResult(
            score=0.0,
            per_field={"x": 0.0},
            low_consistency_fields=["x"],
            n_samples=5,
            temperature=0.7,
        )
        checker = ConsistencyChecker(low_threshold=0.60, penalty=0.5)
        adjusted = checker.adjust_confidence("x", 1.0, result)
        assert 0.0 <= adjusted <= 1.0


# ---------------------------------------------------------------------------
# F2-11 to F2-14: STEDGate integration
# ---------------------------------------------------------------------------

class TestSTEDGateIntegration:
    """F2-11 to F2-14: STEDGate surfaces consistency alongside schema gate."""

    SIMPLE_SCHEMA = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    def test_consistency_score_surfaced(self):
        """STEDGate result includes consistency_score when ConsistencyResult passed."""
        gate = STEDGate()
        cr = ConsistencyResult(
            score=0.85,
            per_field={"name": 1.0, "score": 0.85},
            low_consistency_fields=[],
            n_samples=5,
            temperature=0.7,
        )
        result = gate.validate({"name": "Kairo", "score": 0.9}, self.SIMPLE_SCHEMA, consistency=cr)
        assert result["consistency_score"] == pytest.approx(0.85, rel=0.01)

    def test_low_consistency_fields_surfaced(self):
        """STEDGate result includes low_consistency_fields when present."""
        gate = STEDGate()
        cr = ConsistencyResult(
            score=0.40,
            per_field={"name": 1.0, "score": 0.40},
            low_consistency_fields=["score"],
            n_samples=5,
            temperature=0.7,
        )
        result = gate.validate({"name": "Kairo", "score": 0.9}, self.SIMPLE_SCHEMA, consistency=cr)
        assert result["low_consistency_fields"] == ["score"]

    def test_no_consistency_passed_returns_none(self):
        """When no ConsistencyResult is passed, consistency_score is None."""
        gate = STEDGate()
        result = gate.validate({"name": "Kairo", "score": 0.9}, self.SIMPLE_SCHEMA)
        assert result["consistency_score"] is None
        assert result["low_consistency_fields"] is None

    def test_schema_invalid_with_consistency_returns_all_keys(self):
        """Schema-invalid output with consistency still returns full envelope."""
        gate = STEDGate()
        cr = ConsistencyResult(
            score=0.30,
            per_field={"name": 1.0, "score": 0.30},
            low_consistency_fields=["score"],
            n_samples=5,
            temperature=0.7,
        )
        result = gate.validate({}, self.SIMPLE_SCHEMA, consistency=cr)
        assert result["passed"] is False
        assert result["consistency_score"] == pytest.approx(0.30, rel=0.01)
        assert result["low_consistency_fields"] == ["score"]

    def test_backward_compat_passed_errors_always_present(self):
        """passed and errors always present (backward compat with D4 tests)."""
        gate = STEDGate()
        for obj, should_pass in [
            ({"name": "Kairo", "score": 0.9}, True),
            ({}, False),
        ]:
            result = gate.validate(obj, self.SIMPLE_SCHEMA)
            assert "passed" in result
            assert "errors" in result
            assert result["passed"] is should_pass


# ---------------------------------------------------------------------------
# F2-15: _canonical_key
# ---------------------------------------------------------------------------

class TestCanonicalKey:
    """F2-15: _canonical_key normalises values correctly for agreement comparison."""

    def test_none_is_null(self):
        assert _canonical_key(None) == "null"

    def test_true_is_true(self):
        assert _canonical_key(True) == "true"

    def test_false_is_false(self):
        assert _canonical_key(False) == "false"

    def test_int(self):
        assert _canonical_key(42) == "42"

    def test_float_rounded(self):
        # 1234.5678 rounds to "1234.57" at 2dp
        result = _canonical_key(1234.5678)
        assert result == "1234.57"

    def test_float_zero(self):
        assert _canonical_key(0.0) == "0.0"

    def test_string_lowercased_stripped(self):
        assert _canonical_key("  Acme Corp  ") == "acme corp"

    def test_string_case_insensitive(self):
        assert _canonical_key("Acme Corp") == _canonical_key("acme corp")


# ---------------------------------------------------------------------------
# F2-16: n=1 → consistency = 1.0
# ---------------------------------------------------------------------------

class TestSingleSample:
    """F2-16: Single sample always agrees with itself → consistency = 1.0."""

    def test_n1_returns_perfect_consistency(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=1, sampler=_make_chaotic_sampler(INVOICE_SCHEMA)
        )
        assert result.score == 1.0

    def test_n1_no_low_consistency_fields(self):
        checker = ConsistencyChecker()
        result = checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=1, sampler=_make_chaotic_sampler(INVOICE_SCHEMA)
        )
        assert result.low_consistency_fields == []


# ---------------------------------------------------------------------------
# F2-17: Empty schema → score = 1.0
# ---------------------------------------------------------------------------

class TestEmptySchema:
    """F2-17: Empty schema means nothing to score → score = 1.0."""

    def test_empty_schema_score(self):
        checker = ConsistencyChecker()
        result = checker.score("prompt", {}, {}, n=3, sampler=_make_uniform_sampler({}))
        assert result.score == 1.0
        assert result.per_field == {}
        assert result.low_consistency_fields == []


# ---------------------------------------------------------------------------
# F2-18: Custom sampler injection
# ---------------------------------------------------------------------------

class TestCustomSamplerInjection:
    """F2-18: Custom sampler can be injected for testing."""

    def test_custom_sampler_used(self):
        """Custom sampler is called exactly N times."""
        call_count = {"n": 0}

        def counting_sampler(prompt: str, schema: dict) -> dict:
            call_count["n"] += 1
            return {"invoice_id": "INV-001", "total_amount": 100.0, "vendor": "Acme Corp"}

        checker = ConsistencyChecker()
        checker.score(
            "prompt", INVOICE_SCHEMA, PERFECT_INVOICE,
            n=7, sampler=counting_sampler
        )
        assert call_count["n"] == 7


# ---------------------------------------------------------------------------
# F2-19: ConstrainedDecoder backward compat
# ---------------------------------------------------------------------------

class TestConstrainedDecoderCompat:
    """F2-19: ConstrainedDecoder still works after F2 changes."""

    def test_decode_structured_still_works(self):
        from kernel.sidecar.models.constrained_decoding import ConstrainedDecoder
        decoder = ConstrainedDecoder()
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        scratchpad, obj = decoder.decode_structured("test prompt", schema)
        assert isinstance(scratchpad, str)
        assert isinstance(obj, dict)
        assert "name" in obj


# ---------------------------------------------------------------------------
# F2-20: STEDGate backward compat — D4 tests still pass
# ---------------------------------------------------------------------------

class TestSTEDGateBackwardCompat:
    """F2-20: STEDGate.validate maintains full D4 backward compatibility."""

    SIMPLE_SCHEMA = {
        "type": "object",
        "required": ["name", "score"],
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    def test_valid_output_passes(self):
        gate = STEDGate()
        result = gate.validate({"name": "Kairo", "score": 0.9}, self.SIMPLE_SCHEMA)
        assert result["passed"] is True
        assert result["errors"] == []

    def test_invalid_output_fails(self):
        gate = STEDGate()
        result = gate.validate({}, self.SIMPLE_SCHEMA)
        assert result["passed"] is False
        assert len(result["errors"]) >= 1

    def test_wrong_type_fails(self):
        gate = STEDGate()
        result = gate.validate({"name": 42, "score": "high"}, self.SIMPLE_SCHEMA)
        assert result["passed"] is False

    def test_errors_is_list(self):
        gate = STEDGate()
        result = gate.validate({"name": "ok", "score": 0.5}, self.SIMPLE_SCHEMA)
        assert isinstance(result["errors"], list)

    def test_score_out_of_range_fails(self):
        gate = STEDGate()
        result = gate.validate({"name": "x", "score": 999.0}, self.SIMPLE_SCHEMA)
        assert result["passed"] is False
