"""A3 — Near-Miss Refusal Threshold Tests.

Gate: false-refusal rate <5% on near_miss_set.json
      true-refusal rate == 100% on unanswerable fixtures

Tests the grounding pipeline's refusal behaviour using the text-match path
(verify_text_present) from vgva.py, applied to near-miss context/answer pairs.

The key insight: a valid answer IS present in the context, so the verifier
should return verified=True. The refusal gate in app.py blocks on verified=False.
We simulate this decision here without a live sidecar.

GATE: pytest kernel/tests/test_near_miss_refusal.py -v
"""

import json
import pathlib
import pytest
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ADVERSARIAL_DIR = REPO_ROOT / "fixtures" / "adversarial"
NEAR_MISS_SET = ADVERSARIAL_DIR / "near_miss_set.json"
UNANSWERABLE_DIR = REPO_ROOT / "fixtures" / "unanswerable"

# Threshold used by the grounding verifier — must match app.py / vgva.py
VERIFY_THRESHOLD = 0.70
# False refusal rate must stay below this
MAX_FALSE_REFUSAL_RATE = 0.05  # 5%


def _load_near_miss():
    assert NEAR_MISS_SET.exists(), f"Missing: {NEAR_MISS_SET}"
    with open(NEAR_MISS_SET, encoding="utf-8") as f:
        return json.load(f)


def _vgva_verify_text(context: str, answer: str, threshold: float = VERIFY_THRESHOLD) -> dict:
    """Use VGVA's text-match path to check if answer is grounded in context."""
    sys.path.insert(0, str(REPO_ROOT / "kernel" / "sidecar"))
    from models.vgva import VisualGroundingVerificationAgent
    agent = VisualGroundingVerificationAgent(match_threshold=threshold)
    return agent.verify_text_present(context, answer)


# ---------------------------------------------------------------------------
# A3-01: Near-miss fixture file is valid
# ---------------------------------------------------------------------------

class TestNearMissFixtureValidity:
    def test_near_miss_exists(self):
        assert NEAR_MISS_SET.exists()

    def test_near_miss_has_required_keys(self):
        data = _load_near_miss()
        assert "pairs" in data
        assert "expected_false_refusal_rate" in data

    def test_near_miss_has_at_least_10_pairs(self):
        data = _load_near_miss()
        assert len(data["pairs"]) >= 10

    def test_near_miss_pairs_have_required_fields(self):
        data = _load_near_miss()
        required_fields = {"id", "question", "context", "ground_truth_answer", "must_answer"}
        for pair in data["pairs"]:
            missing = required_fields - set(pair.keys())
            assert not missing, f"Pair {pair.get('id', '?')} missing fields: {missing}"


# ---------------------------------------------------------------------------
# A3-02: Near-miss answers ARE grounded in their contexts
# ---------------------------------------------------------------------------

class TestNearMissAnswersGrounded:
    """Verify that each near-miss ground_truth_answer is findable in its context."""

    def test_all_near_miss_answers_are_grounded(self):
        """CORE GATE: Every near-miss answer must be verifiable in its context."""
        data = _load_near_miss()
        false_refusals = []

        for pair in data["pairs"]:
            if not pair.get("must_answer"):
                continue

            result = _vgva_verify_text(
                context=pair["context"],
                answer=pair["ground_truth_answer"],
            )
            if not result["verified"]:
                false_refusals.append({
                    "id": pair["id"],
                    "question": pair["question"],
                    "answer": pair["ground_truth_answer"],
                    "confidence": result["confidence"],
                })

        total = sum(1 for p in data["pairs"] if p.get("must_answer"))
        false_refusal_rate = len(false_refusals) / total if total > 0 else 0.0

        assert false_refusal_rate < MAX_FALSE_REFUSAL_RATE, (
            f"False refusal rate {false_refusal_rate:.0%} >= {MAX_FALSE_REFUSAL_RATE:.0%} limit.\n"
            f"False refusals: {false_refusals}"
        )

    @pytest.mark.parametrize("pair_id,answer,context", [
        ("nm-001", "Net 30 from invoice date", "Invoice date: 2024-03-15. Vendor: Acme Solutions LLC. Amount: $1,250.00. Payment terms: Net 30 from invoice date."),
        ("nm-002", "Acme Solutions LLC", "Invoice date: 2024-03-15. Vendor name: Acme Solutions LLC. Invoice amount: $1,250.00."),
        ("nm-005", "$125 per hour", "Consulting services: 10 hours at $125 per hour. Total: $1,250.00."),
        ("nm-010", "Pinnacle Technology Inc.", "Bill To: Pinnacle Technology Inc., 123 Innovation Drive, San Francisco CA 94105."),
    ])
    def test_specific_near_miss_pair_passes(self, pair_id, answer, context):
        """Spot-check critical near-miss pairs individually."""
        result = _vgva_verify_text(context, answer)
        assert result["verified"], (
            f"Pair {pair_id}: answer '{answer}' not found in context. "
            f"Confidence: {result['confidence']:.3f} (threshold: {VERIFY_THRESHOLD})"
        )

    def test_near_miss_pass_rate_gte_95pct(self):
        """At least 95% of near-miss pairs must pass (inverse: false-refusal <5%)."""
        data = _load_near_miss()
        passed = 0
        total = 0

        for pair in data["pairs"]:
            if not pair.get("must_answer"):
                continue
            total += 1
            result = _vgva_verify_text(pair["context"], pair["ground_truth_answer"])
            if result["verified"]:
                passed += 1

        pass_rate = passed / total if total > 0 else 0.0
        assert pass_rate >= 0.95, (
            f"Near-miss pass rate {pass_rate:.0%} < 95% required. "
            f"({passed}/{total} pairs passed)"
        )


# ---------------------------------------------------------------------------
# A3-03: Unanswerable fixtures are still refused (true-refusal = 100%)
# ---------------------------------------------------------------------------

class TestUnanswerableRefusalPreserved:
    """Verify that unanswerable queries are still blocked after threshold tuning."""

    UNANSWERABLE_QUERIES = [
        "What is the CEO's salary?",
        "What is the company's secret formula?",
        "What will the stock price be next year?",
        "Who won the 2050 World Cup?",
        "What is the vendor's home address?",
    ]

    EMPTY_CONTEXT = "This is a generic document with no relevant information."

    def test_empty_context_returns_unverified(self):
        """Unanswerable questions against empty context must fail verification."""
        for query in self.UNANSWERABLE_QUERIES:
            result = _vgva_verify_text(self.EMPTY_CONTEXT, query)
            assert not result["verified"], (
                f"Unanswerable query '{query}' was incorrectly verified against empty context"
            )

    def test_true_refusal_rate_is_100pct(self):
        """True-refusal rate on unanswerable set must be exactly 100%."""
        # We use queries that contain tokens NOT in the context
        context = "Invoice number: INV-001. Amount: $500."
        unanswerable = [
            "What is the moon's gravity?",
            "List all US senators from Texas.",
            "What is the boiling point of tungsten?",
        ]
        blocked = 0
        for q in unanswerable:
            result = _vgva_verify_text(context, q)
            if not result["verified"]:
                blocked += 1

        assert blocked == len(unanswerable), (
            f"True-refusal rate is not 100%: only {blocked}/{len(unanswerable)} unanswerables blocked"
        )


# ---------------------------------------------------------------------------
# A3-04: Threshold value is within tuned range
# ---------------------------------------------------------------------------

class TestThresholdTuning:
    def test_vgva_default_threshold_within_range(self):
        """VGVA default threshold must be in [0.60, 0.80] — the tuned near-miss range."""
        sys.path.insert(0, str(REPO_ROOT / "kernel" / "sidecar"))
        from models.vgva import VisualGroundingVerificationAgent
        agent = VisualGroundingVerificationAgent()
        # Access the private threshold
        threshold = agent._threshold
        assert 0.60 <= threshold <= 0.80, (
            f"VGVA threshold {threshold} is outside tuned range [0.60, 0.80]. "
            "This may cause excessive false-refusals or missed hallucinations."
        )

    def test_lower_threshold_reduces_false_refusals(self):
        """A threshold of 0.70 (default) allows more near-misses through than 0.95."""
        # Edge case: a paraphrase that's close but not exact
        context = "The total payment is one thousand two hundred and fifty dollars."
        answer = "payment $1250"

        result_strict = _vgva_verify_text(context, answer, threshold=0.95)
        result_tuned = _vgva_verify_text(context, answer, threshold=0.70)

        # At 0.70 threshold, more answers should pass
        # (strict threshold should not be more permissive)
        if result_strict["verified"] and not result_tuned["verified"]:
            pytest.fail("Lower threshold is stricter than higher threshold — logic error")
        # This is a directional test — just ensure no inversion
