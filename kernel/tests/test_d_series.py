"""D-series stub tests — D1/D2/D3/D4 interface verification.

These tests confirm that each v2.2 forward stub:
  - Is importable from its canonical path.
  - Exposes the correct public interface (class name + method signatures).
  - Raises or returns the documented fallback without hard-crashing.

GATE:
    pytest kernel/tests/test_d_series.py -v
"""

import sys
import os
import pytest

# Ensure the repo root is on sys.path so imports resolve identically whether
# tests are run from the repo root or from inside kernel/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------------------------------------------------------
# D1 — LeannGraphIndex
# ---------------------------------------------------------------------------

from kernel.sidecar.retrieval.leann_index import LeannGraphIndex  # noqa: E402


class TestLeannGraphIndex:
    """Tests for D1: LEANN graph-ANNS index stub."""

    def test_class_is_importable(self):
        """LeannGraphIndex can be instantiated from its canonical module path."""
        idx = LeannGraphIndex()
        assert idx is not None

    def test_index_raises_not_implemented(self):
        """index() raises NotImplementedError with the install hint when leann is absent."""
        idx = LeannGraphIndex()
        with pytest.raises(NotImplementedError) as exc_info:
            idx.index([{"id": "c1", "embedding": [0.1, 0.2], "text": "hello"}])
        assert "LEANN not available" in str(exc_info.value)
        assert "kairo[leann]" in str(exc_info.value)

    def test_search_raises_not_implemented(self):
        """search() raises NotImplementedError with the install hint when leann is absent."""
        idx = LeannGraphIndex()
        with pytest.raises(NotImplementedError) as exc_info:
            idx.search([0.1, 0.2, 0.3], top_k=3)
        assert "LEANN not available" in str(exc_info.value)
        assert "kairo[leann]" in str(exc_info.value)

    def test_search_signature_accepts_top_k(self):
        """search() signature accepts query_emb and top_k positional/keyword args."""
        idx = LeannGraphIndex()
        # We only care that TypeError is NOT raised for the arguments themselves —
        # NotImplementedError is the expected inner outcome.
        with pytest.raises(NotImplementedError):
            idx.search(query_emb=[0.0] * 768, top_k=10)

    def test_index_signature_accepts_list_of_dicts(self):
        """index() signature accepts a list[dict] argument."""
        idx = LeannGraphIndex()
        with pytest.raises(NotImplementedError):
            idx.index([{"id": "x", "embedding": [0.0], "text": "t"}])


# ---------------------------------------------------------------------------
# D2 — VisualGroundingVerificationAgent
# ---------------------------------------------------------------------------

from kernel.sidecar.models.vgva import VisualGroundingVerificationAgent  # noqa: E402


class TestVisualGroundingVerificationAgent:
    """Tests for D2: VGVA visual-grounding stub."""

    DUMMY_BYTES = b"\x89PNG\r\n\x1a\n"  # minimal PNG magic bytes
    DUMMY_BBOX = {"x": 10, "y": 20, "w": 100, "h": 50}

    def test_class_is_importable(self):
        """VisualGroundingVerificationAgent can be instantiated."""
        agent = VisualGroundingVerificationAgent()
        assert agent is not None

    def test_verify_returns_dict(self):
        """verify() returns a dict regardless of whether vision deps are installed."""
        agent = VisualGroundingVerificationAgent()
        result = agent.verify(self.DUMMY_BYTES, "The total is $120", self.DUMMY_BBOX)
        assert isinstance(result, dict)

    def test_verify_keys_present(self):
        """verify() always returns 'verified', 'confidence', and 'method' keys."""
        agent = VisualGroundingVerificationAgent()
        result = agent.verify(self.DUMMY_BYTES, "The total is $120", self.DUMMY_BBOX)
        assert "verified" in result
        assert "confidence" in result
        assert "method" in result

    def test_verify_types(self):
        """verify() key types are bool, float, str."""
        agent = VisualGroundingVerificationAgent()
        result = agent.verify(self.DUMMY_BYTES, "claim", self.DUMMY_BBOX)
        assert isinstance(result["verified"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["method"], str)

    def test_verify_fallback_when_deps_absent(self):
        """verify() returns a valid method regardless of which vision deps are installed.

        Updated for E3 VGVA upgrade: method is now one of:
          - "not_available"       : PIL not installed
          - "ocr_pytesseract"     : PIL + pytesseract installed
          - "ocr_fallback_empty"  : PIL installed, pytesseract missing
          - "stub_not_implemented": legacy fallback (kept for backwards compat)
        In all cases, verified is a bool and confidence is in [0.0, 1.0].
        """
        agent = VisualGroundingVerificationAgent()
        result = agent.verify(self.DUMMY_BYTES, "claim", self.DUMMY_BBOX)
        VALID_METHODS = {
            "not_available", "stub_not_implemented",
            "ocr_pytesseract", "ocr_fallback_empty",
        }
        assert result["method"] in VALID_METHODS, (
            f"Unexpected method {result['method']!r}. Expected one of {VALID_METHODS}"
        )
        assert isinstance(result["verified"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_verify_confidence_in_range(self):
        """confidence is always in [0.0, 1.0]."""
        agent = VisualGroundingVerificationAgent()
        result = agent.verify(self.DUMMY_BYTES, "claim", self.DUMMY_BBOX)
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# D3 — ModelRouter
# ---------------------------------------------------------------------------

from kernel.sidecar.models.model_routing import ModelRouter  # noqa: E402


class TestModelRouter:
    """Tests for D3: two-tier worker/reasoner model router."""

    def test_class_is_importable(self):
        """ModelRouter can be instantiated."""
        router = ModelRouter()
        assert router is not None

    def test_simple_routes_to_worker(self):
        """'simple' complexity maps to the worker tier."""
        router = ModelRouter()
        assert router.route("simple") == "worker"

    def test_retrieval_routes_to_worker(self):
        """'retrieval' complexity maps to the worker tier."""
        router = ModelRouter()
        assert router.route("retrieval") == "worker"

    def test_reasoning_routes_to_reasoner(self):
        """'reasoning' complexity maps to the reasoner tier."""
        router = ModelRouter()
        assert router.route("reasoning") == "reasoner"

    def test_code_generation_routes_to_reasoner(self):
        """'code_generation' complexity maps to the reasoner tier."""
        router = ModelRouter()
        assert router.route("code_generation") == "reasoner"

    def test_synthesis_routes_to_reasoner(self):
        """'synthesis' complexity maps to the reasoner tier."""
        router = ModelRouter()
        assert router.route("synthesis") == "reasoner"

    def test_unknown_task_routes_to_reasoner(self):
        """Any unrecognised task label defaults to the reasoner tier."""
        router = ModelRouter()
        assert router.route("unknown_future_task") == "reasoner"

    def test_return_type_is_string(self):
        """route() always returns a str."""
        router = ModelRouter()
        result = router.route("simple")
        assert isinstance(result, str)

    def test_only_valid_tiers_returned(self):
        """route() only ever returns 'worker' or 'reasoner'."""
        router = ModelRouter()
        valid = {"worker", "reasoner"}
        for task in ("simple", "retrieval", "reasoning", "code_generation",
                     "synthesis", "multi_hop", "summarization"):
            assert router.route(task) in valid


# ---------------------------------------------------------------------------
# D4 — STEDGate
# ---------------------------------------------------------------------------

from kernel.sidecar.models.sted import STEDGate  # noqa: E402

SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["name", "score"],
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class TestSTEDGate:
    """Tests for D4: STED structured-output consistency gate."""

    def test_class_is_importable(self):
        """STEDGate can be instantiated."""
        gate = STEDGate()
        assert gate is not None

    def test_valid_output_passes(self):
        """A schema-conformant dict yields passed=True and empty errors."""
        gate = STEDGate()
        result = gate.validate({"name": "Kairo", "score": 0.9}, SIMPLE_SCHEMA)
        assert result["passed"] is True
        assert result["errors"] == []

    def test_invalid_output_fails(self):
        """A dict missing required keys yields passed=False with a non-empty errors list."""
        gate = STEDGate()
        result = gate.validate({}, SIMPLE_SCHEMA)
        assert result["passed"] is False
        assert len(result["errors"]) >= 1

    def test_wrong_type_fails(self):
        """A dict with wrong value type yields passed=False."""
        gate = STEDGate()
        result = gate.validate({"name": 42, "score": "high"}, SIMPLE_SCHEMA)
        assert result["passed"] is False

    def test_return_keys_always_present(self):
        """validate() always returns 'passed' and 'errors'."""
        gate = STEDGate()
        result = gate.validate({"name": "ok", "score": 0.5}, SIMPLE_SCHEMA)
        assert "passed" in result
        assert "errors" in result

    def test_errors_is_list(self):
        """'errors' is always a list."""
        gate = STEDGate()
        result = gate.validate({"name": "ok", "score": 0.5}, SIMPLE_SCHEMA)
        assert isinstance(result["errors"], list)

    def test_score_out_of_range_fails(self):
        """A numeric value violating minimum/maximum fails validation."""
        gate = STEDGate()
        result = gate.validate({"name": "x", "score": 999.0}, SIMPLE_SCHEMA)
        assert result["passed"] is False
        assert len(result["errors"]) >= 1

    def test_additional_properties_pass(self):
        """Extra keys not disallowed by schema do not cause failure."""
        gate = STEDGate()
        result = gate.validate(
            {"name": "Kairo", "score": 0.5, "extra": "ignored"}, SIMPLE_SCHEMA
        )
        assert result["passed"] is True

    def test_empty_schema_passes_any_dict(self):
        """An empty schema ({}) accepts any dict."""
        gate = STEDGate()
        result = gate.validate({"anything": True}, {})
        assert result["passed"] is True
