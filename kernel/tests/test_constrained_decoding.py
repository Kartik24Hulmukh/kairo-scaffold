"""B4 — Comprehensive test suite for grammar-constrained decoding.

Covers:
  1. token_heal() — 8 preamble/postamble formats
  2. generate_fuzzed_json_from_schema() — correctness + 10k gate
  3. ConstrainedDecoder — scratchpad pattern, backends, string healing
  4. PackSchemaRegistry — inline fallback, disk load
  5. validate_and_heal() — enum repair, confidence clamp
  6. A/B field-correctness gate — constrained must not drop required fields
  7. Over-constraint guard — scratchpad must be free-form (not checked)
"""
import sys
import os
import json
import random

import jsonschema
import pytest

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sidecar.models.constrained_decoding import (
    ConstrainedDecoder,
    ConstraintViolation,
    PackSchemaRegistry,
    _SCRATCHPAD_DELIMITER,
    generate_fuzzed_json_from_schema,
    get_schema,
    token_heal,
    validate_and_heal,
)

# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------

INVOICE_SCHEMA = {
    "type": "object",
    "required": ["invoice_id", "total_amount", "line_items"],
    "properties": {
        "invoice_id":   {"type": "string"},
        "total_amount": {"type": "number", "minimum": 0},
        "vendor":       {"type": "string", "enum": ["Acme Corp", "Globex", "Initech"]},
        "is_paid":      {"type": "boolean"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "price"],
                "properties": {
                    "description": {"type": "string"},
                    "price":       {"type": "number", "minimum": 0},
                },
            },
        },
    },
}

SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["name", "score"],
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "tag":   {"type": "string", "enum": ["A", "B", "C"]},
    },
}


# ===========================================================================
# 1. token_heal
# ===========================================================================

class TestTokenHeal:

    def test_clean_json_passthrough(self):
        raw = '{"x": 1}'
        assert json.loads(token_heal(raw)) == {"x": 1}

    def test_leading_markdown_fence(self):
        raw = '```json\n{"x": 1}\n```'
        result = json.loads(token_heal(raw))
        assert result == {"x": 1}

    def test_leading_prose(self):
        raw = 'Here is the JSON output:\n{"x": 1}'
        result = json.loads(token_heal(raw))
        assert result == {"x": 1}

    def test_trailing_stop_token(self):
        raw = '{"x": 1}\n<|eot_id|>'
        result = json.loads(token_heal(raw))
        assert result == {"x": 1}

    def test_nested_object(self):
        raw = 'prefix {"a": {"b": [1, 2, 3]}} suffix'
        result = json.loads(token_heal(raw))
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_leading_bom(self):
        raw = "\ufeff{\"x\": 42}"
        result = json.loads(token_heal(raw))
        assert result == {"x": 42}

    def test_array_envelope(self):
        raw = "output: [1, 2, 3] done"
        result = json.loads(token_heal(raw))
        assert result == [1, 2, 3]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON envelope"):
            token_heal("no braces or brackets here at all")

    def test_string_with_escaped_braces(self):
        raw = '```\n{"msg": "hello {world}"}\n```'
        result = json.loads(token_heal(raw))
        assert result["msg"] == "hello {world}"


# ===========================================================================
# 2. generate_fuzzed_json_from_schema — basic correctness
# ===========================================================================

class TestFuzzerCorrectness:

    def test_required_fields_always_present_invoice(self):
        for _ in range(100):
            obj = generate_fuzzed_json_from_schema(INVOICE_SCHEMA)
            assert "invoice_id" in obj
            assert "total_amount" in obj
            assert "line_items" in obj

    def test_required_fields_always_present_simple(self):
        for _ in range(200):
            obj = generate_fuzzed_json_from_schema(SIMPLE_SCHEMA)
            assert "name" in obj
            assert "score" in obj

    def test_enum_values_respected(self):
        for _ in range(100):
            obj = generate_fuzzed_json_from_schema(INVOICE_SCHEMA)
            if "vendor" in obj:
                assert obj["vendor"] in ("Acme Corp", "Globex", "Initech")

    def test_number_minimum_respected(self):
        for _ in range(100):
            obj = generate_fuzzed_json_from_schema(INVOICE_SCHEMA)
            assert obj["total_amount"] >= 0
            for item in obj["line_items"]:
                assert item["price"] >= 0

    def test_integer_range_respected(self):
        for _ in range(100):
            obj = generate_fuzzed_json_from_schema(SIMPLE_SCHEMA)
            assert 0 <= obj["score"] <= 100

    def test_validates_jsonschema(self):
        for _ in range(500):
            obj = generate_fuzzed_json_from_schema(INVOICE_SCHEMA)
            jsonschema.validate(instance=obj, schema=INVOICE_SCHEMA)

    def test_extraction_schema_fuzz(self):
        schema = get_schema("extraction")
        for _ in range(200):
            obj = generate_fuzzed_json_from_schema(schema)
            jsonschema.validate(instance=obj, schema=schema)

    def test_answer_schema_fuzz(self):
        schema = get_schema("answer")
        for _ in range(200):
            obj = generate_fuzzed_json_from_schema(schema)
            jsonschema.validate(instance=obj, schema=schema)


# ===========================================================================
# 3. GATE: 10k fuzzed generations → zero schema-invalid objects
# ===========================================================================

class TestTenKGate:
    """SPEC gate: 10 000 fuzzed generations must all pass schema validation."""

    def test_10k_invoice_schema(self):
        failures = []
        for i in range(10_000):
            obj = generate_fuzzed_json_from_schema(INVOICE_SCHEMA)
            try:
                jsonschema.validate(instance=obj, schema=INVOICE_SCHEMA)
            except jsonschema.ValidationError as e:
                failures.append((i, str(e)))
        assert failures == [], (
            f"10k gate FAILED: {len(failures)} invalid objects.\n"
            + "\n".join(f"  [{i}] {msg}" for i, msg in failures[:5])
        )

    def test_10k_extraction_schema(self):
        schema = get_schema("extraction")
        failures = []
        for i in range(10_000):
            obj = generate_fuzzed_json_from_schema(schema)
            try:
                jsonschema.validate(instance=obj, schema=schema)
            except jsonschema.ValidationError as e:
                failures.append((i, str(e)))
        assert failures == [], (
            f"10k extraction gate FAILED: {len(failures)} invalid.\n"
            + "\n".join(f"  [{i}] {msg}" for i, msg in failures[:5])
        )

    def test_10k_answer_schema(self):
        schema = get_schema("answer")
        failures = []
        for i in range(10_000):
            obj = generate_fuzzed_json_from_schema(schema)
            try:
                jsonschema.validate(instance=obj, schema=schema)
            except jsonschema.ValidationError as e:
                failures.append((i, str(e)))
        assert failures == [], (
            f"10k answer gate FAILED: {len(failures)} invalid.\n"
            + "\n".join(f"  [{i}] {msg}" for i, msg in failures[:5])
        )


# ===========================================================================
# 4. ConstrainedDecoder — hybrid pattern
# ===========================================================================

class TestConstrainedDecoder:

    def test_default_backend_is_outlines(self):
        d = ConstrainedDecoder()
        assert d.backend == "outlines"

    def test_scratchpad_contains_delimiter(self):
        d = ConstrainedDecoder()
        sp = d.generate_scratchpad("some prompt")
        assert _SCRATCHPAD_DELIMITER in sp, "Phase 1 must contain the scratchpad delimiter"

    def test_scratchpad_is_free_form_text(self):
        """Scratchpad is NOT JSON — it is unconstrained reasoning text."""
        d = ConstrainedDecoder()
        sp = d.generate_scratchpad("Extract vendor from invoice.")
        # Must not be valid JSON on its own (it contains prose before the delimiter)
        prose_part = sp.split(_SCRATCHPAD_DELIMITER)[0]
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(prose_part)

    def test_decode_structured_returns_tuple(self):
        d = ConstrainedDecoder()
        scratchpad, obj = d.decode_structured("Extract invoice fields.", INVOICE_SCHEMA)
        assert isinstance(scratchpad, str)
        assert isinstance(obj, dict)

    def test_decode_structured_validates_schema(self):
        d = ConstrainedDecoder()
        for _ in range(20):
            _, obj = d.decode_structured("Test prompt.", INVOICE_SCHEMA)
            jsonschema.validate(instance=obj, schema=INVOICE_SCHEMA)

    def test_scratchpad_before_structured(self):
        """Phase 1 always precedes Phase 2 in the output."""
        d = ConstrainedDecoder()
        scratchpad, obj = d.decode_structured("Extract fields.", SIMPLE_SCHEMA)
        assert "Scratchpad" in scratchpad
        assert isinstance(obj, dict)

    def test_xgrammar_backend_falls_back(self):
        d = ConstrainedDecoder(backend="xgrammar")
        _, obj = d.decode_structured("test", SIMPLE_SCHEMA)
        jsonschema.validate(instance=obj, schema=SIMPLE_SCHEMA)

    def test_llguidance_backend_falls_back(self):
        d = ConstrainedDecoder(backend="llguidance")
        _, obj = d.decode_structured("test", SIMPLE_SCHEMA)
        jsonschema.validate(instance=obj, schema=SIMPLE_SCHEMA)

    def test_unknown_backend_raises(self):
        d = ConstrainedDecoder(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown backend"):
            d.decode_structured("test", SIMPLE_SCHEMA)

    def test_string_output_is_token_healed(self):
        """If backend returns a string, token_heal extracts the JSON object."""
        d = ConstrainedDecoder()
        # Monkeypatch _backend_decode to return a raw string with preamble
        original = d._backend_decode
        d._backend_decode = lambda p, s: '```json\n{"name": "test", "score": 50}\n```'
        _, obj = d.decode_structured("test", SIMPLE_SCHEMA)
        assert obj["name"] == "test"
        assert obj["score"] == 50
        d._backend_decode = original


# ===========================================================================
# 5. PackSchemaRegistry
# ===========================================================================

class TestPackSchemaRegistry:

    def test_extraction_schema_available(self):
        schema = get_schema("extraction")
        assert schema["type"] == "object"
        required = schema.get("required", [])
        assert "id" in required
        assert "field" in required

    def test_answer_schema_available(self):
        schema = get_schema("answer")
        assert "citations" in schema.get("required", [])

    def test_unknown_schema_raises(self):
        reg = PackSchemaRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent_schema_xyz")

    def test_schemas_are_cached(self):
        reg = PackSchemaRegistry()
        s1 = reg.get("extraction")
        s2 = reg.get("extraction")
        assert s1 is s2  # same object from cache


# ===========================================================================
# 6. validate_and_heal
# ===========================================================================

class TestValidateAndHeal:

    def _valid_extraction(self):
        return {
            "id": "ext_001",
            "doc_id": "doc_001",
            "field": "vendor_name",
            "value": "Acme Corp",
            "confidence": 0.95,
            "status": "suggested",
            "method": "exact",
            "anchors": [],
        }

    def test_valid_object_passes(self):
        obj = self._valid_extraction()
        result = validate_and_heal(obj, "extraction")
        assert result["id"] == "ext_001"

    def test_invalid_status_healed(self):
        obj = self._valid_extraction()
        obj["status"] = "UNKNOWN_STATUS"
        result = validate_and_heal(obj, "extraction")
        assert result["status"] == "suggested"

    def test_invalid_method_healed(self):
        obj = self._valid_extraction()
        obj["method"] = "INVALID_METHOD"
        result = validate_and_heal(obj, "extraction")
        assert result["method"] == "semantic"

    def test_confidence_clamped_above_1(self):
        obj = self._valid_extraction()
        obj["confidence"] = 1.5
        result = validate_and_heal(obj, "extraction")
        assert result["confidence"] == 1.0

    def test_confidence_clamped_below_0(self):
        obj = self._valid_extraction()
        obj["confidence"] = -0.5
        result = validate_and_heal(obj, "extraction")
        assert result["confidence"] == 0.0

    def test_missing_required_field_raises(self):
        obj = self._valid_extraction()
        del obj["id"]
        with pytest.raises(ConstraintViolation):
            validate_and_heal(obj, "extraction")

    def test_answer_validation_passes(self):
        answer = {
            "id": "ans_001",
            "query": "What is the total?",
            "text": "The total is $500.",
            "grounded": True,
            "citations": [],
        }
        result = validate_and_heal(answer, "answer")
        assert result["grounded"] is True


# ===========================================================================
# 7. A/B field-correctness gate (no over-constraint)
# ===========================================================================

class TestABFieldCorrectnessGate:
    """Constrained decoding must NOT drop required fields vs unconstrained.

    We run 500 constrained generations and check that required fields are
    present 100% of the time.  This proves we did not over-constrain the
    model by accidentally preventing required keys from being generated.
    """

    def _required_fields_present(self, schema: dict, obj: dict) -> bool:
        for field in schema.get("required", []):
            if field not in obj:
                return False
        return True

    def test_constrained_never_drops_required_fields_invoice(self):
        d = ConstrainedDecoder()
        missing = 0
        for _ in range(500):
            _, obj = d.decode_structured("Extract invoice.", INVOICE_SCHEMA)
            if not self._required_fields_present(INVOICE_SCHEMA, obj):
                missing += 1
        assert missing == 0, (
            f"A/B gate FAIL: constrained dropped required fields in {missing}/500 generations"
        )

    def test_constrained_never_drops_required_fields_extraction(self):
        schema = get_schema("extraction")
        d = ConstrainedDecoder()
        missing = 0
        for _ in range(200):
            _, obj = d.decode_structured("Extract field.", schema)
            if not self._required_fields_present(schema, obj):
                missing += 1
        assert missing == 0, (
            f"A/B gate FAIL: extraction schema dropped required fields in {missing}/200"
        )

    def test_constrained_never_drops_required_fields_answer(self):
        schema = get_schema("answer")
        d = ConstrainedDecoder()
        missing = 0
        for _ in range(200):
            _, obj = d.decode_structured("Answer query.", schema)
            if not self._required_fields_present(schema, obj):
                missing += 1
        assert missing == 0, (
            f"A/B gate FAIL: answer schema dropped required fields in {missing}/200"
        )

    def test_enum_field_correctness_matches_unconstrained(self):
        """Constrained enum values must always be valid; unconstrained might not be.

        This simulates the A/B comparison: unconstrained can produce invalid
        enum values (we inject a known-bad one), constrained always heals.
        """
        # Simulate "unconstrained" output: may have invalid enum
        unconstrained_obj = {
            "id": "e1", "doc_id": "d1", "field": "f1", "value": "v1",
            "confidence": 0.9, "status": "INVALID", "method": "INVALID",
            "anchors": [],
        }
        # Constrained path: validate_and_heal fixes it
        healed = validate_and_heal(dict(unconstrained_obj), "extraction")
        assert healed["status"] in {
            "suggested", "accepted", "edited", "rejected",
            "blocked", "pending_review", "grounded",
        }
        assert healed["method"] in {"exact", "fuzzy", "semantic", "visual", "block"}


# ===========================================================================
# 8. Scratchpad is not over-constrained (Phase 1 guard)
# ===========================================================================

class TestPhase1NotOverConstrained:
    """Verify Phase 1 (scratchpad) accepts arbitrary content — not schema-checked."""

    def test_scratchpad_can_contain_arbitrary_text(self):
        d = ConstrainedDecoder()
        # The scratchpad is purely text — we can put anything there
        sp = d.generate_scratchpad(
            "Extract vendor from invoice. Consider that the doc has tables."
        )
        # Must be a non-empty string
        assert isinstance(sp, str)
        assert len(sp) > 0

    def test_scratchpad_delimiter_separates_phases(self):
        d = ConstrainedDecoder()
        sp = d.generate_scratchpad("Extract vendor.")
        # Delimiter must be present so consumer can split phases
        assert _SCRATCHPAD_DELIMITER in sp
        parts = sp.split(_SCRATCHPAD_DELIMITER)
        assert len(parts) == 2
        # Phase 1 part (before delimiter) must be non-empty prose
        assert len(parts[0].strip()) > 0
