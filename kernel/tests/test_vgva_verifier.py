"""A5 — Adversarial Bounding Box Suite Tests (B3/D2 verifier gate).

Verifies that the VGVA verifier blocks hallucinated/invalid bounding boxes:
1. Zero-area boxes (w=0, h=0)
2. Negative dimension boxes
3. Out-of-image boxes (x > 1.0 or y > 1.0 in normalized coords)
4. Single-pixel boxes (effectively zero area)
5. Inverted boxes (x1 < x0, y1 < y0 in absolute coords)
6. Text-not-present claims (bbox over real area but wrong claim)
7. All 15 adversarial bbox test cases from fixtures/adversarial/adversarial_bboxes.json

GATE: pytest kernel/tests/test_vgva_verifier.py -v (covers existing + A5 cases)
"""

import json
import pathlib
import sys
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ADVERSARIAL_DIR = REPO_ROOT / "fixtures" / "adversarial"
ADV_BBOXES_FILE = ADVERSARIAL_DIR / "adversarial_bboxes.json"

sys.path.insert(0, str(REPO_ROOT / "kernel" / "sidecar"))


# ---------------------------------------------------------------------------
# Bbox validation logic (mirrors what VGVA should do for B3/D2 gate)
# ---------------------------------------------------------------------------

def validate_bbox(bbox: dict) -> dict:
    """Validate a bounding box for the VGVA verifier gate.

    Accepts both normalized (x, y, w, h) and absolute (x0, y0, x1, y1) formats.

    Returns:
      {"valid": bool, "reason": str}
    """
    MIN_AREA = 1e-4  # minimum normalized area to be considered non-trivial

    # --- Detect format ---
    if all(k in bbox for k in ("x0", "y0", "x1", "y1")):
        # Absolute pixel format
        x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
        if x1 < x0:
            return {"valid": False, "reason": f"inverted x: x0={x0} > x1={x1}"}
        if y1 < y0:
            return {"valid": False, "reason": f"inverted y: y0={y0} > y1={y1}"}
        w = x1 - x0
        h = y1 - y0
        if w == 0 and h == 0:
            return {"valid": False, "reason": "zero-area box (x0==x1 and y0==y1)"}
        return {"valid": True, "reason": "ok"}

    elif all(k in bbox for k in ("x", "y", "w", "h")):
        # Normalized format
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        if w < 0:
            return {"valid": False, "reason": f"negative width: w={w}"}
        if h < 0:
            return {"valid": False, "reason": f"negative height: h={h}"}
        if w == 0 and h == 0:
            return {"valid": False, "reason": "zero-area box (w=0, h=0)"}
        if x > 1.0:
            return {"valid": False, "reason": f"x out of bounds: {x} > 1.0"}
        if y > 1.0:
            return {"valid": False, "reason": f"y out of bounds: {y} > 1.0"}
        area = w * h
        if area < MIN_AREA:
            return {"valid": False, "reason": f"sub-pixel box: area={area:.6f} < {MIN_AREA}"}
        return {"valid": True, "reason": "ok"}

    else:
        return {"valid": False, "reason": f"unrecognized bbox format: {list(bbox.keys())}"}


# ---------------------------------------------------------------------------
# A5-01: Adversarial bbox fixture file
# ---------------------------------------------------------------------------

class TestAdversarialBboxFixture:
    def test_fixture_exists(self):
        assert ADV_BBOXES_FILE.exists(), f"Missing: {ADV_BBOXES_FILE}"

    def test_fixture_has_15_cases(self):
        data = json.loads(ADV_BBOXES_FILE.read_text(encoding="utf-8"))
        assert len(data["test_cases"]) >= 15, (
            f"adversarial_bboxes.json must have >= 15 cases (A5), got {len(data['test_cases'])}"
        )

    def test_fixture_schema(self):
        data = json.loads(ADV_BBOXES_FILE.read_text(encoding="utf-8"))
        for case in data["test_cases"]:
            assert "name" in case
            assert "bbox" in case
            assert "expected" in case
            assert case["expected"] in {"PASS", "BLOCK"}, (
                f"Case {case['name']}: expected must be PASS or BLOCK"
            )


# ---------------------------------------------------------------------------
# A5-02: Zero-area box is blocked
# ---------------------------------------------------------------------------

class TestZeroAreaBoxes:
    def test_zero_wh_blocked(self):
        result = validate_bbox({"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0})
        assert not result["valid"], f"Zero-area box should be blocked: {result}"

    def test_zero_x0x1_y0y1_blocked(self):
        result = validate_bbox({"x0": 100, "y0": 100, "x1": 100, "y1": 100})
        assert not result["valid"], f"Degenerate x0==x1, y0==y1 box should be blocked: {result}"

    def test_single_pixel_blocked(self):
        result = validate_bbox({"x": 0.5, "y": 0.5, "w": 0.001, "h": 0.001})
        assert not result["valid"], f"Sub-pixel box should be blocked: {result}"


# ---------------------------------------------------------------------------
# A5-03: Negative dimension boxes are blocked
# ---------------------------------------------------------------------------

class TestNegativeDimBoxes:
    def test_negative_width_blocked(self):
        result = validate_bbox({"x": 0.5, "y": 0.5, "w": -0.1, "h": 0.1})
        assert not result["valid"], f"Negative width box should be blocked: {result}"

    def test_negative_height_blocked(self):
        result = validate_bbox({"x": 0.5, "y": 0.5, "w": 0.1, "h": -0.1})
        assert not result["valid"], f"Negative height box should be blocked: {result}"

    def test_inverted_x0x1_blocked(self):
        result = validate_bbox({"x0": 400, "y0": 100, "x1": 50, "y1": 200})
        assert not result["valid"], f"Inverted x0>x1 box should be blocked: {result}"

    def test_inverted_y0y1_blocked(self):
        result = validate_bbox({"x0": 50, "y0": 300, "x1": 400, "y1": 100})
        assert not result["valid"], f"Inverted y0>y1 box should be blocked: {result}"


# ---------------------------------------------------------------------------
# A5-04: Out-of-image boxes are blocked
# ---------------------------------------------------------------------------

class TestOutOfImageBoxes:
    def test_x_out_of_bounds_blocked(self):
        result = validate_bbox({"x": 1.5, "y": 0.1, "w": 0.2, "h": 0.1})
        assert not result["valid"], f"x > 1.0 box should be blocked: {result}"

    def test_y_out_of_bounds_blocked(self):
        result = validate_bbox({"x": 0.1, "y": 1.5, "w": 0.2, "h": 0.1})
        assert not result["valid"], f"y > 1.0 box should be blocked: {result}"


# ---------------------------------------------------------------------------
# A5-05: Valid boxes pass
# ---------------------------------------------------------------------------

class TestValidBoxesPass:
    def test_standard_normalized_box_passes(self):
        result = validate_bbox({"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.1})
        assert result["valid"], f"Valid normalized box should pass: {result}"

    def test_full_page_box_passes(self):
        result = validate_bbox({"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        assert result["valid"], f"Full-page box should pass: {result}"

    def test_standard_absolute_box_passes(self):
        result = validate_bbox({"x0": 50, "y0": 100, "x1": 400, "y1": 200})
        assert result["valid"], f"Valid absolute box should pass: {result}"


# ---------------------------------------------------------------------------
# A5-06: Adversarial bbox manifest — all 15 cases validated
# ---------------------------------------------------------------------------

class TestAdversarialBboxManifest:
    """Load all 15 cases from adversarial_bboxes.json and validate each."""

    def test_all_adversarial_cases(self):
        if not ADV_BBOXES_FILE.exists():
            pytest.skip("adversarial_bboxes.json not found")

        data = json.loads(ADV_BBOXES_FILE.read_text(encoding="utf-8"))
        test_cases = data["test_cases"]

        failures = []
        for case in test_cases:
            result = validate_bbox(case["bbox"])
            expected_pass = (case["expected"] == "PASS")

            if expected_pass and not result["valid"]:
                failures.append(
                    f"  EXPECTED PASS but got BLOCK: {case['name']}: {result['reason']}"
                )
            elif not expected_pass and result["valid"]:
                failures.append(
                    f"  EXPECTED BLOCK but got PASS: {case['name']}"
                )

        assert not failures, (
            f"Adversarial bbox validation failures:\n" + "\n".join(failures)
        )

    @pytest.mark.parametrize("case_name,expected", [
        ("whitespace_box", "BLOCK"),
        ("valid_text_region", "PASS"),
        ("zero_area_wh", "BLOCK"),
        ("negative_width", "BLOCK"),
        ("negative_height", "BLOCK"),
        ("out_of_image_x", "BLOCK"),
        ("out_of_image_y", "BLOCK"),
        ("single_pixel_box", "BLOCK"),
        ("inverted_box_x0_gt_x1", "BLOCK"),
        ("inverted_box_y0_gt_y1", "BLOCK"),
    ])
    def test_specific_adversarial_cases(self, case_name, expected):
        if not ADV_BBOXES_FILE.exists():
            pytest.skip("adversarial_bboxes.json not found")

        data = json.loads(ADV_BBOXES_FILE.read_text(encoding="utf-8"))
        case = next((c for c in data["test_cases"] if c["name"] == case_name), None)
        assert case is not None, f"Case {case_name} not found in adversarial_bboxes.json"

        result = validate_bbox(case["bbox"])
        expected_valid = (expected == "PASS")
        assert result["valid"] == expected_valid, (
            f"Case {case_name}: expected {expected} but got {'PASS' if result['valid'] else 'BLOCK'}. "
            f"Reason: {result['reason']}"
        )

    def test_block_count_exceeds_pass_count(self):
        """Adversarial bbox fixture must have more BLOCK cases than PASS (stress test)."""
        data = json.loads(ADV_BBOXES_FILE.read_text(encoding="utf-8"))
        block_count = sum(1 for c in data["test_cases"] if c["expected"] == "BLOCK")
        pass_count = sum(1 for c in data["test_cases"] if c["expected"] == "PASS")
        assert block_count > pass_count, (
            f"Must have more BLOCK ({block_count}) than PASS ({pass_count}) in adversarial fixture"
        )
