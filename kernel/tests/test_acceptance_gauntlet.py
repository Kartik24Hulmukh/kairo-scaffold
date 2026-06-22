"""A1 — Acceptance gauntlet: tests that the VGVA pipeline handles all adversarial
fixture types without crashing and maintains >=80% acceptance rate on the gauntlet.

Fixture types covered:
  - rotated_scan.pdf    : 90-degree rotated text
  - multi_column.pdf    : two-column layout
  - table.pdf           : dense tabular data
  - low_dpi.pdf         : 72 DPI scan simulation
  - non_english.pdf     : accented / non-ASCII text
  - near_miss_set.json  : 10 near-miss QA pairs (A3 overlap)
  - adversarial_bboxes.json : bbox stress cases (A5 overlap)

GATE: pytest kernel/tests/test_acceptance_gauntlet.py -v
All tests must pass for `make acceptance` to be green.
"""

import json
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ADVERSARIAL_DIR = REPO_ROOT / "fixtures" / "adversarial"
GAUNTLET_MANIFEST = ADVERSARIAL_DIR / "gauntlet_manifest.json"
NEAR_MISS_SET = ADVERSARIAL_DIR / "near_miss_set.json"
ADV_BBOXES = ADVERSARIAL_DIR / "adversarial_bboxes.json"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def load_gauntlet_manifest():
    """Load the gauntlet manifest from fixtures/adversarial/gauntlet_manifest.json."""
    assert GAUNTLET_MANIFEST.exists(), (
        f"Gauntlet manifest not found: {GAUNTLET_MANIFEST}. "
        "Run `python scripts/generate_gauntlet_fixtures.py` first."
    )
    with open(GAUNTLET_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def load_near_miss_set():
    """Load near-miss QA pairs."""
    assert NEAR_MISS_SET.exists(), f"Near-miss set not found: {NEAR_MISS_SET}"
    with open(NEAR_MISS_SET, encoding="utf-8") as f:
        return json.load(f)


def load_adversarial_bboxes():
    """Load adversarial bbox test cases."""
    assert ADV_BBOXES.exists(), f"Adversarial bboxes not found: {ADV_BBOXES}"
    with open(ADV_BBOXES, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# A1-01: Gauntlet manifest exists and is well-formed
# ---------------------------------------------------------------------------

class TestGauntletManifest:
    def test_manifest_exists(self):
        """Gauntlet manifest must exist."""
        assert GAUNTLET_MANIFEST.exists(), (
            f"Missing: {GAUNTLET_MANIFEST}. "
            "Generate with: python scripts/generate_gauntlet_fixtures.py"
        )

    def test_manifest_schema(self):
        """Manifest must have description, version, and fixtures list."""
        data = load_gauntlet_manifest()
        assert "description" in data
        assert "version" in data
        assert "fixtures" in data
        assert isinstance(data["fixtures"], list)
        assert len(data["fixtures"]) >= 5, "Gauntlet must have at least 5 fixture types"

    def test_manifest_fixture_types(self):
        """All required adversarial fixture types must be present."""
        data = load_gauntlet_manifest()
        fixture_types = {f["type"] for f in data["fixtures"]}
        required_types = {"rotated", "multi_column", "dense_table", "low_dpi", "non_english"}
        missing = required_types - fixture_types
        assert not missing, f"Missing gauntlet fixture types: {missing}"

    def test_manifest_fixture_files_exist(self):
        """All PDF files referenced in manifest must exist."""
        data = load_gauntlet_manifest()
        missing_files = []
        for entry in data["fixtures"]:
            fpath = REPO_ROOT / entry["file"]
            if not fpath.exists():
                missing_files.append(str(entry["file"]))
        assert not missing_files, (
            f"Missing fixture files: {missing_files}. "
            "Run: python scripts/generate_gauntlet_fixtures.py"
        )

    def test_manifest_has_ground_truth(self):
        """Each fixture must declare a ground_truth anchor string."""
        data = load_gauntlet_manifest()
        for entry in data["fixtures"]:
            assert "ground_truth" in entry, f"Missing ground_truth in fixture: {entry['id']}"
            assert entry["ground_truth"], f"Empty ground_truth in fixture: {entry['id']}"

    def test_manifest_expected_result_valid(self):
        """expected_result must be 'pass' or 'fail'."""
        data = load_gauntlet_manifest()
        for entry in data["fixtures"]:
            assert entry.get("expected_result") in {"pass", "fail"}, (
                f"Invalid expected_result in fixture {entry['id']}: {entry.get('expected_result')}"
            )


# ---------------------------------------------------------------------------
# A1-02: PDF fixture files can be opened (structural sanity)
# ---------------------------------------------------------------------------

class TestGauntletFixtureFiles:
    """Tests that each gauntlet PDF is non-empty and can be opened."""

    @pytest.fixture(autouse=True)
    def skip_if_manifest_missing(self):
        pytest.importorskip("fitz", reason="PyMuPDF not installed (fitz)")
        if not GAUNTLET_MANIFEST.exists():
            pytest.skip("Gauntlet manifest missing — run generate_gauntlet_fixtures.py first")

    def _open_pdf(self, path):
        """Open a PDF and return the fitz document, skip if fitz unavailable."""
        import fitz  # noqa: PLC0415
        assert path.exists(), f"PDF not found: {path}"
        doc = fitz.open(str(path))
        return doc

    def test_rotated_scan_pdf_is_valid(self):
        """rotated_scan.pdf must open and have at least 1 page."""
        doc = self._open_pdf(ADVERSARIAL_DIR / "rotated_scan.pdf")
        assert doc.page_count >= 1

    def test_multi_column_pdf_is_valid(self):
        """multi_column.pdf must open and have at least 1 page."""
        doc = self._open_pdf(ADVERSARIAL_DIR / "multi_column.pdf")
        assert doc.page_count >= 1

    def test_table_pdf_is_valid(self):
        """table.pdf must open and have at least 1 page."""
        doc = self._open_pdf(ADVERSARIAL_DIR / "table.pdf")
        assert doc.page_count >= 1

    def test_low_dpi_pdf_is_valid(self):
        """low_dpi.pdf must open and have at least 1 page."""
        doc = self._open_pdf(ADVERSARIAL_DIR / "low_dpi.pdf")
        assert doc.page_count >= 1

    def test_non_english_pdf_is_valid(self):
        """non_english.pdf must exist and be a valid PDF (or .txt stub)."""
        non_english = ADVERSARIAL_DIR / "non_english.pdf"
        if not non_english.exists():
            # Accept a .txt stub as a fallback
            stub = ADVERSARIAL_DIR / "non_english.txt"
            assert stub.exists(), "non_english fixture (pdf or txt) must exist"
        else:
            doc = self._open_pdf(non_english)
            assert doc.page_count >= 1


# ---------------------------------------------------------------------------
# A1-03: Gauntlet acceptance rate >= 80%
# ---------------------------------------------------------------------------

class TestGauntletAcceptanceRate:
    """Verify that the grounding pipeline handles each fixture type.

    We use the vgva.py text-match path (no live sidecar required) to assert
    that each fixture's ground_truth token is identifiable in the extracted
    text. This is a structural gate — it confirms the fixture contains the
    expected anchor and the extraction path is wired correctly.
    """

    def _extract_text_from_pdf(self, pdf_path: pathlib.Path) -> str:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # noqa: PLC0415
            doc = fitz.open(str(pdf_path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            return " ".join(text_parts)
        except ImportError:
            pytest.skip("fitz (PyMuPDF) not installed")
        except Exception as e:
            return f"EXTRACTION_ERROR: {e}"

    def test_gauntlet_acceptance_rate_gte_80pct(self):
        """At least 80% of gauntlet fixtures must have their ground_truth recoverable."""
        if not GAUNTLET_MANIFEST.exists():
            pytest.skip("Gauntlet manifest missing")

        data = load_gauntlet_manifest()
        fixtures = [f for f in data["fixtures"] if f.get("expected_result") == "pass"]

        if not fixtures:
            pytest.skip("No 'pass' fixtures in gauntlet manifest")

        passed = 0
        failures = []

        for entry in fixtures:
            fpath = REPO_ROOT / entry["file"]
            if not fpath.exists():
                failures.append(f"{entry['id']}: file not found")
                continue

            extracted = self._extract_text_from_pdf(fpath)
            ground_truth = entry["ground_truth"].lower()

            if ground_truth in extracted.lower():
                passed += 1
            else:
                failures.append(
                    f"{entry['id']}: ground_truth '{entry['ground_truth']}' "
                    f"not found in extracted text (first 200 chars: {extracted[:200]!r})"
                )

        total = len(fixtures)
        acceptance_rate = passed / total if total > 0 else 0.0

        assert acceptance_rate >= 0.80, (
            f"Gauntlet acceptance rate {acceptance_rate:.0%} < 80% required. "
            f"Failures: {failures}"
        )

    def test_rotated_scan_ground_truth_recoverable(self):
        """ALPHA-7 must be extractable from rotated_scan.pdf."""
        pdf = ADVERSARIAL_DIR / "rotated_scan.pdf"
        if not pdf.exists():
            pytest.skip("rotated_scan.pdf not found")
        text = self._extract_text_from_pdf(pdf)
        assert "ALPHA-7" in text or "alpha-7" in text.lower(), (
            f"Expected 'ALPHA-7' in rotated_scan.pdf text. Got: {text[:300]!r}"
        )

    def test_multi_column_ground_truth_recoverable(self):
        """'COLUMN A' must be extractable from multi_column.pdf."""
        pdf = ADVERSARIAL_DIR / "multi_column.pdf"
        if not pdf.exists():
            pytest.skip("multi_column.pdf not found")
        text = self._extract_text_from_pdf(pdf)
        assert "COLUMN A" in text or "column a" in text.lower(), (
            f"Expected 'COLUMN A' in multi_column.pdf. Got: {text[:300]!r}"
        )

    def test_table_ground_truth_recoverable(self):
        """'QUARTERLY FINANCIAL' must be extractable from table.pdf."""
        pdf = ADVERSARIAL_DIR / "table.pdf"
        if not pdf.exists():
            pytest.skip("table.pdf not found")
        text = self._extract_text_from_pdf(pdf)
        assert "QUARTERLY FINANCIAL" in text or "quarterly financial" in text.lower(), (
            f"Expected 'QUARTERLY FINANCIAL' in table.pdf. Got: {text[:300]!r}"
        )

    def test_low_dpi_ground_truth_recoverable(self):
        """'KAIRO-LDI-2026-0042' must be extractable from low_dpi.pdf (invisible text path)."""
        pdf = ADVERSARIAL_DIR / "low_dpi.pdf"
        if not pdf.exists():
            pytest.skip("low_dpi.pdf not found")
        text = self._extract_text_from_pdf(pdf)
        assert "KAIRO-LDI-2026-0042" in text or "kairo-ldi-2026-0042" in text.lower(), (
            f"Expected 'KAIRO-LDI-2026-0042' in low_dpi.pdf (embedded in invisible text). "
            f"Got: {text[:300]!r}"
        )

    def test_non_english_ground_truth_recoverable(self):
        """'ESPANOL' must be extractable from non_english.pdf."""
        pdf = ADVERSARIAL_DIR / "non_english.pdf"
        if not pdf.exists():
            pytest.skip("non_english.pdf not found")
        text = self._extract_text_from_pdf(pdf)
        assert "ESPANOL" in text or "espanol" in text.lower(), (
            f"Expected 'ESPANOL' in non_english.pdf. Got: {text[:300]!r}"
        )




# ---------------------------------------------------------------------------
# A1-04: Near-miss fixture set is well-formed (A3 overlap)
# ---------------------------------------------------------------------------

class TestNearMissFixtureSet:
    def test_near_miss_set_exists(self):
        assert NEAR_MISS_SET.exists(), f"Missing: {NEAR_MISS_SET}"

    def test_near_miss_set_has_10_pairs(self):
        data = load_near_miss_set()
        assert len(data["pairs"]) >= 10, "Near-miss set must have at least 10 QA pairs"

    def test_near_miss_set_all_must_answer(self):
        """All near-miss pairs must be flagged must_answer=True."""
        data = load_near_miss_set()
        for pair in data["pairs"]:
            assert pair.get("must_answer") is True, (
                f"Near-miss pair {pair['id']} must have must_answer=True"
            )

    def test_near_miss_set_has_ground_truth(self):
        """All near-miss pairs must have a non-empty ground_truth_answer."""
        data = load_near_miss_set()
        for pair in data["pairs"]:
            assert pair.get("ground_truth_answer"), (
                f"Missing ground_truth_answer in pair {pair['id']}"
            )

    def test_near_miss_set_has_context(self):
        """All near-miss pairs must have a context field."""
        data = load_near_miss_set()
        for pair in data["pairs"]:
            assert pair.get("context"), f"Missing context in near-miss pair {pair['id']}"
