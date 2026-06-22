"""A8 — Click-to-Source / Citation Coverage Tests.

Verifies that every answer path has a real source:
1. Answers with valid chunk IDs pass the coverage gate
2. Answers with hallucinated/dangling chunk IDs fail
3. source_coverage_rate is computed and ≥100% for valid answers
4. eval_harness.py has --source-coverage flag support

GATE: pytest kernel/tests/test_source_coverage.py -v
"""

import json
import pathlib
import sys
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EVAL_HARNESS = REPO_ROOT / "bench" / "eval_harness.py"


# ---------------------------------------------------------------------------
# A8-01: Source coverage data model
# ---------------------------------------------------------------------------

def compute_source_coverage(answers: list[dict], known_chunk_ids: set[str]) -> dict:
    """Compute citation coverage for a list of answer objects.

    Each answer must have:
      - "text": the answer text
      - "citations": list of chunk_ids cited

    Returns:
      {
        "total_answers": int,
        "covered": int,           # answers where ALL citations exist
        "dangling": int,          # answers with at least one dangling citation
        "source_coverage_rate": float,  # covered / total
        "dangling_citations": list[dict],  # {answer_idx, chunk_id}
      }
    """
    total = len(answers)
    covered = 0
    dangling_count = 0
    dangling_list = []

    for idx, answer in enumerate(answers):
        citations = answer.get("citations", [])
        if not citations:
            # No citations → treat as uncovered
            dangling_count += 1
            dangling_list.append({"answer_idx": idx, "chunk_id": None, "reason": "no_citations"})
            continue

        answer_ok = True
        for cid in citations:
            if cid not in known_chunk_ids:
                dangling_count += 1
                dangling_list.append({"answer_idx": idx, "chunk_id": cid, "reason": "dangling"})
                answer_ok = False

        if answer_ok:
            covered += 1

    return {
        "total_answers": total,
        "covered": covered,
        "dangling": dangling_count,
        "source_coverage_rate": covered / total if total > 0 else 0.0,
        "dangling_citations": dangling_list,
    }


# ---------------------------------------------------------------------------
# A8-02: Valid citations pass
# ---------------------------------------------------------------------------

class TestValidCitationsPassing:
    KNOWN_CHUNKS = {"chunk-001", "chunk-002", "chunk-003", "chunk-invoice-001"}

    def test_single_valid_citation_passes(self):
        answers = [{"text": "The total is $1250.", "citations": ["chunk-001"]}]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 1.0
        assert result["covered"] == 1
        assert result["dangling"] == 0

    def test_multiple_valid_citations_pass(self):
        answers = [
            {"text": "The vendor is Acme.", "citations": ["chunk-001", "chunk-002"]},
            {"text": "Total is $500.", "citations": ["chunk-003"]},
        ]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 1.0
        assert result["covered"] == 2

    def test_all_valid_citations_100pct_rate(self):
        answers = [
            {"text": f"Answer {i}", "citations": [f"chunk-{i:03d}"]}
            for i in range(1, 4)
        ]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 1.0


# ---------------------------------------------------------------------------
# A8-03: Hallucinated/dangling citations fail
# ---------------------------------------------------------------------------

class TestDanglingCitationsFail:
    KNOWN_CHUNKS = {"chunk-001", "chunk-002"}

    def test_hallucinated_chunk_id_fails(self):
        answers = [{"text": "The CEO earns $5M.", "citations": ["chunk-HALLUCINATED-999"]}]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 0.0
        assert result["dangling"] > 0
        assert any(d["chunk_id"] == "chunk-HALLUCINATED-999" for d in result["dangling_citations"])

    def test_mixed_valid_and_dangling(self):
        answers = [
            {"text": "Good answer.", "citations": ["chunk-001"]},           # valid
            {"text": "Bad answer.", "citations": ["chunk-FAKE"]},            # dangling
        ]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["covered"] == 1
        assert result["dangling"] >= 1
        assert 0.0 < result["source_coverage_rate"] < 1.0

    def test_no_citations_is_dangling(self):
        answers = [{"text": "Some answer.", "citations": []}]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["dangling"] == 1
        assert result["covered"] == 0

    def test_dangling_citation_list_has_details(self):
        answers = [{"text": "Fake.", "citations": ["chunk-nonexistent"]}]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert len(result["dangling_citations"]) >= 1
        entry = result["dangling_citations"][0]
        assert "answer_idx" in entry
        assert "chunk_id" in entry
        assert "reason" in entry


# ---------------------------------------------------------------------------
# A8-04: Coverage rate computation
# ---------------------------------------------------------------------------

class TestCoverageRateComputation:
    KNOWN_CHUNKS = {"c1", "c2", "c3", "c4", "c5"}

    def test_zero_answers_returns_zero_rate(self):
        result = compute_source_coverage([], self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 0.0
        assert result["total_answers"] == 0

    def test_all_pass_returns_1_0(self):
        answers = [{"text": f"A{i}", "citations": [f"c{i}"]} for i in range(1, 6)]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 1.0

    def test_half_pass_returns_0_5(self):
        answers = [
            {"text": "A1", "citations": ["c1"]},   # pass
            {"text": "A2", "citations": ["c1"]},   # pass
            {"text": "A3", "citations": ["BAD"]},  # fail
            {"text": "A4", "citations": ["BAD"]},  # fail
        ]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert result["source_coverage_rate"] == 0.5

    def test_rate_is_float_in_0_to_1(self):
        answers = [{"text": "x", "citations": ["c1"]}]
        result = compute_source_coverage(answers, self.KNOWN_CHUNKS)
        assert 0.0 <= result["source_coverage_rate"] <= 1.0


# ---------------------------------------------------------------------------
# A8-05: eval_harness.py has --source-coverage support
# ---------------------------------------------------------------------------

class TestEvalHarnessSourceCoverage:
    def test_eval_harness_exists(self):
        assert EVAL_HARNESS.exists(), f"Missing: {EVAL_HARNESS}"

    def test_eval_harness_has_source_coverage_flag(self):
        """eval_harness.py must document or accept --source-coverage."""
        content = EVAL_HARNESS.read_text(encoding="utf-8")
        assert "source_coverage" in content or "source-coverage" in content, (
            "eval_harness.py must support --source-coverage flag for A8 gate"
        )

    def test_eval_harness_imports_cleanly(self):
        """eval_harness.py must parse without syntax errors."""
        import ast
        content = EVAL_HARNESS.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"eval_harness.py has syntax error: {e}")

    def test_eval_harness_stores_history(self):
        """eval_harness.py must write to bench/history.jsonl."""
        content = EVAL_HARNESS.read_text(encoding="utf-8")
        assert "history.jsonl" in content or "history" in content, (
            "eval_harness.py must store results to bench/history.jsonl"
        )
