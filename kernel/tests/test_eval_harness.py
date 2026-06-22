"""Tests for bench/eval_harness.py (C4 - Eval Harness)."""

import json
import pathlib
import sys
import os

# Make bench/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bench.eval_harness import (
    EvalHarness,
    EvalReport,
    _is_grounded,
    _is_refusal,
    _citation_hallucinated,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


def test_is_refusal_detects_cannot_find():
    assert _is_refusal("I cannot find this information in the document.") is True


def test_is_refusal_detects_empty_string():
    assert _is_refusal("") is True


def test_is_refusal_returns_false_for_real_answer():
    assert _is_refusal("The contract starts on January 1, 2024.") is False


def test_is_grounded_five_gram_match():
    chunks = ["The contract commences on January 1 2024 and runs for twelve months."]
    answer = "The contract commences on January 1 2024."
    assert _is_grounded(answer, chunks) is True


def test_is_grounded_returns_false_when_no_overlap():
    chunks = ["Total invoice amount is five hundred dollars."]
    answer = "The moon is made of cheese and craters."
    assert _is_grounded(answer, chunks) is False


def test_citation_hallucinated_absent_citation():
    chunks = ["The contract commences on January 1, 2024."]
    citations = ["invoice amount is five hundred"]  # not in chunks
    assert _citation_hallucinated(citations, chunks) is True


def test_citation_hallucinated_present_citation():
    chunks = ["The contract commences on January 1, 2024."]
    citations = ["contract commences on january 1, 2024"]
    assert _citation_hallucinated(citations, chunks) is False


def test_citation_hallucinated_empty_citations():
    chunks = ["Some document text."]
    assert _citation_hallucinated([], chunks) is False


# ---------------------------------------------------------------------------
# EvalHarness.run() integration tests
# ---------------------------------------------------------------------------


def test_run_returns_eval_report(tmp_path):
    history = tmp_path / "history.jsonl"
    harness = EvalHarness(history_file=history)

    qa_pairs = [
        {
            "question": "What is the start date?",
            "expected_answer": "January 1, 2024",
            "context_chunks": [
                "The contract commences on January 1 2024 and runs for twelve months."
            ],
            "model_answer": "The contract commences on January 1 2024.",
            "citations": ["contract commences on january 1 2024"],
        },
        {
            "question": "Who wrote this?",
            "expected_answer": "",  # unanswerable
            "context_chunks": ["The contract commences on January 1, 2024."],
            "model_answer": "I cannot find this information in the provided document.",
            "citations": [],
        },
    ]

    report = harness.run(qa_pairs)

    assert isinstance(report, EvalReport)
    assert report.n_pairs == 2
    # First pair is grounded (5-gram match), second is unanswerable refusal (also grounded)
    assert report.grounded_answer_rate == 1.0
    # No citations hallucinated
    assert report.citation_hallucination_rate == 0.0
    # Unanswerable question correctly refused
    assert report.refusal_correctness == 1.0


def test_run_appends_to_history_jsonl(tmp_path):
    history = tmp_path / "history.jsonl"
    harness = EvalHarness(history_file=history)

    pairs = [
        {
            "question": "What is the payment term?",
            "expected_answer": "Net 30",
            "context_chunks": ["Payment is due within Net 30 days of invoice."],
            "model_answer": "Payment is due within Net 30 days of invoice.",
            "citations": ["payment is due within net 30 days of invoice"],
        }
    ]

    harness.run(pairs)
    harness.run(pairs)  # run twice

    lines = history.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, "Two runs should produce exactly two JSONL lines"

    for line in lines:
        record = json.loads(line)
        assert "report" in record
        assert "n_pairs" in record
        assert record["n_pairs"] == 1


def test_run_detects_citation_hallucination(tmp_path):
    history = tmp_path / "history.jsonl"
    harness = EvalHarness(history_file=history)

    pairs = [
        {
            "question": "What did the author say?",
            "expected_answer": "Something real",
            "context_chunks": ["The author said something real."],
            "model_answer": "The author said something real.",
            "citations": ["invented phrase not in context at all"],
        }
    ]

    report = harness.run(pairs)
    assert report.citation_hallucination_rate == 1.0


def test_run_empty_pairs(tmp_path):
    history = tmp_path / "history.jsonl"
    harness = EvalHarness(history_file=history)

    report = harness.run([])
    assert report.n_pairs == 0
    assert report.grounded_answer_rate == 0.0

    # History file should still be written
    assert history.exists()
