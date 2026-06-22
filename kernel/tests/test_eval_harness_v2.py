"""Tests for upgraded bench/eval_harness.py (C4 - Eval Harness v2)."""

import json
import pathlib
import sys
import os

# Make bench/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bench.eval_harness import (
    EvalHarness,
    EvalReport,
    compute_answer_relevance,
    compute_citation_correctness,
    _is_grounded
)

def test_compute_answer_relevance():
    # Overlapping non-stopwords
    q = "What is the start date of the ACME contract?"
    ans = "The ACME contract start date is June 1, 2026."
    score = compute_answer_relevance(q, ans)
    assert score > 0.0
    
    # Exact mismatch
    assert compute_answer_relevance("Question about apples", "Answer about oranges") == 0.0

def test_compute_citation_correctness():
    chunks = [
        "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."
    ]
    # Good citation
    assert compute_citation_correctness(["laws of the State of Delaware"], chunks) == 1.0
    # Hallucinated citation
    assert compute_citation_correctness(["laws of the State of California"], chunks) == 0.0
    # Mix of correct and incorrect
    assert compute_citation_correctness(["laws of the State of Delaware", "State of California"], chunks) == 0.5

def test_eval_harness_v2_metrics(tmp_path):
    history = tmp_path / "history.jsonl"
    harness = EvalHarness(history_file=history)

    qa_pairs = [
        {
            "question": "What is the contract start date?",
            "expected_answer": "January 1, 2024",
            "context_chunks": [
                "The contract commences on January 1, 2024 and runs for twelve months."
            ],
            "model_answer": "The contract commences on January 1, 2024.",
            "citations": ["contract commences on January 1, 2024"],
        }
    ]

    report = harness.run(qa_pairs)
    assert isinstance(report, EvalReport)
    assert report.faithfulness == 1.0
    assert report.answer_relevance > 0.0
    assert report.citation_correctness == 1.0
    assert report.refusal_correctness == 1.0

def test_is_grounded_punctuation_tolerance():
    chunks = ["The contract commences on January 1, 2024."]
    # The answer has a dot at the end, and different casing/punctuation
    answer = "The contract commences on January 1, 2024."
    assert _is_grounded(answer, chunks) is True
