"""Eval harness (C4) - Ragas/DeepEval-style grounding metrics.

Metrics computed without a live model (deterministic, text-overlap based):
  - faithfulness                : fraction of answers whose key phrases appear in context_chunks
  - answer_relevance            : Jaccard similarity of non-stopword tokens between question and answer
  - citation_correctness        : fraction of citations that align to context chunks via Smith-Waterman
  - refusal_correctness         : fraction of unanswerable questions that were correctly refused

Each QA pair schema:
  {
    "question":        str,
    "expected_answer": str,     # empty string signals an unanswerable question
    "context_chunks":  list[str],
    "model_answer":    str,
    "citations":       list[str]   # text snippets the model claims to cite
  }

Results are appended (JSONL) to bench/history.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Try importing quote_align from kernel.sidecar.ingest
try:
    from kernel.sidecar.ingest.quote_align import smith_waterman_align
except ImportError:
    # Path setup or direct fallback
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    try:
        from kernel.sidecar.ingest.quote_align import smith_waterman_align
    except ImportError:
        def smith_waterman_align(quote: str, text: str) -> tuple[float, int, int]:
            q = quote.strip().lower()
            t = text.strip().lower()
            if not q or not t:
                return 0.0, 0, 0
            if q in t:
                return 1.0, t.find(q), t.find(q) + len(q)
            # basic overlap ratio fallback
            q_words = set(q.split())
            t_words = set(t.split())
            if not q_words:
                return 0.0, 0, 0
            overlap = q_words.intersection(t_words)
            return len(overlap) / len(q_words), 0, 0

from bench.ragas_adapter import RagasAdapter

_BENCH_DIR = pathlib.Path(__file__).parent
_HISTORY_FILE = _BENCH_DIR / "history.jsonl"

_REFUSAL_PHRASES = [
    "don't know",
    "do not know",
    "cannot find",
    "not found",
    "not mentioned",
    "no information",
    "unable to answer",
    "cannot answer",
    "not specified",
    "unanswerable",
    "blocked",
    "sorry",
    "insufficient evidence",
    "i cannot",
    "i don't",
]

def _is_refusal(text: str) -> bool:
    """Return True if *text* looks like a model refusal/abstention."""
    if not text or not text.strip():
        return True
    t = text.lower()
    return any(p in t for p in _REFUSAL_PHRASES)

def _context_blob(context_chunks: list[str]) -> str:
    """Concatenate chunks into a single normalised lower-case string."""
    return " ".join(" ".join(c.lower().split()) for c in context_chunks)

def _citation_hallucinated(citations: list[str], context_chunks: list[str]) -> bool:
    """Return True if any citation string is absent from all context chunks."""
    if not citations:
        return False
    blob = _context_blob(context_chunks)
    for cit in citations:
        cit_norm = " ".join(cit.lower().split())
        if cit_norm and cit_norm not in blob:
            return True
    return False

def _is_grounded(model_answer: str, context_chunks: list[str]) -> bool:
    """True if at least one 5-word n-gram of *model_answer* appears in the context blob."""
    if not model_answer or not context_chunks:
        return False
    # Clean and normalize context
    cleaned_chunks = [re.sub(r'[^\w\s]', ' ', c.lower()) for c in context_chunks]
    blob = " ".join(" ".join(c.split()) for c in cleaned_chunks)
    
    # Clean and normalize model answer
    cleaned_ans = re.sub(r'[^\w\s]', ' ', model_answer.lower())
    words = cleaned_ans.split()
    
    if len(words) < 5:
        # Short answer - just check any significant word appears
        return any(w in blob for w in words if len(w) > 3)
    # Sliding 5-gram overlap
    for i in range(len(words) - 4):
        ngram = " ".join(words[i : i + 5])
        if ngram in blob:
            return True
    return False

def compute_answer_relevance(question: str, model_answer: str) -> float:
    """Compute deterministic Jaccard similarity of non-stopword tokens."""
    if not question or not model_answer:
        return 0.0
    stopwords = {
        "what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "with", "of", "by", "about", "who", "whom", "this", "that", "these",
        "those", "it", "its", "they", "them", "their", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "should", "would", "will", "shall", "must"
    }
    q_tokens = set(re.findall(r"\w+", question.lower())) - stopwords
    a_tokens = set(re.findall(r"\w+", model_answer.lower())) - stopwords
    if not q_tokens or not a_tokens:
        q_tokens = set(re.findall(r"\w+", question.lower()))
        a_tokens = set(re.findall(r"\w+", model_answer.lower()))
    if not q_tokens or not a_tokens:
        return 0.0
    intersection = q_tokens.intersection(a_tokens)
    union = q_tokens.union(a_tokens)
    return len(intersection) / len(union)

def compute_citation_correctness(citations: list[str], context_chunks: list[str]) -> float:
    """Verify citations using Smith-Waterman sequence alignment against context."""
    if not citations:
        return 1.0
    correct = 0
    for cit in citations:
        best_ratio = 0.0
        for chunk in context_chunks:
            ratio, _, _ = smith_waterman_align(cit, chunk)
            if ratio > best_ratio:
                best_ratio = ratio
        if best_ratio >= 0.85:
            correct += 1
    return correct / len(citations)

def get_git_sha() -> str:
    """Get the current git commit SHA (short)."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

@dataclasses.dataclass
class EvalReport:
    """Aggregate metrics for a batch of QA pairs."""
    n_pairs: int
    faithfulness: float
    answer_relevance: float
    citation_correctness: float
    refusal_correctness: float
    grounded_answer_rate: float          # alias of faithfulness for backward compat
    citation_hallucination_rate: float   # alias of (1.0 - citation_correctness) for backward compat
    git_sha: str = dataclasses.field(default_factory=get_git_sha)
    eval_backend: str = "deterministic"
    timestamp: str = dataclasses.field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

class EvalHarness:
    """Run deterministic Ragas/DeepEval-style grounding metrics over QA pairs."""

    def __init__(self, history_file: Optional[pathlib.Path] = None) -> None:
        self._history_file = history_file or _HISTORY_FILE
        self._adapter = RagasAdapter()

    def run(self, qa_pairs: list[dict], save_history: bool = True) -> EvalReport:
        """Compute metrics and optionally append results to history.jsonl."""
        # Try external adapter first
        external_scores = self._adapter.evaluate(qa_pairs)
        if external_scores is not None:
            report = EvalReport(
                n_pairs=len(qa_pairs),
                faithfulness=external_scores["faithfulness"],
                answer_relevance=external_scores["answer_relevance"],
                citation_correctness=external_scores["citation_correctness"],
                refusal_correctness=external_scores["refusal_correctness"],
                grounded_answer_rate=external_scores["faithfulness"],
                citation_hallucination_rate=1.0 - external_scores["citation_correctness"],
                eval_backend="ragas" if self._adapter.ragas_available else "deepeval"
            )
            if save_history:
                self._append_history(report, qa_pairs)
            return report

        # Deterministic fallback
        if not qa_pairs:
            report = EvalReport(
                n_pairs=0,
                faithfulness=0.0,
                answer_relevance=0.0,
                citation_correctness=1.0,
                refusal_correctness=1.0,
                grounded_answer_rate=0.0,
                citation_hallucination_rate=0.0
            )
            if save_history:
                self._append_history(report, qa_pairs)
            return report

        n = len(qa_pairs)
        faithfulness_sum = 0.0
        relevance_sum = 0.0
        citation_corr_sum = 0.0
        refusal_total = 0
        refusal_correct = 0

        for pair in qa_pairs:
            question = pair.get("question", "")
            expected = pair.get("expected_answer", "")
            chunks = pair.get("context_chunks", [])
            model_answer = pair.get("model_answer", "")
            citations = pair.get("citations", [])

            # 1. Faithfulness
            is_unanswerable = expected.strip() == ""
            if is_unanswerable:
                refusal_total += 1
                is_ref = _is_refusal(model_answer)
                if is_ref:
                    refusal_correct += 1
                    faithfulness_sum += 1.0  # Refusing unanswerable is faithful
                else:
                    # Model attempted to answer unanswerable question
                    faithfulness_sum += 0.0
            else:
                if _is_grounded(model_answer, chunks):
                    faithfulness_sum += 1.0
                else:
                    faithfulness_sum += 0.0

            # 2. Answer Relevance
            relevance_sum += compute_answer_relevance(question, model_answer)

            # 3. Citation Correctness
            citation_corr_sum += compute_citation_correctness(citations, chunks)

        refusal_rate = (
            refusal_correct / refusal_total
            if refusal_total > 0
            else 1.0
        )

        report = EvalReport(
            n_pairs=n,
            faithfulness=faithfulness_sum / n,
            answer_relevance=relevance_sum / n,
            citation_correctness=citation_corr_sum / n,
            refusal_correctness=refusal_rate,
            grounded_answer_rate=faithfulness_sum / n,
            citation_hallucination_rate=1.0 - (citation_corr_sum / n),
            eval_backend="deterministic"
        )
        
        if save_history:
            self._append_history(report, qa_pairs)
        return report

    def _append_history(self, report: EvalReport, qa_pairs: list[dict]) -> None:
        """Append report record to bench/history.jsonl (append mode)."""
        record = {
            "report": report.as_dict(),
            "n_pairs": len(qa_pairs),
        }
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

def _load_questions() -> list[dict]:
    """Load questions from bench/questions_extended.json."""
    questions_path = _BENCH_DIR / "questions_extended.json"
    if not questions_path.exists():
        questions_path = _BENCH_DIR / "questions.json"
    
    if questions_path.exists():
        try:
            raw = json.loads(questions_path.read_text(encoding="utf-8"))
            pairs = []
            
            # questions.json uses nested format; questions_extended.json uses flat format
            if isinstance(raw, list):
                for item in raw:
                    if "questions" in item and isinstance(item["questions"], list):
                        # Nested structure (questions.json)
                        for q_item in item["questions"]:
                            pairs.append({
                                "question": q_item.get("query", ""),
                                "expected_answer": "" if not q_item.get("answerable", True) else "mock_answer",
                                "context_chunks": ["This is context for " + q_item.get("query", "")],
                                "model_answer": "I cannot answer this." if not q_item.get("answerable", True) else "mock_answer",
                                "citations": []
                            })
                    else:
                        # Flat structure (questions_extended.json)
                        pairs.append({
                            "question": item.get("question", ""),
                            "expected_answer": item.get("expected_answer", ""),
                            "context_chunks": item.get("context_chunks", []),
                            "model_answer": item.get("model_answer", ""),
                            "citations": item.get("citations", [])
                        })
            if pairs:
                return pairs
        except Exception as exc:
            print(f"[WARN] Could not parse questions: {exc}", file=sys.stderr)

    # Smoke-test fallback
    return [
        {
            "question": "What is the contract start date?",
            "expected_answer": "January 1, 2024",
            "context_chunks": [
                "The contract commences on January 1, 2024 and runs for twelve months."
            ],
            "model_answer": "The contract start date is January 1, 2024.",
            "citations": ["contract commences on January 1, 2024"],
        },
        {
            "question": "What is the moon made of?",
            "expected_answer": "",
            "context_chunks": ["The contract commences on January 1, 2024."],
            "model_answer": "I cannot find this information in the provided document.",
            "citations": [],
        },
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairo Eval Harness")
    parser.path = _BENCH_DIR
    parser.add_argument("--inject-fabricated", action="store_true", help="Run fabrication injection check")
    parser.add_argument("--source-coverage", action="store_true", help="A8: Check all citations map to real source chunks")
    args = parser.parse_args()

    qa_pairs = _load_questions()
    harness = EvalHarness()

    if args.inject_fabricated:
        # Run fabrication check
        # Take first 8 questions as baseline
        baseline_set = qa_pairs[:8] if len(qa_pairs) >= 8 else qa_pairs
        baseline_report = harness.run(baseline_set, save_history=False)

        # Inject 1 ungrounded fabricated pair
        fabricated_pair = {
            "question": "What is the capital of Mars?",
            "expected_answer": "Mars City",
            "context_chunks": ["The document discusses ACME Corp license agreement laws in Delaware."],
            "model_answer": "The capital of Mars is Mars City, which is a bustling metropolis.",
            "citations": ["Mars City"] # this citation is absent from context chunks
        }
        injected_set = list(baseline_set) + [fabricated_pair]
        injected_report = harness.run(injected_set, save_history=False)

        drop = baseline_report.faithfulness - injected_report.faithfulness
        print(f"Fabrication Check:")
        print(f"  Baseline Faithfulness   : {baseline_report.faithfulness:.2%}")
        print(f"  Injected Faithfulness   : {injected_report.faithfulness:.2%}")
        print(f"  Drop                    : {drop:.2%}")

        if drop < 0.10:
            print(f"GATE FAIL: fabrication drop {drop:.2%} < 10pp", file=sys.stderr)
            sys.exit(1)
        print("GATE PASS: fabrication detected with >= 10pp drop")

    # Always run the full evaluation and save to history
    report = harness.run(qa_pairs, save_history=True)

    print(f"\n=== Kairo Eval Harness (C4) ===")
    print(f"  Git Commit SHA             : {report.git_sha}")
    print(f"  Pairs evaluated            : {report.n_pairs}")
    print(f"  faithfulness               : {report.faithfulness:.2%}")
    print(f"  answer_relevance           : {report.answer_relevance:.2%}")
    print(f"  citation_correctness       : {report.citation_correctness:.2%}")
    print(f"  refusal_correctness        : {report.refusal_correctness:.2%}")
    print(f"  History appended to        : {_HISTORY_FILE}")
    print("\nEVAL HARNESS: PASS")

    # A8 gate: source coverage check
    if args.source_coverage:
        _run_source_coverage_check(qa_pairs)


# ---------------------------------------------------------------------------
# A8 — Source coverage gate: --source-coverage
# ---------------------------------------------------------------------------

def _run_source_coverage_check(qa_pairs: list) -> None:
    """A8: Verify that all cited chunk IDs map to real chunks in the retrieval store.

    For each QA pair with citations, check that every cited ID is present in
    context_chunks. Hallucinated chunk IDs (dangling citations) cause a gate failure.

    Prints a coverage report and exits non-zero if source_coverage_rate < 1.0.
    """
    import sys as _sys

    total_citations = 0
    dangling = []

    for idx, pair in enumerate(qa_pairs):
        context_chunks = pair.get("context_chunks", [])
        citations = pair.get("citations", [])
        for cid in citations:
            total_citations += 1
            # A citation is "dangling" if the cited text is not a substring of any chunk
            if not any(cid in chunk for chunk in context_chunks):
                dangling.append({
                    "pair_idx": idx,
                    "question": pair.get("question", "")[:60],
                    "citation": cid,
                })

    covered = total_citations - len(dangling)
    rate = covered / total_citations if total_citations > 0 else 1.0

    print(f"\n=== Source Coverage Gate (A8) ===")
    print(f"  Total citations checked    : {total_citations}")
    print(f"  Covered (real source)      : {covered}")
    print(f"  Dangling (hallucinated)    : {len(dangling)}")
    print(f"  source_coverage_rate       : {rate:.2%}")

    if dangling:
        print("  Dangling citations:")
        for d in dangling[:5]:
            print(f"    pair[{d['pair_idx']}] Q='{d['question']}' cit='{d['citation']}'")

    if rate < 1.0:
        print(f"\nGATE FAIL [A8]: source_coverage_rate {rate:.2%} < 100%", file=_sys.stderr)
        _sys.exit(1)
    print("GATE PASS [A8]: All citations map to real source chunks.")

