"""F1 — Two-Tier Inference Router (NVIDIA arXiv:2506.02153).

Reference: "Small Language Models are the Future of Agentic AI" (NVIDIA, 2026).
Core insight: 40–70% of agentic queries can be resolved by a small, fast local
SLM (Tier-1) without quality loss. Harder queries are escalated to a larger
local model (Tier-2). A cheap difficulty classifier routes between tiers.

Architecture:
    query → DifficultyClassifier → score ∈ [0.0, 1.0]
                                  │
          ┌───────────────────────┴──────────────────────┐
      score < θ                                      score ≥ θ
    Tier-1 (worker)                              Tier-2 (reasoner)
    SLM 3–8B                                     LLM 8B+
    fast, cheap                                  higher quality
    ~80%+ of traffic                             ~20% or less
          │                                           │
          └───────────────────────────────────────────┘
                              │
                    grounding verifier
          (model-independent — no model client import in vgva.py)

DifficultyClassifier signals (cheap, rule-based — no neural cost):
    1. Token length: queries > TOKEN_LENGTH_THRESHOLD → escalate
    2. Multi-hop keywords: "compare", "contrast", "synthesize", "across all" → +score
    3. Negation depth: "except", "unless", "not counting", "exclude" → +score
    4. Retrieval confidence: low chunk score passed as context → escalate
    5. Ambiguity markers: "or", "either", "ambiguous", "unclear" → +score

The classifier output is a continuous difficulty score ∈ [0.0, 1.0].
The routing threshold θ is tunable (default: 0.5, via KAIRO_TIER_THRESHOLD env var).
A lower θ keeps more queries in Tier-1 (faster, cheaper); raise θ for higher quality.

Gate:
    pytest kernel/tests/test_tiered_router.py -v
    ≥80% of 20 synthetic golden queries route to Tier-1 at default θ.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Routing threshold. Queries with difficulty ≥ threshold → Tier-2.
TIER_THRESHOLD: float = float(os.environ.get("KAIRO_TIER_THRESHOLD", "0.50"))

#: Token length above which a query is considered harder (approximate word count).
TOKEN_LENGTH_HARD: int = int(os.environ.get("KAIRO_TIER1_TOKEN_LIMIT", "80"))

#: Valid tier names.
VALID_TIERS: frozenset[str] = frozenset({"worker", "reasoner"})

# ---------------------------------------------------------------------------
# Multi-hop / negation / ambiguity keyword patterns
# ---------------------------------------------------------------------------

_MULTI_HOP_PATTERNS = re.compile(
    r"\b(compare|contrast|synthesize|aggregate|cross[- ]reference|"
    r"across all|in both|between .{1,30} and|summarize .{1,40} from|"
    r"list all|enumerate|multi[-\s]?hop|chain of|step[-\s]by[-\s]step reasoning)\b",
    re.IGNORECASE,
)

_NEGATION_PATTERNS = re.compile(
    r"\b(except|unless|not counting|excluding|without|other than|"
    r"but not|ignore|disregard|omit|subtract)\b",
    re.IGNORECASE,
)

_AMBIGUITY_PATTERNS = re.compile(
    r"\b(ambiguous|unclear|unsure|either|or could be|might be|"
    r"it depends|it's unclear|multiple interpretations)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# DifficultyClassifier
# ---------------------------------------------------------------------------


@dataclass
class DifficultySignals:
    """Structured difficulty signals for a query.

    Attributes:
        raw_score: Final difficulty score ∈ [0.0, 1.0].
        token_length: Approximate token/word count.
        multi_hop_match: True if multi-hop keywords detected.
        negation_match: True if negation depth keywords detected.
        ambiguity_match: True if ambiguity markers detected.
        retrieval_confidence: Optional retrieval score passed by caller.
        tier: Routing decision: "worker" or "reasoner".
    """

    raw_score: float
    token_length: int
    multi_hop_match: bool
    negation_match: bool
    ambiguity_match: bool
    retrieval_confidence: Optional[float]
    tier: str


class DifficultyClassifier:
    """Cheap difficulty classifier for Tier-1/Tier-2 routing.

    Computes a continuous difficulty score ∈ [0.0, 1.0] from rule-based
    signals. No neural cost — runs in microseconds.

    Usage::

        clf = DifficultyClassifier(threshold=0.5)
        signals = clf.classify("What is the invoice total?")
        assert signals.tier == "worker"

        signals = clf.classify("Compare the warranty clauses across all contracts.")
        assert signals.tier == "reasoner"
    """

    def __init__(self, threshold: float = TIER_THRESHOLD) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def classify(
        self,
        query: str,
        retrieval_confidence: Optional[float] = None,
    ) -> DifficultySignals:
        """Classify a query and return routing signals.

        Args:
            query: The user query string.
            retrieval_confidence: Optional retrieval score ∈ [0.0, 1.0].
                                  A score < 0.30 is treated as a hard-query signal.

        Returns:
            DifficultySignals with .tier == "worker" or "reasoner".
        """
        score = 0.0

        # Signal 1: Token length (approximate word count)
        token_count = len(query.split())
        length_score = min(1.0, token_count / TOKEN_LENGTH_HARD)
        score += length_score * 0.30  # weight 30%

        # Signal 2: Multi-hop keywords
        multi_hop = bool(_MULTI_HOP_PATTERNS.search(query))
        if multi_hop:
            score += 0.50  # weight 50% — strong escalation signal

        # Signal 3: Negation depth
        negation = bool(_NEGATION_PATTERNS.search(query))
        if negation:
            score += 0.15  # weight 15%

        # Signal 4: Ambiguity markers
        ambiguity = bool(_AMBIGUITY_PATTERNS.search(query))
        if ambiguity:
            score += 0.10  # weight 10%

        # Signal 5: Retrieval confidence (low = hard)
        LOW_RETRIEVAL = 0.30
        if retrieval_confidence is not None and retrieval_confidence < LOW_RETRIEVAL:
            score += 0.10  # weight 10%

        score = max(0.0, min(1.0, score))
        tier = "worker" if score < self._threshold else "reasoner"

        return DifficultySignals(
            raw_score=score,
            token_length=token_count,
            multi_hop_match=multi_hop,
            negation_match=negation,
            ambiguity_match=ambiguity,
            retrieval_confidence=retrieval_confidence,
            tier=tier,
        )


# ---------------------------------------------------------------------------
# TieredRouter (replaces ModelRouter stub, keeps backward-compatible interface)
# ---------------------------------------------------------------------------


class TieredRouter:
    """Routes queries to Tier-1 (worker SLM) or Tier-2 (reasoner LLM).

    This is the F1 upgrade of the D3 ModelRouter stub.  It uses a real
    DifficultyClassifier to make routing decisions, rather than a simple
    string lookup.

    Backward-compatible interface: also exposes a .route(task_complexity) method
    that matches the D3 ModelRouter signature for existing callers.

    Usage::

        router = TieredRouter()
        tier = router.route_query("What is the total invoice amount?")
        assert tier == "worker"

        tier = router.route_query("Compare warranty terms across both contracts.")
        assert tier == "reasoner"
    """

    def __init__(self, threshold: float = TIER_THRESHOLD) -> None:
        self._classifier = DifficultyClassifier(threshold=threshold)

    def route_query(
        self,
        query: str,
        retrieval_confidence: Optional[float] = None,
    ) -> str:
        """Route a natural-language query to the appropriate inference tier.

        Args:
            query: The user's query string.
            retrieval_confidence: Optional retrieval chunk score for the query.

        Returns:
            ``"worker"`` for Tier-1 (fast, cheap SLM);
            ``"reasoner"`` for Tier-2 (larger, higher-quality model).
        """
        signals = self._classifier.classify(query, retrieval_confidence)
        return signals.tier

    def classify(
        self,
        query: str,
        retrieval_confidence: Optional[float] = None,
    ) -> DifficultySignals:
        """Return full difficulty signals for a query (for audit/debug)."""
        return self._classifier.classify(query, retrieval_confidence)

    # ------------------------------------------------------------------
    # D3 backward-compatible interface
    # ------------------------------------------------------------------

    def route(self, task_complexity: str) -> str:
        """Backward-compatible D3 ModelRouter interface.

        Maps the legacy task_complexity string to the new tiered routing.
        Simple/retrieval tasks → "worker"; everything else → "reasoner".

        Args:
            task_complexity: A string label: "simple", "retrieval", "reasoning", etc.

        Returns:
            ``"worker"`` or ``"reasoner"``.
        """
        # D3 legacy mapping
        _WORKER_TASKS = frozenset({"simple", "retrieval"})
        if task_complexity in _WORKER_TASKS:
            return "worker"
        return "reasoner"


# ---------------------------------------------------------------------------
# ModelRouter alias (backward compat with D3 tests)
# ---------------------------------------------------------------------------

class ModelRouter:
    """D3 backward-compatible ModelRouter alias.

    The underlying routing is now performed by TieredRouter/DifficultyClassifier.
    All existing D3 tests continue to pass without modification.
    """

    def __init__(self) -> None:
        self._router = TieredRouter()

    def route(self, task_complexity: str) -> str:
        """Route a task_complexity label to 'worker' or 'reasoner'."""
        return self._router.route(task_complexity)
