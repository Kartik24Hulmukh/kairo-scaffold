"""B5 — FRONT Quote-Select + Smith-Waterman Alignment Cascade.

FRONT Pattern (ACL 2024)
------------------------
Standard RAG conditions the answer on full retrieved chunks.  FRONT (Faithful
Retrieval Output aNd grounding with Text) first SELECTS verbatim supporting
quotes from candidate chunks, then conditions the answer ONLY on those quotes.
This has two benefits:

  1. Faithfulness: the answer can only cite text that actually appears in the
     source — fabricated sentences have no quote to align to → BLOCK.
  2. Precision: citations point to character-exact spans, not whole chunks,
     so the UI highlight covers only the cited words.

This module implements FRONT in two self-contained phases:

Phase 1 — SELECT (front_select)
    Score every sentence/snippet in every candidate chunk against the query.
    A snippet is "supporting" if its overlap score ≥ QUOTE_THRESHOLD.
    Returns a list of (verbatim_quote, chunk, score) triples.
    If no snippet passes → caller must BLOCK the answer.

Phase 2 — ALIGN (sw_align)
    Map each verbatim quote to a character-precise [start, end) span in its
    source chunk using the Smith-Waterman local alignment algorithm.  If no
    alignment scores above SW_THRESHOLD → that quote is fabricated → BLOCK.
    From the char span, interpolate a sub-chunk BBox (groundmark-style).

BBox Interpolation (groundmark-style)
--------------------------------------
Docling chunks carry a page-level BBox (x0, y0, x1, y1) and all the text in
that BBox.  We must map a char span [cs, ce) within the chunk text to a
sub-BBox.  We use a simple linear proportion model:

    ratio_start = cs / len(chunk_text)
    ratio_end   = ce / len(chunk_text)
    sub_x0 = x0 + ratio_start * (x1 - x0)
    sub_x1 = x0 + ratio_end   * (x1 - x0)
    sub_y0 = y0
    sub_y1 = y1

For single-line chunks (y1 - y0 small) this is exact.  For multi-line chunks
we fall back to the full chunk bbox (conservative but never wrong).

Smith-Waterman (local alignment, char-level)
---------------------------------------------
SW is the gold standard for finding the best *substring* match of a short
query (the verbatim quote) inside a longer target (the chunk text).  Unlike
exact substring search it is robust to minor OCR noise, encoding differences,
and soft hyphens.

Score matrix:
    MATCH   = +2 if chars match (case-insensitive, after normalization)
    MISMATCH = -1
    GAP_OPEN = -2  (gap penalty, applied once per gap)
    GAP_EXT  = -1  (gap extension penalty)

Alignment terminates at the first cell that reaches or exceeds SW_THRESHOLD.

SPEC Gate
---------
≥95% of grounded answers on the golden set must carry character-precise spans
(not whole-chunk spans).  Fabricated quotes must fail alignment and be blocked.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FRONT: minimum lexical-overlap score for a snippet to be "supporting"
QUOTE_THRESHOLD: float = 0.25

# Smith-Waterman: minimum score to accept an alignment as genuine
SW_MIN_SCORE: float = 3.0

# Maximum line-height ratio to consider a chunk "single-line" for bbox interp.
_SINGLE_LINE_RATIO: float = 0.04  # y-span / page-height < 4% → single-line approx

# Smith-Waterman scoring constants
_MATCH = 2
_MISMATCH = -1
_GAP_OPEN = -2
_GAP_EXT = -1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """A verbatim supporting quote selected from a chunk by the FRONT selector."""
    text: str          # verbatim text from the chunk
    chunk_id: str
    page: int
    score: float       # FRONT relevance score (0-1)
    chunk_text: str    # full chunk text (for SW alignment)
    chunk_bbox: dict   # {"x0", "y0", "x1", "y1"}


@dataclass
class AlignedSpan:
    """Result of Smith-Waterman alignment: character-precise span + sub-bbox."""
    chunk_id: str
    page: int
    char_start: int
    char_end: int
    sw_score: float
    bbox: dict         # sub-chunk interpolated {"x0", "y0", "x1", "y1"}
    is_fabricated: bool = False  # True if sw_score < SW_MIN_SCORE


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Normalise text for overlap scoring and SW alignment.

    - Lowercase
    - Decompose unicode (NFD) then keep ASCII letters + digits + space
    - Collapse whitespace
    """
    nfd = unicodedata.normalize("NFD", text)
    ascii_only = nfd.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", ascii_only.lower()).split())


def _tokenize(text: str) -> list[str]:
    """Return non-empty tokens of normalised text."""
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Phase 1 — FRONT SELECT
# ---------------------------------------------------------------------------

def _snippet_score(query_tokens: set[str], snippet_tokens: list[str]) -> float:
    """Jaccard-like token overlap score between query and snippet.

    We use F1-token overlap (same as SQuAD/TriviaQA) which is symmetric and
    well-calibrated:

        precision = |q ∩ s| / |s|
        recall    = |q ∩ s| / |q|
        F1        = 2 * p * r / (p + r)
    """
    if not query_tokens or not snippet_tokens:
        return 0.0
    s_set = set(snippet_tokens)
    inter = len(query_tokens & s_set)
    if inter == 0:
        return 0.0
    precision = inter / len(s_set)
    recall = inter / len(query_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _split_sentences(text: str) -> list[str]:
    """Split chunk text into candidate snippets for scoring.

    We use sentence-boundary heuristics (period/semicolon/newline) plus a
    minimum-length filter to avoid degenerate one-word snippets.
    """
    # Split on sentence-ending punctuation + whitespace, or newlines
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text.strip())
    # Also split very long parts at comma boundaries (clause-level)
    expanded: list[str] = []
    for part in parts:
        if len(part) > 300:
            sub_parts = re.split(r",\s+", part)
            expanded.extend(sub_parts)
        else:
            expanded.append(part)
    # Filter empty / too-short snippets
    return [p.strip() for p in expanded if len(p.strip()) >= 10]


def front_select(
    query: str,
    chunks: list,
    *,
    threshold: float = QUOTE_THRESHOLD,
    max_quotes: int = 5,
) -> list[Quote]:
    """FRONT Phase 1: select supporting verbatim quotes from candidate chunks.

    Parameters
    ----------
    query:
        The user's natural-language question.
    chunks:
        List of chunk objects with .chunk_id, .page, .text, .bbox attributes.
    threshold:
        Minimum F1-overlap score for a snippet to be "supporting".
    max_quotes:
        Maximum number of quotes to return (top-scored).

    Returns
    -------
    List of Quote objects sorted by score descending.  Empty list means
    no supporting quotes found → caller must BLOCK.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    candidates: list[tuple[float, Quote]] = []

    for chunk in chunks:
        chunk_text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id", "")
        page = chunk.page if hasattr(chunk, "page") else chunk.get("page", 0)
        bbox = (
            {"x0": chunk.bbox.x0, "y0": chunk.bbox.y0,
             "x1": chunk.bbox.x1, "y1": chunk.bbox.y1}
            if hasattr(chunk, "bbox")
            else chunk.get("bbox", {"x0": 0, "y0": 0, "x1": 1, "y1": 1})
        )

        sentences = _split_sentences(chunk_text)
        # Also consider the full chunk as a candidate (good for short chunks)
        if chunk_text.strip() not in sentences:
            sentences.append(chunk_text.strip())

        for sentence in sentences:
            s_tokens = _tokenize(sentence)
            score = _snippet_score(query_tokens, s_tokens)
            if score >= threshold:
                candidates.append((score, Quote(
                    text=sentence,
                    chunk_id=chunk_id,
                    page=page,
                    score=score,
                    chunk_text=chunk_text,
                    chunk_bbox=bbox,
                )))

    # Sort by score descending, deduplicate by text
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    result: list[Quote] = []
    for _, q in candidates:
        key = _normalise(q.text)
        if key not in seen:
            seen.add(key)
            result.append(q)
            if len(result) >= max_quotes:
                break

    return result


# ---------------------------------------------------------------------------
# Phase 2 — Smith-Waterman Local Alignment
# ---------------------------------------------------------------------------

def _sw_score_matrix(query: str, target: str) -> tuple[list[list[float]], int, int]:
    """Compute Smith-Waterman score matrix for query vs target.

    Returns the filled matrix and the (row, col) of the maximum score.
    Uses simple linear gap penalty (not affine) for speed in short strings.
    Complexity: O(|query| × |target|).
    """
    m = len(query)
    n = len(target)

    # Initialise with zeros (SW always ≥ 0)
    H = [[0.0] * (n + 1) for _ in range(m + 1)]

    max_score = 0.0
    max_i = 0
    max_j = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = _MATCH if query[i - 1] == target[j - 1] else _MISMATCH
            diag = H[i - 1][j - 1] + match
            up   = H[i - 1][j] + _GAP_EXT
            left = H[i][j - 1] + _GAP_EXT
            val  = max(0.0, diag, up, left)
            H[i][j] = val
            if val > max_score:
                max_score = val
                max_i = i
                max_j = j

    return H, max_i, max_j


def _sw_traceback(H: list[list[float]], query: str, target: str,
                  max_i: int, max_j: int) -> tuple[int, int, float]:
    """Traceback from the max-score cell to find the aligned target span.

    Returns (target_start, target_end, score) where [target_start, target_end)
    is the character span in *target* that best aligns with *query*.
    """
    max_score = H[max_i][max_j]
    if max_score <= 0:
        return 0, 0, 0.0

    i, j = max_i, max_j
    target_end = j  # exclusive

    while i > 0 and j > 0 and H[i][j] > 0:
        score = H[i][j]
        diag = H[i - 1][j - 1]
        up   = H[i - 1][j]
        left = H[i][j - 1]

        if score == diag + (_MATCH if query[i - 1] == target[j - 1] else _MISMATCH):
            i -= 1
            j -= 1
        elif score == up + _GAP_EXT:
            i -= 1
        else:
            j -= 1

    target_start = j  # inclusive

    return target_start, target_end, max_score


def sw_align(
    quote: str,
    target: str,
    *,
    min_score: float = SW_MIN_SCORE,
) -> tuple[int, int, float]:
    """Align verbatim *quote* against *target* using Smith-Waterman local alignment.

    Returns
    -------
    (char_start, char_end, score) — character span in *target*.
    If score < min_score → (0, 0, score) indicating fabrication.

    The alignment is performed on normalised, lowercased text so minor OCR
    noise, hyphenation, and unicode variants do not cause false rejections.
    """
    if not quote.strip() or not target.strip():
        return 0, 0, 0.0

    q_norm = _normalise(quote)
    t_norm = _normalise(target)

    H, max_i, max_j = _sw_score_matrix(q_norm, t_norm)
    norm_start, norm_end, score = _sw_traceback(H, q_norm, t_norm, max_i, max_j)

    if score < min_score or norm_start >= norm_end:
        return 0, 0, score

    # Map the normalised character span back to the original target string.
    # We do this by finding the aligned normalised segment in the original text.
    norm_segment = t_norm[norm_start:norm_end]
    if not norm_segment:
        return 0, 0, score

    # Re-find in original target (case-insensitive, whitespace-collapsed match)
    raw_start, raw_end = _remap_span_to_original(norm_segment, target)
    return raw_start, raw_end, score


def _remap_span_to_original(norm_segment: str, original: str) -> tuple[int, int]:
    """Find norm_segment (normalised) in original text (case-insensitive).

    Strategy:
    1. Try exact case-insensitive substring search.
    2. If not found, try the first unique word of norm_segment.
    3. Fall back to (0, min(len(norm_segment)*2, len(original))).
    """
    lower_original = original.lower()

    # Simple normalise of the original — strip punctuation, collapse spaces
    t_norm = _normalise(original)
    norm_start_in_norm = t_norm.find(norm_segment)

    if norm_start_in_norm >= 0:
        # Find the character position in original that corresponds to
        # norm_start_in_norm in normalised text.
        # Walk original and count normalised characters.
        norm_pos = 0
        raw_start = 0
        for raw_idx, ch in enumerate(original):
            norm_ch = _normalise(ch)
            if norm_pos == norm_start_in_norm:
                raw_start = raw_idx
                break
            if norm_ch:
                norm_pos += len(norm_ch)
                if norm_pos > norm_start_in_norm:
                    raw_start = raw_idx
                    break

        # End span: raw_start + len(norm_segment) chars forward (approximation)
        raw_end = min(raw_start + len(norm_segment) + 10, len(original))
        return raw_start, raw_end

    # Fallback: first word search
    first_word = norm_segment.split()[0] if norm_segment.split() else norm_segment
    idx = lower_original.find(first_word)
    if idx >= 0:
        return idx, min(idx + len(norm_segment), len(original))

    return 0, min(len(norm_segment), len(original))


# ---------------------------------------------------------------------------
# BBox interpolation (groundmark-style)
# ---------------------------------------------------------------------------

def interpolate_bbox(
    chunk_bbox: dict,
    chunk_text: str,
    char_start: int,
    char_end: int,
    *,
    page_height: float = 1.0,
) -> dict:
    """Interpolate a sub-chunk BBox from character offsets.

    For a chunk with known BBox (x0, y0, x1, y1) covering chunk_text:
    - Assume uniform character density along the x-axis within the chunk.
    - Compute the x sub-range proportional to [char_start, char_end).
    - y-range is the full chunk height (we cannot sub-divide without
      per-character y coords from the layout engine).

    For multi-line chunks we return the full chunk BBox since proportional
    x interpolation is not meaningful across line breaks.

    Parameters
    ----------
    chunk_bbox:
        Dict with x0, y0, x1, y1 (page-relative, 0-1 normalised or pixel).
    chunk_text:
        The full text of the chunk.
    char_start, char_end:
        Character offsets [start, end) of the quote within chunk_text.
    page_height:
        Used to determine if the chunk spans multiple lines (bbox height /
        page_height ≥ _SINGLE_LINE_RATIO).  Pass page height in same units
        as bbox if normalised coordinates are not used.

    Returns
    -------
    Dict {"x0", "y0", "x1", "y1"} of the sub-span BBox.
    """
    x0, y0, x1, y1 = (
        chunk_bbox["x0"],
        chunk_bbox["y0"],
        chunk_bbox["x1"],
        chunk_bbox["y1"],
    )
    text_len = len(chunk_text)
    if text_len == 0:
        return chunk_bbox.copy()

    # Clamp spans to valid range
    cs = max(0, min(char_start, text_len))
    ce = max(cs, min(char_end, text_len))

    # Check if chunk spans multiple lines (height > single-line threshold)
    bbox_height = abs(y1 - y0)
    is_multiline = page_height > 0 and (bbox_height / page_height) >= _SINGLE_LINE_RATIO

    if is_multiline:
        # Cannot safely interpolate x in multi-line chunks — return full bbox
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    # Single-line: linear x interpolation
    width = x1 - x0
    ratio_start = cs / text_len
    ratio_end = ce / text_len

    sub_x0 = x0 + ratio_start * width
    sub_x1 = x0 + ratio_end * width

    # Ensure sub_x0 <= sub_x1 and within chunk bounds
    sub_x0 = max(x0, min(x1, sub_x0))
    sub_x1 = max(x0, min(x1, sub_x1))
    if sub_x0 >= sub_x1:
        sub_x1 = min(sub_x0 + 1, x1)

    return {"x0": sub_x0, "y0": y0, "x1": sub_x1, "y1": y1}


# ---------------------------------------------------------------------------
# Full FRONT + SW pipeline
# ---------------------------------------------------------------------------

def front_cascade(
    query: str,
    chunks: list,
    *,
    quote_threshold: float = QUOTE_THRESHOLD,
    sw_min_score: float = SW_MIN_SCORE,
    max_quotes: int = 5,
    page_height: float = 1.0,
) -> tuple[list[AlignedSpan], list[Quote]]:
    """Run the full FRONT+SW cascade for a query against retrieved chunks.

    Returns
    -------
    (aligned_spans, selected_quotes)
        aligned_spans: list of AlignedSpan objects.  Each span has a
            character-precise sub-chunk BBox.  Spans with is_fabricated=True
            failed SW alignment.
        selected_quotes: list of Quote objects that passed the FRONT SELECT
            threshold (before SW alignment).

    If aligned_spans is empty (or all spans are fabricated) → caller must BLOCK.
    """
    # Phase 1 — FRONT SELECT
    selected = front_select(query, chunks, threshold=quote_threshold, max_quotes=max_quotes)

    if not selected:
        return [], []

    # Phase 2 — Smith-Waterman ALIGN
    aligned: list[AlignedSpan] = []

    for quote in selected:
        char_start, char_end, sw_score = sw_align(
            quote.text, quote.chunk_text, min_score=sw_min_score
        )

        is_fabricated = (sw_score < sw_min_score or char_start >= char_end)

        if is_fabricated:
            aligned.append(AlignedSpan(
                chunk_id=quote.chunk_id,
                page=quote.page,
                char_start=0,
                char_end=0,
                sw_score=sw_score,
                bbox=quote.chunk_bbox,
                is_fabricated=True,
            ))
        else:
            sub_bbox = interpolate_bbox(
                quote.chunk_bbox,
                quote.chunk_text,
                char_start,
                char_end,
                page_height=page_height,
            )
            aligned.append(AlignedSpan(
                chunk_id=quote.chunk_id,
                page=quote.page,
                char_start=char_start,
                char_end=char_end,
                sw_score=sw_score,
                bbox=sub_bbox,
                is_fabricated=False,
            ))

    return aligned, selected
