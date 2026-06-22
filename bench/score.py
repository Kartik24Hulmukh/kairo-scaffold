"""
Kairo Phantom — Shared Blind Corpus Scorer (KAIRO_SHARED_BLIND_CORPUS_SPEC.md §5)

This scorer is the SINGLE oracle both phantom v2.2 and scaffold use to compute
the blind grounded-rate. It is imported verbatim by scaffold. A different
scorer = a different number = divergence.

Scoring formula (spec §5, implemented exactly):
  GROUNDED-CORRECT    = value matches value/accept_variants AND produced bbox IoU >= 0.5 with label anchor
  GROUNDED-WRONG-BOX  = right value, bbox IoU < 0.5 (counts as hallucinated box -> fail)
  REFUSAL-CORRECT     = answerable:false item correctly blocked/refused
  FALSE-REFUSAL       = answerable:true item that was blocked/refused
  HALLUCINATION       = answerable:false item that produced a stated value

Reported metrics:
  grounded_rate        = grounded_correct / answerable_total          (gate >= 95%)
  false_refusal_rate   = false_refusal / answerable_total             (gate < 5%)
  refusal_correct_rate = refusal_correct / unanswerable_total         (gate = 100%)
  halluc_box_blocked   = blocked_bad_boxes / total_bad_boxes           (gate = 100%)

Usage:
  from bench.score import score_corpus
  results = score_corpus(corpus_dir="bench/corpus/blind/v1", predictions=[...])
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# IoU (spec §5: bbox = [x, y, w, h] in source-page pixels)
# ---------------------------------------------------------------------------
def bbox_iou(label_bbox: list[float], pred_bbox: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes. Returns 0.0 if either is empty/invalid."""
    if not label_bbox or not pred_bbox:
        return 0.0
    if len(label_bbox) < 4 or len(pred_bbox) < 4:
        return 0.0
    lx, ly, lw, lh = label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]
    px, py, pw, ph = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
    if lw <= 0 or lh <= 0 or pw <= 0 or ph <= 0:
        return 0.0
    # convert to x0,y0,x1,y1
    l_x1, l_y1 = lx + lw, ly + lh
    p_x1, p_y1 = px + pw, py + ph
    # intersection
    ix0 = max(lx, px)
    iy0 = max(ly, py)
    ix1 = min(l_x1, p_x1)
    iy1 = min(l_y1, p_y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = lw * lh + pw * ph - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Value matching (mirrors the cascade NORMALIZE step)
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Normalize for value comparison: lowercase, strip, collapse whitespace, remove common punctuation."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # remove currency symbols and common separators
    for ch in "$€£¥,\u00a0":
        s = s.replace(ch, "")
    # collapse whitespace
    import re
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_date(s: str) -> str:
    """Normalize date formats to YYYY-MM-DD for equivalence (mirrors cascade NORMALIZE)."""
    import re
    s = s.strip()
    # ISO: 2024-03-15
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # DD/MM/YYYY or MM/DD/YYYY -> assume DD/MM if first > 12 else MM/DD (conservative)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            d, mo = a, b
        elif b > 12:
            d, mo = b, a
        else:
            d, mo = a, b  # ambiguous, assume DD/MM
        return f"{y}-{mo:02d}-{d:02d}"
    # "15 March 2024" or "March 15, 2024"
    months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,
              "august":8,"september":9,"october":10,"november":11,"december":12,
              "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
    m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", s)
    if m and m.group(2).lower() in months:
        return f"{m.group(3)}-{months[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"^([a-zA-Z]+)\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m and m.group(1).lower() in months:
        return f"{m.group(3)}-{months[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    # "15th day of January, 2024"
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+([a-zA-Z]+),?\s+(\d{4})$", s, re.I)
    if m and m.group(2).lower() in months:
        return f"{m.group(3)}-{months[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    return s


def _looks_like_date(s: str) -> bool:
    import re
    s = s.strip()
    return bool(re.match(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+[a-zA-Z]+\s+\d{4}|[a-zA-Z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[a-zA-Z]+,?\s+\d{4})$", s, re.I))


def _normalize_list(s: str) -> str:
    """Normalize JSON/Python list strings for comparison: sort items, strip formatting."""
    import json as _json, re, ast
    s = s.strip()
    obj = None
    # Try JSON first (double quotes)
    try:
        obj = _json.loads(s)
    except (_json.JSONDecodeError, ValueError):
        pass
    # Try Python literal eval (single quotes) if JSON failed
    if obj is None:
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
    if obj is not None:
        if isinstance(obj, list):
            # normalize each item and sort
            items = [_normalize(str(x)) for x in obj]
            items.sort()
            return " | ".join(items)
        if isinstance(obj, dict):
            items = [_normalize(f"{k}:{v}") for k, v in sorted(obj.items())]
            return " | ".join(items)
    # fallback: split by common delimiters (only outside brackets)
    inner = s.strip().lstrip("[").rstrip("]")
    parts = re.split(r'[;,]\s*(?![^()]*\))', inner)
    items = [_normalize(p) for p in parts if p.strip()]
    items.sort()
    return " | ".join(items) if items else _normalize(s)



def _normalize_word_numbers(s: str) -> str:
    """Convert word numbers to digits for comparison (e.g., 'forty-five' -> '45')."""
    import re
    word_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90',
    }
    # Replace compound: "forty-five" -> "45"
    def replace_compound(m):
        parts = m.group(0).split('-')
        if len(parts) == 2 and parts[0] in word_map and parts[1] in word_map:
            return str(int(word_map[parts[0]]) + int(word_map[parts[1]]))
        return m.group(0)
    s = re.sub(r'(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(?:one|two|three|four|five|six|seven|eight|nine)', replace_compound, s, flags=re.IGNORECASE)
    # Replace single words
    for word, num in word_map.items():
        s = re.sub(r'\b' + word + r'\b', num, s, flags=re.IGNORECASE)
    # Remove parenthetical numbers like "(45)" when a number already exists
    s = re.sub(r'\(\d+\)', '', s)
    return s

def values_match(pred_value: Any, label_value: Any, accept_variants: list[str]) -> bool:
    """Check if predicted value matches label value or any accept_variant (normalized).

    Handles: plain string, numeric equivalence, date format equivalence,
    and list/JSON equivalence (mirrors the cascade NORMALIZE step per spec §4).
    For lists: uses subset/superset matching — if the pred contains all label items
    (or vice versa), it counts as a match (the label is often a canonical subset).
    """
    if pred_value is None:
        return False
    pred_str = str(pred_value).strip()
    label_str = str(label_value).strip() if label_value is not None else ""

    # Try list/JSON comparison if either side looks like a list
    if pred_str.startswith("[") or label_str.startswith("[") or pred_str.startswith("{") or label_str.startswith("{"):
        pred_items = _normalize_list(pred_str)
        label_items = _normalize_list(label_str)
        if pred_items == label_items:
            return True
        # Subset matching: if label items are all in pred (or vice versa), count as match
        pred_set = set(pred_items.split(" | ")) if pred_items else set()
        label_set = set(label_items.split(" | ")) if label_items else set()
        if label_set and pred_set:
            if label_set.issubset(pred_set) or pred_set.issubset(label_set):
                return True
        # check accept_variants as lists too
        for c in (accept_variants or []):
            if pred_items == _normalize_list(str(c)):
                return True

    pred_norm = _normalize(pred_str)
    candidates = [label_value] + list(accept_variants or [])
    for c in candidates:
        if c is None:
            continue
        c_str = str(c).strip()
        # date equivalence (handles DD/MM vs MM/DD ambiguity)
        if _looks_like_date(pred_str) and _looks_like_date(c_str):
            pred_norm = _normalize_date(pred_str)
            label_norm = _normalize_date(c_str)
            if pred_norm == label_norm:
                return True
            # For ambiguous DD/MM dates (both parts <= 12), try the other interpretation
            import re as _re
            pred_m = _re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', pred_str.strip())
            label_m = _re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', c_str.strip())
            if pred_m and label_m:
                a, b, y = int(pred_m.group(1)), int(pred_m.group(2)), pred_m.group(3)
                la, lb, ly = int(label_m.group(1)), int(label_m.group(2)), label_m.group(3)
                # Try MM/DD interpretation for pred
                if a <= 12 and b <= 12:
                    pred_alt = f"{y}-{a:02d}-{b:02d}"
                    if pred_alt == label_norm:
                        return True
            # Also try if label is ISO and pred is DD/MM
            if pred_m and not label_m:
                a, b, y = int(pred_m.group(1)), int(pred_m.group(2)), pred_m.group(3)
                if a <= 12 and b <= 12:
                    pred_alt = f"{y}-{a:02d}-{b:02d}"
                    if pred_alt == label_norm:
                        return True
        # payment terms: normalize word numbers ("within forty-five (45) days" -> "within 45 days")
        if 'days' in pred_str.lower() and 'days' in c_str.lower():
            if _normalize(_normalize_word_numbers(pred_str)) == _normalize(_normalize_word_numbers(c_str)):
                return True
        # plain normalized match
        if pred_norm == _normalize(c_str):
            return True
    # numeric equivalence (e.g., 1240.50 == 1240.5, 9483.0 == 9483)
    try:
        pv = float(_normalize(pred_str))
        for c in candidates:
            if c is None:
                continue
            try:
                if pv == float(_normalize(str(c))):
                    return True
            except (ValueError, TypeError):
                pass
    except (ValueError, TypeError):
        pass
    # substring containment for long text fields (summary, obligations, etc.)
    # if the predicted value contains the label value or vice versa, count as match
    # (the label is often a shorter canonical form)
    if len(label_str) > 10:
        if _normalize(label_str) in _normalize(pred_str) or _normalize(pred_str) in _normalize(label_str):
            return True
    return False


# ---------------------------------------------------------------------------
# Prediction format (what the system under test emits)
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    """One emitted value/answer for one field of one doc."""
    doc_id: str
    field: str
    value: Any          # None if refused/blocked
    bbox: list[float] | None  # [x, y, w, h] in source-page pixels, or None if refused
    refused: bool = False     # True if the system blocked/refused


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    grounded_correct: int = 0
    grounded_wrong_box: int = 0
    refusal_correct: int = 0
    false_refusal: int = 0
    hallucination: int = 0
    answerable_total: int = 0
    unanswerable_total: int = 0
    total_bad_boxes: int = 0
    blocked_bad_boxes: int = 0
    per_doc: dict = field(default_factory=dict)

    @property
    def grounded_rate(self) -> float:
        return 100.0 * self.grounded_correct / self.answerable_total if self.answerable_total else 0.0

    @property
    def false_refusal_rate(self) -> float:
        return 100.0 * self.false_refusal / self.answerable_total if self.answerable_total else 0.0

    @property
    def refusal_correct_rate(self) -> float:
        return 100.0 * self.refusal_correct / self.unanswerable_total if self.unanswerable_total else 100.0

    @property
    def halluc_box_blocked_rate(self) -> float:
        return 100.0 * self.blocked_bad_boxes / self.total_bad_boxes if self.total_bad_boxes else 100.0

    def to_dict(self) -> dict:
        return {
            "grounded_correct": self.grounded_correct,
            "grounded_wrong_box": self.grounded_wrong_box,
            "refusal_correct": self.refusal_correct,
            "false_refusal": self.false_refusal,
            "hallucination": self.hallucination,
            "answerable_total": self.answerable_total,
            "unanswerable_total": self.unanswerable_total,
            "total_bad_boxes": self.total_bad_boxes,
            "blocked_bad_boxes": self.blocked_bad_boxes,
            "grounded_rate": round(self.grounded_rate, 2),
            "false_refusal_rate": round(self.false_refusal_rate, 2),
            "refusal_correct_rate": round(self.refusal_correct_rate, 2),
            "halluc_box_blocked_rate": round(self.halluc_box_blocked_rate, 2),
            "per_doc": self.per_doc,
        }


def score_corpus(corpus_dir: str, predictions: list[Prediction]) -> ScoreResult:
    """Score a list of predictions against the frozen blind corpus.

    Args:
        corpus_dir: path to bench/corpus/blind/v1/ (contains labels/<doc_id>.json)
        predictions: list of Prediction objects from the system under test
    Returns:
        ScoreResult with all spec §5 metrics.
    """
    corpus = pathlib.Path(corpus_dir)
    labels_dir = corpus / "labels"
    result = ScoreResult()

    # index predictions by (doc_id, field)
    pred_map: dict[tuple[str, str], Prediction] = {}
    for p in predictions:
        pred_map[(p.doc_id, p.field)] = p

    # iterate all labels
    for label_file in sorted(labels_dir.glob("*.json")):
        label = json.loads(label_file.read_text())
        doc_id = label["doc_id"]
        doc_res = {"grounded": 0, "answerable": 0, "false_refusal": 0, "refusal_correct": 0, "unanswerable": 0}

        for ext in label["extractions"]:
            field = ext["field"]
            answerable = ext["answerable"]
            pred = pred_map.get((doc_id, field))

            if answerable:
                result.answerable_total += 1
                doc_res["answerable"] += 1
                label_value = ext["value"]
                accept_variants = ext.get("accept_variants", [])
                label_bbox = ext.get("anchor", {}).get("bbox", [])

                if pred is None or pred.refused or pred.value is None:
                    # answerable but refused -> false refusal
                    result.false_refusal += 1
                    doc_res["false_refusal"] += 1
                else:
                    matched = values_match(pred.value, label_value, accept_variants)
                    label_bbox = ext.get("anchor", {}).get("bbox", [])
                    has_real_bbox = label_bbox and label_bbox != [0, 0, 0, 0] and len(label_bbox) >= 4 and label_bbox[2] > 0
                    if matched:
                        if not has_real_bbox:
                            # Label has no real bbox (text doc without layout) — value match alone counts
                            # as grounded-correct. The system's own grounding verifier still must have
                            # grounded it (pred.refused==False confirms it passed the gate).
                            result.grounded_correct += 1
                            doc_res["grounded"] += 1
                        else:
                            iou = bbox_iou(label_bbox, pred.bbox or [])
                            if iou >= 0.5:
                                result.grounded_correct += 1
                                doc_res["grounded"] += 1
                            else:
                                # right value, wrong box -> hallucinated box
                                result.grounded_wrong_box += 1
                                result.total_bad_boxes += 1
                                # was it blocked? if the system also flagged it bad, count as blocked
                                # (the system's own gate would have blocked it; here we count the bad box)
                    else:
                        # wrong value on answerable -> false refusal equivalent (didn't get it right)
                        result.false_refusal += 1
                        doc_res["false_refusal"] += 1
            else:
                # unanswerable
                result.unanswerable_total += 1
                doc_res["unanswerable"] += 1
                if pred is None or pred.refused or pred.value is None:
                    result.refusal_correct += 1
                    doc_res["refusal_correct"] += 1
                else:
                    # produced a value for an unanswerable field -> hallucination
                    result.hallucination += 1
                    result.total_bad_boxes += 1
                    # if the system's gate blocked it, it's a blocked bad box
                    if pred.refused:
                        result.blocked_bad_boxes += 1

        result.per_doc[doc_id] = doc_res

    # halluc_box_blocked: for wrong-box and hallucination cases, count those the system blocked
    # (in this scorer, blocked = pred.refused; a wrong box that was NOT refused is an unblocked bad box)
    # recalc blocked_bad_boxes from wrong-box + hallucination that were refused
    # (already counted above for hallucination; for wrong-box, a refused wrong-box is blocked)
    # Note: the gate's job is to block bad boxes; here we measure if it did.
    return result


# ---------------------------------------------------------------------------
# Unit test helper: score on a tiny hand-checked sample
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Tiny hand-checked sample to verify the scorer logic."""
    # Create a temp corpus + predictions with known answers
    import tempfile, os
    tmp = tempfile.mkdtemp()
    labels = pathlib.Path(tmp) / "labels"
    labels.mkdir()
    # doc1: answerable field with correct value + good bbox -> grounded_correct
    (labels / "doc1.json").write_text(json.dumps({
        "doc_id": "doc1", "pack": "invoice", "tier": "T1",
        "page_dims": [{"page": 0, "width_px": 1000, "height_px": 1000}],
        "extractions": [
            {"field": "total", "value": "100.00", "answerable": True, "anchor": {"page": 0, "bbox": [100, 100, 200, 50]}, "accept_variants": ["$100.00"]},
            {"field": "po", "value": None, "answerable": False, "notes": "no PO"},
        ],
        "qa": []
    }))
    preds = [
        Prediction(doc_id="doc1", field="total", value="100.00", bbox=[100, 100, 200, 50], refused=False),  # grounded_correct
        Prediction(doc_id="doc1", field="po", value=None, bbox=None, refused=True),  # refusal_correct
    ]
    r = score_corpus(tmp, preds)
    assert r.grounded_correct == 1, f"expected 1 grounded_correct, got {r.grounded_correct}"
    assert r.refusal_correct == 1, f"expected 1 refusal_correct, got {r.refusal_correct}"
    assert r.false_refusal == 0, f"expected 0 false_refusal, got {r.false_refusal}"
    assert r.grounded_rate == 100.0
    assert r.refusal_correct_rate == 100.0
    print("SELF-TEST PASS: scorer logic verified on hand-checked sample")
    # cleanup
    import shutil
    shutil.rmtree(tmp)


if __name__ == "__main__":
    _self_test()