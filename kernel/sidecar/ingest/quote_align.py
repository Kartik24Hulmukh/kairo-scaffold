"""B5 — FRONT quote-first + groundmark quote→bbox alignment.

PLAN:
1. Implement smith_waterman_align(quote, text) local sequence alignment for character sequences.
2. Implement align_quote_to_chunks(quote, chunks, threshold=0.85) to locate the best aligned chunk.
3. If a match is found above threshold, compute the character-precise sub-bbox by interpolating the original chunk bbox coordinates.
4. Otherwise return None (Unaligned quote -> BLOCK).
"""

import hashlib
from typing import Any, Dict, List, Tuple, Optional

def smith_waterman_align(quote: str, text: str) -> Tuple[float, int, int]:
    """Aligns a quote to a text using Smith-Waterman local alignment.
    
    Returns (score_ratio, start_idx, end_idx).
    """
    q = quote.strip().lower()
    t = text.strip().lower()
    
    M, N = len(q), len(t)
    if M == 0 or N == 0:
        return 0.0, 0, 0
        
    # Scoring scheme
    MATCH = 2
    MISMATCH = -1
    GAP = -2
    
    # DP table
    dp = [[0] * (N + 1) for _ in range(M + 1)]
    
    max_score = 0
    max_pos = (0, 0)
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            match_score = MATCH if q[i-1] == t[j-1] else MISMATCH
            score = max(
                0,
                dp[i-1][j-1] + match_score,
                dp[i-1][j] + GAP,
                dp[i][j-1] + GAP
            )
            dp[i][j] = score
            if score > max_score:
                max_score = score
                max_pos = (i, j)
                
    if max_score == 0:
        return 0.0, 0, 0
        
    # Traceback to find the starting position in text
    i, j = max_pos
    end_idx = j
    
    while i > 0 and j > 0 and dp[i][j] > 0:
        score = dp[i][j]
        match_score = MATCH if q[i-1] == t[j-1] else MISMATCH
        if score == dp[i-1][j-1] + match_score:
            i -= 1
            j -= 1
        elif score == dp[i-1][j] + GAP:
            i -= 1
        else:
            j -= 1
            
    start_idx = j
    
    # Calculate score ratio: max possible score is len(q) * MATCH
    max_possible = len(q) * MATCH
    ratio = max_score / max_possible if max_possible > 0 else 0.0
    return ratio, start_idx, end_idx

def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Robust helper to extract field from dict, object, or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name, default)
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict().get(name, default)
        except Exception:
            pass
    # Try model_dump for Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump().get(name, default)
        except Exception:
            pass
    return default

def align_quote_to_chunks(quote: str, chunks: List[Any], threshold: float = 0.85) -> Optional[Dict[str, Any]]:
    """Aligns a verbatim quote against all document chunks.
    
    Returns the aligned anchor dict or None if no chunk aligns above the threshold.
    """
    if not quote or not quote.strip():
        return None
        
    best_ratio = 0.0
    best_match = None
    
    for chunk in chunks:
        chunk_text = get_field(chunk, "text", "")
        if not chunk_text:
            continue
            
        ratio, start_idx, end_idx = smith_waterman_align(quote, chunk_text)
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best_match = (chunk, start_idx, end_idx)
            
    if not best_match:
        return None
        
    chunk, start_idx, end_idx = best_match
    chunk_text = get_field(chunk, "text", "")
    chunk_id = get_field(chunk, "id") or get_field(chunk, "chunk_id") or ""
    page = get_field(chunk, "page") or get_field(chunk, "page_index") or 1
    
    # Get bbox
    bbox_obj = get_field(chunk, "bbox")
    x0 = get_field(bbox_obj, "x0", 0.0)
    y0 = get_field(bbox_obj, "y0", 0.0)
    x1 = get_field(bbox_obj, "x1", 1.0)
    y1 = get_field(bbox_obj, "y1", 1.0)
    
    # Horizontal interpolation for sub-bbox
    text_len = len(chunk_text)
    if text_len > 0:
        sub_x0 = x0 + (start_idx / text_len) * (x1 - x0)
        sub_x1 = x0 + (end_idx / text_len) * (x1 - x0)
    else:
        sub_x0, sub_x1 = x0, x1
        
    return {
        "chunk_id": chunk_id,
        "char_span": (start_idx, end_idx),
        "page": page,
        "bbox": {
            "x0": sub_x0,
            "y0": y0,
            "x1": sub_x1,
            "y1": y1
        }
    }
