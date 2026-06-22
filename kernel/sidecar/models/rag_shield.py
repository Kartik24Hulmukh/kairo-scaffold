"""RAGShield: KB-poisoning defense for the Tier-2 federated flywheel.
PENDING-REAL-APP: Full RAGShield integration requires the Tier-2 opt-in flywheel.
This stub provides the interface and detects obvious poisoning patterns.
"""
import re

POISONING_PATTERNS = [
    r'ignore previous instructions',
    r'ignore all previous',
    r'disregard your',
    r'you are now',
    r'system:\s*you',
    r'<\|system\|>',
    r'\[INST\].*ignore',
    r'from now on act as',   # DAN jailbreak: "From now on act as DAN..."
]


def scan_content_for_poisoning(text: str) -> tuple[bool, list[str]]:
    """Scan cached/learned content for poisoning patterns before it influences answers.
    Returns (is_clean, list_of_detected_patterns).
    """
    detected = []
    text_lower = text.lower()
    for pattern in POISONING_PATTERNS:
        if re.search(pattern, text_lower):
            detected.append(pattern)
    return (len(detected) == 0, detected)


def quarantine_if_poisoned(content: str, doc_id: str) -> dict:
    """Returns quarantine decision."""
    is_clean, patterns = scan_content_for_poisoning(content)
    return {
        "doc_id": doc_id,
        "is_clean": is_clean,
        "detected_patterns": patterns,
        "action": "allow" if is_clean else "quarantine",
    }


MAX_QUERY_LENGTH = 2048


def sanitize_user_query(query: str) -> str:
    """Sanitize a user-supplied query before it enters the RAG pipeline.

    Steps applied in order:
    1. Strip leading and trailing whitespace.
    2. Remove null bytes (\\x00) — prevent terminator-injection attacks.
    3. Truncate to MAX_QUERY_LENGTH (2048) characters.

    Returns the sanitized string.
    """
    query = query.strip()
    query = query.replace("\x00", "")
    query = query[:MAX_QUERY_LENGTH]
    return query


# ---------------------------------------------------------------------------
# A7 — KB poisoning defense: validate_before_cache
# ---------------------------------------------------------------------------

import math as _math


def _entropy(text: str) -> float:
    """Compute Shannon entropy of the text (bits per character).

    Low-entropy text (e.g., 'token token token token...') may indicate
    adversarial padding or repetitive injection content.
    Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0
    freq: dict = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        entropy -= p * _math.log2(p)
    return entropy


# Minimum entropy threshold: text below this is considered suspiciously repetitive.
# Normal English prose has entropy ~4.0 bits/char; adversarial padding is <1.0.
MIN_ENTROPY_THRESHOLD = 1.5

# Additional semantic injection patterns beyond POISONING_PATTERNS
EXTENDED_PATTERNS = POISONING_PATTERNS + [
    r'act as.*unrestricted',
    r'dan.*do anything now',
    r'developer mode.*ignore',
    r'override.*safety',
    r'pretend.*no restrictions',
    r'forget.*instructions',
]


def validate_before_cache(content: str, doc_id: str) -> dict:
    """Validate content before it is stored in the knowledge cache.

    Applies two layers of defense:
    1. Pattern scan (scan_content_for_poisoning): checks for known injection strings.
    2. Entropy check: flags suspiciously low-entropy content (adversarial padding).

    Returns a dict:
      {
        "doc_id": str,
        "is_clean": bool,
        "detected_patterns": list[str],
        "entropy": float,
        "low_entropy": bool,
        "action": "allow" | "quarantine",
      }
    """
    # Pattern scan
    detected = []
    text_lower = content.lower()
    for pattern in EXTENDED_PATTERNS:
        if re.search(pattern, text_lower):
            detected.append(pattern)

    # Entropy check
    entropy = _entropy(content)
    low_entropy = len(content) > 100 and entropy < MIN_ENTROPY_THRESHOLD

    is_clean = (len(detected) == 0) and (not low_entropy)

    return {
        "doc_id": doc_id,
        "is_clean": is_clean,
        "detected_patterns": detected,
        "entropy": round(entropy, 4),
        "low_entropy": low_entropy,
        "action": "allow" if is_clean else "quarantine",
    }
