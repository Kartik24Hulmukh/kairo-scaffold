"""
Kairo Document Classifier — Pure Heuristic Document Type Detection.

Classifies documents into invoice, contract, paper, or generic using
keyword matching. No LLM, no ML — pure deterministic heuristics.
"""
from __future__ import annotations

import re


def classify_document(text: str, page_count: int = 1, has_tables: bool = False) -> str:
    """Classify a document into one of: invoice, contract, paper, generic.

    Uses keyword matching and structural heuristics. No LLM needed.

    Args:
        text: The document text content.
        page_count: Number of pages in the document.
        has_tables: Whether the document contains detected tables.

    Returns:
        One of: "invoice", "contract", "paper", "generic"
    """
    text_lower = text.lower()

    # Invoice: strong signals
    invoice_keywords = ["invoice", "invoice number", "amount due", "total amount",
                        "bill to", "payment terms", "subtotal", "sub-total",
                        "tax amount", "faktura", "rechnung", "receipt"]
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)

    # Contract: strong signals
    contract_keywords = ["agreement", "party", "parties", "whereas",
                         "governing law", "termination", "non-disclosure",
                         "confidential", "obligations", "shall", "covenant",
                         "hereby", "hereinafter", "partnership"]
    contract_score = sum(1 for kw in contract_keywords if kw in text_lower)

    # Paper: strong signals
    paper_keywords = ["abstract", "references", "doi", "et al",
                      "methodology", "experimental results", "conclusions",
                      "keywords", "arxiv", "figure", "table"]
    paper_score = sum(1 for kw in paper_keywords if kw in text_lower)

    # Score-based classification with page count tiebreaker
    scores = {
        "invoice": invoice_score,
        "contract": contract_score,
        "paper": paper_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # If no strong signal, use page count heuristics
    if best_score == 0:
        if page_count >= 8:
            return "paper"
        elif page_count >= 5:
            return "contract"
        else:
            return "generic"

    # Contract needs at least 2 signals (many docs mention "agreement" casually)
    if best_type == "contract" and contract_score < 2:
        # Check if it's really a contract vs just mentioning the word
        if "whereas" in text_lower or "governing law" in text_lower or "parties" in text_lower:
            return "contract"
        # Otherwise, let other types compete
        if invoice_score > 0:
            return "invoice"
        if paper_score > 0:
            return "paper"
        return "generic"

    return best_type


def build_source_link(doc_id: str, page: int, bbox: list[float]) -> str:
    """Build a kairo:// source link for a grounded field.

    Format: kairo://doc/{doc_id}?page={page}&x={x}&y={y}&w={w}&h={h}
    """
    if not bbox or len(bbox) < 4:
        return f"kairo://doc/{doc_id}?page={page}"
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return f"kairo://doc/{doc_id}?page={page}&x={x}&y={y}&w={w}&h={h}"