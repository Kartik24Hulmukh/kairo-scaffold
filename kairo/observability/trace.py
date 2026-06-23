"""
Kairo Grounding Trace — Cascade-Specific Tracing + Live Dashboard.

Wraps an LLM tracing library with Kairo-specific cascade spans:
  - Each cascade layer (NORMALIZE, EXACT, FUZZY, SEMANTIC, VISUAL, BLOCK)
    gets its own traced span with confidence, bbox, and decision attributes.
  - A custom HTML dashboard visualizes the cascade waterfall.
  - Auto-instruments FastAPI for request-level tracing.

The tracing library is used for span management. Kairo adds:
  - Cascade-specific span attributes (confidence, bbox, decision, method)
  - A custom dashboard HTML page served at GET /dashboard
  - A trace store queryable via GET /api/traces
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any, Callable

from kernel.core.data_model import GroundingMethod

logger = logging.getLogger(__name__)


@dataclass
class CascadeSpan:
    """A single cascade layer execution span."""
    layer: str  # NORMALIZE, EXACT, FUZZY, SEMANTIC, VISUAL, BLOCK
    decision: str  # "grounded", "refused", "skipped"
    confidence: float
    bbox: list[float] | None = None
    method: str = ""
    wall_time_ms: float = 0.0
    timestamp: str = ""
    doc_id: str = ""
    field: str = ""
    metadata: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "decision": self.decision,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "method": self.method,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "timestamp": self.timestamp,
            "doc_id": self.doc_id,
            "field": self.field,
            "metadata": self.metadata,
        }


@dataclass
class GroundingTrace:
    """A complete grounding trace for one extraction."""
    doc_id: str
    field: str
    value: str
    final_decision: str  # "grounded" or "refused"
    final_method: str  # GroundingMethod value
    final_confidence: float
    final_bbox: list[float] | None
    spans: list[CascadeSpan] = dc_field(default_factory=list)
    total_wall_time_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "field": self.field,
            "value": self.value[:200],  # truncate for display
            "final_decision": self.final_decision,
            "final_method": self.final_method,
            "final_confidence": self.final_confidence,
            "final_bbox": self.final_bbox,
            "spans": [s.to_dict() for s in self.spans],
            "total_wall_time_ms": round(self.total_wall_time_ms, 2),
            "timestamp": self.timestamp,
        }


# In-memory trace store (queryable by the dashboard)
_trace_store: list[GroundingTrace] = []
_MAX_TRACES = 1000


def record_trace(trace: GroundingTrace) -> None:
    """Record a grounding trace in the trace store."""
    _trace_store.append(trace)
    if len(_trace_store) > _MAX_TRACES:
        _trace_store.pop(0)  # keep most recent


def get_traces(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent grounding traces for the dashboard."""
    return [t.to_dict() for t in _trace_store[-limit:]]


def get_trace_stats() -> dict[str, Any]:
    """Get aggregate trace statistics for the dashboard stats panel."""
    if not _trace_store:
        return {
            "total_traces": 0,
            "grounded_pct": 0.0,
            "refused_pct": 0.0,
            "blocked_pct": 0.0,
            "avg_cascade_depth": 0.0,
            "avg_latency_ms": 0.0,
        }
    total = len(_trace_store)
    grounded = sum(1 for t in _trace_store if t.final_decision == "grounded")
    refused = sum(1 for t in _trace_store if t.final_decision == "refused")
    blocked = sum(1 for t in _trace_store if t.final_method == GroundingMethod.BLOCK.value)
    avg_depth = sum(len(t.spans) for t in _trace_store) / total
    avg_latency = sum(t.total_wall_time_ms for t in _trace_store) / total
    return {
        "total_traces": total,
        "grounded_pct": round(grounded / total * 100, 1),
        "refused_pct": round(refused / total * 100, 1),
        "blocked_pct": round(blocked / total * 100, 1),
        "avg_cascade_depth": round(avg_depth, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }


def traced_cascade(
    verifier,
    value: str,
    source_span: str,
    chunks: list,
    doc_id: str = "",
    field_name: str = "",
) -> tuple[GroundingMethod, tuple]:
    """Run the grounding verifier with cascade-specific tracing.

    Wraps each cascade layer in a traced span, recording confidence,
    bbox, and decision. Returns the verifier result + records the trace.
    """
    from kernel.core.grounding import (
        normalize_text, best_fuzzy_match, bbox_iou,
        GroundingVerifierImpl,
    )
    from kernel.core.embeddings import get_embedding, cosine_similarity
    from kernel.core.data_model import Anchor, BBox

    spans: list[CascadeSpan] = []
    t_start = time.monotonic()
    now = datetime.now().isoformat()

    target = source_span if source_span else value
    if not target.strip() or not chunks:
        spans.append(CascadeSpan(
            layer="BLOCK", decision="refused", confidence=0.0,
            method=GroundingMethod.BLOCK.value, timestamp=now,
            doc_id=doc_id, field=field_name,
        ))
        trace = GroundingTrace(
            doc_id=doc_id, field=field_name, value=value,
            final_decision="refused", final_method=GroundingMethod.BLOCK.value,
            final_confidence=0.0, final_bbox=None,
            spans=spans, total_wall_time_ms=0.0, timestamp=now,
        )
        record_trace(trace)
        return GroundingMethod.BLOCK, ()

    # Layer 1: NORMALIZE
    t0 = time.monotonic()
    norm_target = normalize_text(target)
    spans.append(CascadeSpan(
        layer="NORMALIZE", decision="pass", confidence=1.0,
        wall_time_ms=(time.monotonic() - t0) * 1000,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))

    # Layer 2: EXACT
    t0 = time.monotonic()
    exact_found = False
    exact_anchor = None
    for chunk in chunks:
        if target.lower() in chunk.text.lower():
            start = chunk.text.lower().find(target.lower())
            end = start + len(target)
            exact_anchor = Anchor(
                chunk_id=chunk.chunk_id, char_span=(start, end),
                page=chunk.page, bbox=chunk.bbox,
            )
            exact_found = True
            break
    spans.append(CascadeSpan(
        layer="EXACT", decision="grounded" if exact_found else "skip",
        confidence=1.0 if exact_found else 0.0,
        bbox=[exact_anchor.bbox.x0, exact_anchor.bbox.y0,
              exact_anchor.bbox.x1 - exact_anchor.bbox.x0,
              exact_anchor.bbox.y1 - exact_anchor.bbox.y0] if exact_anchor and exact_anchor.bbox else None,
        method=GroundingMethod.EXACT.value if exact_found else "",
        wall_time_ms=(time.monotonic() - t0) * 1000,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))
    if exact_found:
        total_ms = (time.monotonic() - t_start) * 1000
        trace = GroundingTrace(
            doc_id=doc_id, field=field_name, value=value,
            final_decision="grounded", final_method=GroundingMethod.EXACT.value,
            final_confidence=1.0,
            final_bbox=spans[-1].bbox,
            spans=spans, total_wall_time_ms=total_ms, timestamp=now,
        )
        record_trace(trace)
        return GroundingMethod.EXACT, (exact_anchor,)

    # Layer 3: FUZZY
    t0 = time.monotonic()
    best_ratio = 0.0
    best_fuzzy_anchor = None
    for chunk in chunks:
        ratio, span = best_fuzzy_match(target, chunk.text)
        if ratio >= verifier.fuzzy_threshold and ratio > best_ratio:
            best_ratio = ratio
            best_fuzzy_anchor = Anchor(
                chunk_id=chunk.chunk_id, char_span=span,
                page=chunk.page, bbox=chunk.bbox,
            )
    fuzzy_found = best_fuzzy_anchor is not None
    spans.append(CascadeSpan(
        layer="FUZZY", decision="grounded" if fuzzy_found else "skip",
        confidence=best_ratio if fuzzy_found else 0.0,
        bbox=[best_fuzzy_anchor.bbox.x0, best_fuzzy_anchor.bbox.y0,
              best_fuzzy_anchor.bbox.x1 - best_fuzzy_anchor.bbox.x0,
              best_fuzzy_anchor.bbox.y1 - best_fuzzy_anchor.bbox.y0] if best_fuzzy_anchor and best_fuzzy_anchor.bbox else None,
        method=GroundingMethod.FUZZY.value if fuzzy_found else "",
        wall_time_ms=(time.monotonic() - t0) * 1000,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))
    if fuzzy_found:
        total_ms = (time.monotonic() - t_start) * 1000
        trace = GroundingTrace(
            doc_id=doc_id, field=field_name, value=value,
            final_decision="grounded", final_method=GroundingMethod.FUZZY.value,
            final_confidence=best_ratio,
            final_bbox=spans[-1].bbox,
            spans=spans, total_wall_time_ms=total_ms, timestamp=now,
        )
        record_trace(trace)
        return GroundingMethod.FUZZY, (best_fuzzy_anchor,)

    # Layer 4: SEMANTIC
    t0 = time.monotonic()
    target_emb = get_embedding(target)
    best_cosine = 0.0
    best_sem_chunk = None
    for chunk in chunks:
        chunk_emb = chunk.embedding if chunk.embedding else get_embedding(chunk.text)
        sim = cosine_similarity(target_emb, chunk_emb)
        if sim >= verifier.semantic_threshold and sim > best_cosine:
            best_cosine = sim
            best_sem_chunk = chunk
    sem_found = best_sem_chunk is not None
    spans.append(CascadeSpan(
        layer="SEMANTIC", decision="grounded" if sem_found else "skip",
        confidence=best_cosine if sem_found else 0.0,
        method=GroundingMethod.SEMANTIC.value if sem_found else "",
        wall_time_ms=(time.monotonic() - t0) * 1000,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))
    if sem_found:
        anchor = Anchor(
            chunk_id=best_sem_chunk.chunk_id, char_span=(0, len(best_sem_chunk.text)),
            page=best_sem_chunk.page, bbox=best_sem_chunk.bbox,
        )
        total_ms = (time.monotonic() - t_start) * 1000
        trace = GroundingTrace(
            doc_id=doc_id, field=field_name, value=value,
            final_decision="grounded", final_method=GroundingMethod.SEMANTIC.value,
            final_confidence=best_cosine,
            final_bbox=[best_sem_chunk.bbox.x0, best_sem_chunk.bbox.y0,
                        best_sem_chunk.bbox.x1 - best_sem_chunk.bbox.x0,
                        best_sem_chunk.bbox.y1 - best_sem_chunk.bbox.y0] if best_sem_chunk.bbox else None,
            spans=spans, total_wall_time_ms=total_ms, timestamp=now,
        )
        record_trace(trace)
        return GroundingMethod.SEMANTIC, (anchor,)

    # Layer 5: VISUAL (bbox-based)
    t0 = time.monotonic()
    candidate_bbox = verifier._parse_candidate_bbox(source_span)
    visual_found = False
    visual_anchor = None
    if candidate_bbox is not None:
        best_iou = 0.0
        best_vis_chunk = None
        for chunk in chunks:
            if chunk.bbox is None:
                continue
            iou = bbox_iou(candidate_bbox, chunk.bbox)
            if iou >= verifier.visual_threshold and iou > best_iou:
                best_iou = iou
                best_vis_chunk = chunk
        if best_vis_chunk:
            visual_found = True
            visual_anchor = Anchor(
                chunk_id=best_vis_chunk.chunk_id, char_span=(0, len(best_vis_chunk.text)),
                page=best_vis_chunk.page, bbox=best_vis_chunk.bbox,
            )
    spans.append(CascadeSpan(
        layer="VISUAL", decision="grounded" if visual_found else "skip",
        confidence=best_iou if visual_found else 0.0,
        method=GroundingMethod.VISUAL.value if visual_found else "",
        wall_time_ms=(time.monotonic() - t0) * 1000,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))
    if visual_found:
        total_ms = (time.monotonic() - t_start) * 1000
        trace = GroundingTrace(
            doc_id=doc_id, field=field_name, value=value,
            final_decision="grounded", final_method=GroundingMethod.VISUAL.value,
            final_confidence=best_iou,
            final_bbox=spans[-1].bbox,
            spans=spans, total_wall_time_ms=total_ms, timestamp=now,
        )
        record_trace(trace)
        return GroundingMethod.VISUAL, (visual_anchor,)

    # Layer 6: BLOCK
    spans.append(CascadeSpan(
        layer="BLOCK", decision="refused", confidence=0.0,
        method=GroundingMethod.BLOCK.value,
        timestamp=now, doc_id=doc_id, field=field_name,
    ))
    total_ms = (time.monotonic() - t_start) * 1000
    trace = GroundingTrace(
        doc_id=doc_id, field=field_name, value=value,
        final_decision="refused", final_method=GroundingMethod.BLOCK.value,
        final_confidence=0.0, final_bbox=None,
        spans=spans, total_wall_time_ms=total_ms, timestamp=now,
    )
    record_trace(trace)
    return GroundingMethod.BLOCK, ()


def instrument_fastapi(app) -> None:
    """Auto-instrument a FastAPI app with OpenTelemetry tracing."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumented with OpenTelemetry")
    except Exception as e:
        logger.warning(f"FastAPI instrumentation skipped: {e}")