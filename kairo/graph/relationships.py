"""
Kairo Relationship Inferrer — Deterministic relationship extraction.

Infers relationships between entities using deterministic rules (no LLM):
  - Same entity in multiple docs -> edge "appears_in"
  - Same vendor across invoices -> edge "supplies_to"
  - Same governing law across contracts -> edge "governed_by"
"""
from __future__ import annotations

from typing import Any

from kairo.graph.entities import GroundedEntity


def infer_relationships(entities_by_doc: dict[str, list[GroundedEntity]]) -> list[dict[str, Any]]:
    """Infer relationships between entities across documents.

    Args:
        entities_by_doc: Dict mapping doc_id -> list of entities in that doc.

    Returns:
        List of edge dicts: {source, target, relation, docs, bboxes}
    """
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # Collect all entities grouped by value for cross-doc matching
    value_to_entities: dict[str, list[GroundedEntity]] = {}
    for doc_id, entities in entities_by_doc.items():
        for ent in entities:
            key = f"{ent.entity_type}:{ent.value.lower()}"
            value_to_entities.setdefault(key, []).append(ent)

    # Rule 1: Same entity in multiple docs -> "appears_in"
    for key, ents in value_to_entities.items():
        if len(ents) > 1:
            doc_ids = list(set(e.source_doc for e in ents))
            if len(doc_ids) > 1:
                edge_key = (key, "appears_in", ",".join(sorted(doc_ids)))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": key,
                        "relation": "appears_in",
                        "docs": doc_ids,
                        "bboxes": [e.source_bbox for e in ents if e.source_bbox],
                        "entity_type": ents[0].entity_type,
                        "value": ents[0].value,
                    })

    # Rule 2: Same ORG (vendor) across invoices -> "supplies_to"
    org_entities = [e for ents in entities_by_doc.values() for e in ents if e.entity_type == "ORG"]
    org_by_value: dict[str, list[GroundedEntity]] = {}
    for org in org_entities:
        org_by_value.setdefault(org.value.lower(), []).append(org)

    for org_name, ents in org_by_value.items():
        if len(ents) > 1:
            doc_ids = list(set(e.source_doc for e in ents))
            if len(doc_ids) > 1:
                edge_key = (org_name, "supplies_to", ",".join(sorted(doc_ids)))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": f"ORG:{org_name}",
                        "relation": "supplies_to",
                        "docs": doc_ids,
                        "bboxes": [e.source_bbox for e in ents if e.source_bbox],
                        "entity_type": "ORG",
                        "value": ents[0].value,
                    })

    # Rule 3: Same JURISDICTION across contracts -> "governed_by"
    juris_entities = [e for ents in entities_by_doc.values() for e in ents if e.entity_type == "JURISDICTION"]
    juris_by_value: dict[str, list[GroundedEntity]] = {}
    for j in juris_entities:
        juris_by_value.setdefault(j.value.lower(), []).append(j)

    for juris_name, ents in juris_by_value.items():
        if len(ents) > 1:
            doc_ids = list(set(e.source_doc for e in ents))
            if len(doc_ids) > 1:
                edge_key = (juris_name, "governed_by", ",".join(sorted(doc_ids)))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": f"JURISDICTION:{juris_name}",
                        "relation": "governed_by",
                        "docs": doc_ids,
                        "bboxes": [e.source_bbox for e in ents if e.source_bbox],
                        "entity_type": "JURISDICTION",
                        "value": ents[0].value,
                    })

    return edges