"""
Kairo Grounded Entity Extractor — Extracts entities from grounded field values.

Each entity becomes a graph node with attributes: type, value, source_doc,
source_bbox, source_page, confidence. Every node traces to a pixel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kernel.core.data_model import Extraction


@dataclass
class GroundedEntity:
    """An entity extracted from a grounded field value."""
    entity_type: str  # PERSON, ORG, DATE, AMOUNT, JURISDICTION
    value: str
    source_doc: str
    source_bbox: list[float] | None
    source_page: int
    confidence: float
    source_field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.entity_type,
            "value": self.value,
            "source_doc": self.source_doc,
            "source_bbox": self.source_bbox,
            "source_page": self.source_page,
            "confidence": self.confidence,
            "source_field": self.source_field,
        }


def extract_entities(extractions: list[Extraction], doc_id: str) -> list[GroundedEntity]:
    """Extract entities from grounded extraction field values.

    Args:
        extractions: List of Extraction objects from the pipeline.
        doc_id: The document ID these extractions came from.

    Returns:
        List of GroundedEntity objects, each with bbox provenance.
    """
    entities: list[GroundedEntity] = []

    for ext in extractions:
        if ext.method.value == "block":
            continue  # skip ungrounded extractions

        bbox = None
        page = 1
        if ext.anchors:
            a = ext.anchors[0]
            bbox = [a.bbox.x0, a.bbox.y0, a.bbox.x1 - a.bbox.x0, a.bbox.y1 - a.bbox.y0]
            page = a.page

        value = str(ext.value)
        field = ext.field_name

        # Extract entities based on field type
        if field in ("vendor_name", "parties"):
            # ORG entities from vendor/party names
            if field == "parties":
                # parties is a JSON list
                try:
                    import json
                    party_list = json.loads(value)
                    for party in party_list:
                        # Clean party name (remove legal suffixes for matching)
                        clean = re.sub(r'\s+(LLP|LP|Inc|Ltd|Pty|Corp|Corporation|LLC)\.?', '', party).strip()
                        if clean:
                            entities.append(GroundedEntity(
                                entity_type="ORG", value=party,
                                source_doc=doc_id, source_bbox=bbox,
                                source_page=page, confidence=ext.confidence,
                                source_field=field,
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                entities.append(GroundedEntity(
                    entity_type="ORG", value=value,
                    source_doc=doc_id, source_bbox=bbox,
                    source_page=page, confidence=ext.confidence,
                    source_field=field,
                ))

        elif field in ("effective_date", "termination_date", "due_date", "invoice_date"):
            # DATE entities
            entities.append(GroundedEntity(
                entity_type="DATE", value=value,
                source_doc=doc_id, source_bbox=bbox,
                source_page=page, confidence=ext.confidence,
                source_field=field,
            ))

        elif field in ("total_amount", "tax_amount"):
            # AMOUNT entities
            entities.append(GroundedEntity(
                entity_type="AMOUNT", value=value,
                source_doc=doc_id, source_bbox=bbox,
                source_page=page, confidence=ext.confidence,
                source_field=field,
            ))

        elif field == "governing_law":
            # JURISDICTION entities
            entities.append(GroundedEntity(
                entity_type="JURISDICTION", value=value,
                source_doc=doc_id, source_bbox=bbox,
                source_page=page, confidence=ext.confidence,
                source_field=field,
            ))

        elif field == "authors":
            # PERSON entities from paper authors
            try:
                import json
                author_list = json.loads(value)
                for author in author_list:
                    entities.append(GroundedEntity(
                        entity_type="PERSON", value=author,
                        source_doc=doc_id, source_bbox=bbox,
                        source_page=page, confidence=ext.confidence,
                        source_field=field,
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

    return entities