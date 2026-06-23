"""
Kairo Graph Store — NetworkX-backed grounded knowledge graph.

Every node carries a bbox and source document — it is a grounded knowledge
graph where every node traces to a pixel. Persists to data/graph.json.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import networkx as nx

from kairo.graph.entities import GroundedEntity, extract_entities
from kairo.graph.relationships import infer_relationships
from kernel.core.data_model import Extraction

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = pathlib.Path("data/graph.json")


class GroundedKnowledgeGraph:
    """A knowledge graph where every node has bbox provenance."""

    def __init__(self, graph: nx.Graph | None = None) -> None:
        self._graph = graph or nx.Graph()

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def add_entity(self, entity: GroundedEntity) -> str:
        """Add an entity as a graph node. Returns the node ID."""
        node_id = f"{entity.entity_type}:{entity.value.lower()}"
        if node_id not in self._graph:
            self._graph.add_node(node_id, **entity.to_dict())
        else:
            # Merge: add source doc info
            existing = self._graph.nodes[node_id]
            if "source_docs" not in existing:
                existing["source_docs"] = [existing.get("source_doc", "")]
            if entity.source_doc not in existing["source_docs"]:
                existing["source_docs"].append(entity.source_doc)
        return node_id

    def add_edge(self, source: str, target: str, relation: str, **attrs) -> None:
        """Add a relationship edge between two nodes."""
        self._graph.add_edge(source, target, relation=relation, **attrs)

    def build_from_extractions(
        self,
        extractions_by_doc: dict[str, list[Extraction]],
    ) -> dict[str, Any]:
        """Build the graph from extractions across multiple documents.

        Args:
            extractions_by_doc: Dict mapping doc_id -> list of Extractions.

        Returns:
            Build stats: {nodes_added, edges_added, total_nodes, total_edges}
        """
        nodes_before = self._graph.number_of_nodes()
        edges_before = self._graph.number_of_edges()

        # Extract entities per doc
        entities_by_doc: dict[str, list[GroundedEntity]] = {}
        for doc_id, extractions in extractions_by_doc.items():
            entities = extract_entities(extractions, doc_id)
            entities_by_doc[doc_id] = entities
            for ent in entities:
                self.add_entity(ent)

        # Infer relationships
        relationships = infer_relationships(entities_by_doc)
        for rel in relationships:
            source_node = rel["source"]
            for doc_id in rel["docs"]:
                target_node = f"DOC:{doc_id}"
                if target_node not in self._graph:
                    self._graph.add_node(target_node, type="DOCUMENT", doc_id=doc_id)
                self.add_edge(source_node, target_node, rel["relation"],
                              bboxes=rel.get("bboxes", []))

        nodes_after = self._graph.number_of_nodes()
        edges_after = self._graph.number_of_edges()

        return {
            "nodes_added": nodes_after - nodes_before,
            "edges_added": edges_after - edges_before,
            "total_nodes": nodes_after,
            "total_edges": edges_after,
        }

    def query(self, keyword: str) -> list[dict[str, Any]]:
        """Query the graph by keyword. Returns matching nodes with provenance.

        NOT an LLM call — pure keyword + entity matching + graph traversal.
        """
        keyword_lower = keyword.lower()
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if keyword_lower in node_id.lower() or keyword_lower in str(data.get("value", "")).lower():
                results.append({
                    "node_id": node_id,
                    "type": data.get("entity_type", data.get("type", "")),
                    "value": data.get("value", ""),
                    "source_doc": data.get("source_doc", ""),
                    "source_bbox": data.get("source_bbox"),
                    "source_page": data.get("source_page", 1),
                    "confidence": data.get("confidence", 0.0),
                    "connected_docs": list(self._graph.neighbors(node_id)),
                })
        return results

    def save(self, path: pathlib.Path | None = None) -> None:
        """Persist the graph to JSON."""
        path = path or DEFAULT_GRAPH_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Graph saved to {path} ({self._graph.number_of_nodes()} nodes)")

    def load(self, path: pathlib.Path | None = None) -> None:
        """Load the graph from JSON."""
        path = path or DEFAULT_GRAPH_PATH
        if path.exists():
            data = json.loads(path.read_text())
            self._graph = nx.node_link_graph(data)
            logger.info(f"Graph loaded from {path} ({self._graph.number_of_nodes()} nodes)")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph for API responses."""
        return {
            "nodes": [
                {"id": node_id, **data}
                for node_id, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self._graph.edges(data=True)
            ],
            "stats": {
                "total_nodes": self._graph.number_of_nodes(),
                "total_edges": self._graph.number_of_edges(),
            },
        }