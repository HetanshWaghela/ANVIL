"""Build a typed NetworkX DiGraph from DocumentElements."""

from __future__ import annotations

import re

import networkx as nx

from anvil.knowledge.edge_types import EDGE_WEIGHTS, classify_edge
from anvil.knowledge.node_types import document_element_to_node
from anvil.schemas.document import DocumentElement, ElementType
from anvil.schemas.knowledge_graph import EdgeType, GraphEdge, GraphNode

# Matches a table reference like "Table M-1", "Table B-12", or "Table UCS-23a"
# in any variable source string. Captures the table id ("M-1", etc.).
_TABLE_REF_IN_SOURCE = re.compile(
    r"Table\s+([A-Z]+-\d+[A-Za-z]?)", re.IGNORECASE
)


def build_graph(elements: list[DocumentElement]) -> nx.DiGraph:
    """Construct a typed directed graph from a list of DocumentElements.

    Edges encoded:
      - CONTAINS: parent → child (from DocumentElement.parent_id)
      - REFERENCES / REQUIRES / DEFINES / CONSTRAINS: from CrossReference objects
      - REQUIRES: formula → table for every standard variable (best-effort,
        wires formulas to Table M-1 and Table B-12 when S or E appears in vars)

    Returns:
        A `networkx.DiGraph` with node attributes matching `GraphNode` and
        edge attributes matching `GraphEdge`.
    """
    G: nx.DiGraph = nx.DiGraph()

    # 1) Nodes
    by_id: dict[str, DocumentElement] = {e.element_id: e for e in elements}
    for el in elements:
        node = document_element_to_node(el)
        G.add_node(node.node_id, **_node_attrs(node))

    # 2) CONTAINS edges from parent_id
    for el in elements:
        if el.parent_id and el.parent_id in by_id:
            _add_edge(
                G,
                GraphEdge(
                    source=el.parent_id,
                    target=el.element_id,
                    edge_type=EdgeType.CONTAINS,
                    reference_text="contains",
                    weight=EDGE_WEIGHTS[EdgeType.CONTAINS],
                ),
            )

    # 3) Cross-reference edges
    for el in elements:
        for xref in el.cross_references:
            target_el = by_id.get(xref.target_id)
            if target_el is None:
                continue
            target_is_table = target_el.element_type == ElementType.TABLE
            etype = classify_edge(xref.reference_text, target_is_table)
            _add_edge(
                G,
                GraphEdge(
                    source=xref.source_id,
                    target=xref.target_id,
                    edge_type=etype,
                    reference_text=xref.reference_text,
                    weight=EDGE_WEIGHTS[etype],
                ),
            )

    # 4) Formula → table REQUIRES edges derived from variable metadata.
    # We parse the actual "Table X-N" reference out of each variable's source
    # string and look it up in the table_id_index. No hardcoded table IDs:
    # a new table (e.g. "Table UCS-23a") works automatically as long as the
    # formula variable's source mentions it.
    table_id_index: dict[str, str] = {}
    for el in elements:
        if el.element_type == ElementType.TABLE and el.table is not None:
            table_id_index[el.table.table_id.upper()] = el.element_id

    for el in elements:
        if el.element_type != ElementType.FORMULA or el.formula is None:
            continue
        # Every table id that appears in any variable source gets an edge.
        referenced_tables: set[str] = set()
        for var in el.formula.variables:
            for match in _TABLE_REF_IN_SOURCE.finditer(var.source):
                referenced_tables.add(match.group(1).upper())
        for table_id in referenced_tables:
            target_id = table_id_index.get(table_id)
            if not target_id or target_id == el.element_id:
                continue
            _add_edge(
                G,
                GraphEdge(
                    source=el.element_id,
                    target=target_id,
                    edge_type=EdgeType.REQUIRES,
                    reference_text=f"formula requires values from Table {table_id}",
                    weight=EDGE_WEIGHTS[EdgeType.REQUIRES],
                ),
            )

    return G


def _node_attrs(node: GraphNode) -> dict[str, str]:
    return {
        "node_type": node.node_type.value,
        "paragraph_ref": node.paragraph_ref or "",
        "title": node.title or "",
        "content_summary": node.content_summary,
        "source_element_id": node.source_element_id or "",
    }


def _add_edge(G: nx.DiGraph, edge: GraphEdge) -> None:
    # De-duplicate: if a stronger (higher-weight) edge already exists, keep it
    existing = G.get_edge_data(edge.source, edge.target)
    if existing and existing.get("weight", 0.0) >= edge.weight:
        return
    G.add_edge(
        edge.source,
        edge.target,
        edge_type=edge.edge_type.value,
        reference_text=edge.reference_text,
        weight=edge.weight,
    )
