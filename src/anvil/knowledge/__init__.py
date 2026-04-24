"""Knowledge graph construction — DocumentElements → typed graph."""

from __future__ import annotations

from anvil.knowledge.edge_types import EDGE_WEIGHTS
from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore
from anvil.knowledge.node_types import document_element_to_node

__all__ = [
    "EDGE_WEIGHTS",
    "GraphStore",
    "build_graph",
    "document_element_to_node",
]
