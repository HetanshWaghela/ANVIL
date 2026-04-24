"""Persistence and query helpers for the anvil knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


class GraphStore:
    """Thin wrapper around a `networkx.DiGraph` with save/load and common queries."""

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()

    # ---- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save as JSON (node-link format) for inspection and tests."""
        data = nx.node_link_data(self.graph, edges="edges")
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GraphStore:
        data = json.loads(Path(path).read_text())
        try:
            g = nx.node_link_graph(data, edges="edges", directed=True)
        except TypeError:
            g = nx.node_link_graph(data, directed=True)
        return cls(g)

    # ---- queries -----------------------------------------------------------

    def find_by_paragraph_ref(self, paragraph_ref: str) -> list[str]:
        """All node_ids whose `paragraph_ref` matches (case-insensitive)."""
        target = paragraph_ref.upper()
        return [
            n
            for n, attrs in self.graph.nodes(data=True)
            if (attrs.get("paragraph_ref") or "").upper() == target
        ]

    def expand(self, node_ids: list[str], max_hops: int = 2) -> set[str]:
        """Return all nodes within `max_hops` of any node in `node_ids`.

        Follows both incoming and outgoing edges so that retrieving UG-27
        pulls in UW-12 (outgoing) AND any paragraph that references UG-27
        (incoming).
        """
        visited: set[str] = set(node_ids)
        frontier: set[str] = set(node_ids)
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for n in frontier:
                if n not in self.graph:
                    continue
                neighbors = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                next_frontier |= neighbors - visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def neighbors(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return (target, edge_type, reference_text) for outgoing edges."""
        if node_id not in self.graph:
            return []
        out: list[tuple[str, str, str]] = []
        for _, tgt, data in self.graph.out_edges(node_id, data=True):
            out.append(
                (
                    tgt,
                    str(data.get("edge_type", "")),
                    str(data.get("reference_text", "")),
                )
            )
        return out

    def is_weakly_connected(self) -> bool:
        if len(self.graph) == 0:
            return True
        return nx.is_weakly_connected(self.graph)

    def orphan_nodes(self) -> list[str]:
        """Nodes with no incoming or outgoing edges."""
        return [n for n in self.graph.nodes if self.graph.degree(n) == 0]
