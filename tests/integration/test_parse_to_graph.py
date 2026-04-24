"""Integration test: parse synthetic standard → knowledge graph."""

from __future__ import annotations

from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore
from anvil.parsing.markdown_parser import parse_markdown_standard


def test_parse_to_graph_end_to_end(standard_md_path) -> None:
    elements = parse_markdown_standard(standard_md_path)
    assert len(elements) >= 8
    graph = build_graph(elements)
    store = GraphStore(graph)

    # All canonical paragraphs are present
    for ref in ["A-23", "A-25", "A-27", "A-32", "B-12", "M-23"]:
        assert store.find_by_paragraph_ref(ref), f"missing {ref}"

    # We can traverse from the thickness formula to the stress table in ≤3 hops
    a27_nodes = store.find_by_paragraph_ref("A-27")
    assert a27_nodes
    expanded = store.expand(a27_nodes, max_hops=3)
    titles = {str(graph.nodes[n].get("title", "")) for n in expanded}
    assert any("M-1" in t for t in titles)
    assert any("B-12" in t for t in titles)
