"""Phase 2 tests: knowledge graph construction and queries."""

from __future__ import annotations

from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore


def test_graph_builds_nontrivially(parsed_elements) -> None:
    g = build_graph(parsed_elements)
    assert len(g.nodes) >= 8
    assert len(g.edges) >= 5


def test_a27_has_outgoing_edges_to_tables(parsed_elements) -> None:
    g = build_graph(parsed_elements)
    store = GraphStore(g)
    a27_nodes = store.find_by_paragraph_ref("A-27")
    assert a27_nodes, "A-27 node not found in graph"
    # Collect all targets reachable from any A-27* node (section or formula)
    a27_like = [n for n in g.nodes if str(g.nodes[n].get("paragraph_ref", "")).startswith("A-27")]
    targets: set[str] = set()
    for n in a27_like:
        targets |= set(g.successors(n))
    # Must reach both tables M-1 and B-12 from A-27 family
    target_para_refs = {
        str(g.nodes[t].get("paragraph_ref", "")) for t in targets
    }
    # Table node paragraph_refs should include M-1, B-12 (from titles) — at
    # minimum, neighbors must include nodes whose titles mention those tables
    titles = {str(g.nodes[t].get("title", "")) for t in targets}
    assert any("M-1" in p for p in target_para_refs) or any(
        "M-1" in t for t in titles
    )
    assert any("B-12" in p for p in target_para_refs) or any(
        "B-12" in t for t in titles
    )


def test_graph_expand_reaches_tables(parsed_elements) -> None:
    g = build_graph(parsed_elements)
    store = GraphStore(g)
    a27 = store.find_by_paragraph_ref("A-27")
    assert a27
    expanded = store.expand(a27, max_hops=2)
    # The expanded set should include Table M-1 and Table B-12 elements
    titles = {str(g.nodes[n].get("title", "")) for n in expanded}
    assert any("M-1" in t for t in titles), titles
    assert any("B-12" in t for t in titles), titles


def test_graph_no_self_edges(parsed_elements) -> None:
    g = build_graph(parsed_elements)
    for u, v in g.edges:
        assert u != v


def test_graph_save_and_load_roundtrip(parsed_elements, tmp_path) -> None:
    g = build_graph(parsed_elements)
    store = GraphStore(g)
    path = tmp_path / "kg.json"
    store.save(path)
    restored = GraphStore.load(path)
    assert len(restored.graph.nodes) == len(g.nodes)
    assert len(restored.graph.edges) == len(g.edges)


def test_every_edge_has_reference_text(parsed_elements) -> None:
    g = build_graph(parsed_elements)
    for _, _, data in g.edges(data=True):
        assert "edge_type" in data
        assert "reference_text" in data
        assert "weight" in data
