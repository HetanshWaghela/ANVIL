"""Phase 3 tests: retrieval pipeline."""

from __future__ import annotations

from anvil.retrieval import (
    DeterministicHashEmbedder,
    reciprocal_rank_fusion,
)
from anvil.schemas.retrieval import RetrievalQuery


def test_rrf_basic() -> None:
    scores = reciprocal_rank_fusion(
        [["a", "b", "c"], ["c", "a", "b"]], k=60
    )
    # "a" is rank 1 in list1, rank 2 in list2 → highest
    assert list(scores.keys())[0] == "a"


def test_rrf_k_constant_effect() -> None:
    s1 = reciprocal_rank_fusion([["a", "b"]], k=0)
    s2 = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert s1["a"] > s2["a"]  # smaller k → larger top score


def test_deterministic_hash_embedder_shape() -> None:
    emb = DeterministicHashEmbedder(dim=128)
    v = emb.encode(["hello world", "another sentence"])
    assert v.shape == (2, 128)


def test_deterministic_hash_embedder_normalized() -> None:
    import numpy as np

    emb = DeterministicHashEmbedder(dim=128)
    v = emb.encode(["thickness formula"])
    norm = float(np.linalg.norm(v[0]))
    # Norm should be 1 or 0 (empty input)
    assert abs(norm - 1.0) < 1e-5


def test_retrieval_cylindrical_query_finds_a27_and_b12(pipeline) -> None:
    q = RetrievalQuery(
        text="wall thickness formula for a cylindrical shell with joint efficiency",
        top_k=10,
        enable_graph_expansion=True,
    )
    results = pipeline.retriever.retrieve(q)
    refs = {(r.paragraph_ref or "") for r in results}
    # Must have A-27 (the formula) and B-12 (joint efficiency via graph)
    assert any(p.startswith("A-27") for p in refs), refs
    assert any(p.startswith("B-12") or "B-12" in p for p in refs), refs


def test_retrieval_material_query_finds_m1(pipeline) -> None:
    q = RetrievalQuery(
        text="allowable stress for SM-516 Gr 70 at 300°C from Table M-1",
        top_k=10,
        enable_graph_expansion=True,
    )
    results = pipeline.retriever.retrieve(q)
    refs = {(r.paragraph_ref or "") for r in results}
    titles = {r.content[:60] for r in results}
    assert any("M-1" in p for p in refs) or any("M-1" in t for t in titles), (refs, titles)


def test_retrieval_ood_query_low_scores(pipeline) -> None:
    q = RetrievalQuery(text="weather in San Jose today", top_k=10)
    results = pipeline.retriever.retrieve(q)
    # All scores should be low; in particular the max should be under 0.2
    if results:
        assert max(r.score for r in results) < 0.2, [r.score for r in results]
