"""M3 — pipeline ablation flag tests.

Locks in:
  * the 7 ablation catalog entries from the plan are present and named
    exactly as the plan specifies — adding an 8th is a deliberate edit;
  * each ablation flag is actually plumbed through to the place that
    measures it (retriever mode → retrieval lists; pinned flag →
    calculation engine bypass; refusal flag → no pre-LLM refusal;
    citation-enforcer flag → no validation post-LLM);
  * the regression invariant from plan §4.4: disabling pinned-data
    must drop calculation_correctness across calculation queries.
"""

from __future__ import annotations

from typing import Any

import pytest

from anvil.evaluation.ablation import ABLATIONS, PipelineAblation, get_ablation
from anvil.pipeline import build_pipeline
from anvil.schemas.retrieval import RetrievalQuery

# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


def test_ablation_catalog_locks_seven_named_configs() -> None:
    """REGRESSION: the plan specifies exactly 7 ablations. An 8th MUST
    be a deliberate decision (ADR + run-logger entry), so this test
    guards against silent additions/removals."""
    expected = {
        "baseline",
        "bm25-only",
        "vector-only",
        "no-graph",
        "no-pinned",
        "no-refusal",
        "no-citation-enforcer",
    }
    assert set(ABLATIONS) == expected
    # And every entry round-trips through `get_ablation`.
    for name in expected:
        cfg = get_ablation(name)
        assert isinstance(cfg, PipelineAblation)
        assert cfg.name == name


def test_get_ablation_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError) as exc:
        get_ablation("does-not-exist")
    assert "does-not-exist" in str(exc.value)


# ---------------------------------------------------------------------------
# Retrieval-mode plumbing
# ---------------------------------------------------------------------------


def test_baseline_pipeline_uses_hybrid_retriever_mode() -> None:
    pipe = build_pipeline()
    assert pipe.retriever.mode == "hybrid"


@pytest.mark.parametrize(
    "ablation_name,expected_mode",
    [
        ("bm25-only", "bm25_only"),
        ("vector-only", "vector_only"),
        ("no-graph", "hybrid_no_graph"),
        ("no-pinned", "hybrid"),  # retrieval is unchanged
    ],
)
def test_pipeline_threads_retrieval_mode(
    ablation_name: str, expected_mode: str
) -> None:
    pipe = build_pipeline(ablation=ablation_name)
    assert pipe.retriever.mode == expected_mode


def test_unknown_retriever_mode_raises_at_construction() -> None:
    """A typo in the mode string MUST fail loudly at construction so an
    ablation can't silently masquerade as the baseline."""
    import networkx as nx

    from anvil.knowledge.graph_store import GraphStore
    from anvil.parsing.markdown_parser import parse_markdown_standard
    from anvil.retrieval.embedder import get_default_embedder
    from anvil.retrieval.hybrid_retriever import HybridRetriever
    from anvil.retrieval.vector_store import VectorStore

    pipe = build_pipeline()
    with pytest.raises(Exception) as exc:
        HybridRetriever(
            elements=pipe.elements,
            embedder=pipe.embedder,
            vector_store=pipe.vector_store,
            graph_store=pipe.graph_store,
            mode="hybridish",  # typo
        )
    assert "hybridish" in str(exc.value) or "Unknown" in str(exc.value)
    # Silence linters about unused imports kept for clarity.
    _ = (parse_markdown_standard, GraphStore, get_default_embedder, VectorStore, nx)


# ---------------------------------------------------------------------------
# Generator-side plumbing
# ---------------------------------------------------------------------------


def test_baseline_generator_has_all_gates_on() -> None:
    pipe = build_pipeline()
    g = pipe.generator
    assert g.use_pinned_data is True
    assert g.use_refusal_gate is True
    assert g.use_citation_enforcer is True


@pytest.mark.parametrize(
    "ablation_name,attr,expected",
    [
        ("no-pinned", "use_pinned_data", False),
        ("no-refusal", "use_refusal_gate", False),
        ("no-citation-enforcer", "use_citation_enforcer", False),
    ],
)
def test_generator_threads_each_gate_flag(
    ablation_name: str, attr: str, expected: bool
) -> None:
    pipe = build_pipeline(ablation=ablation_name)
    assert getattr(pipe.generator, attr) is expected


# ---------------------------------------------------------------------------
# Behavioral: each ablation actually changes runtime behavior
# ---------------------------------------------------------------------------


def _retrieve(
    pipe: Any, query: str, top_k: int = 10
) -> list[Any]:
    return pipe.retriever.retrieve(
        RetrievalQuery(text=query, top_k=top_k, enable_graph_expansion=True)
    )


def test_bm25_only_results_are_pure_bm25_signal() -> None:
    """REGRESSION: in bm25_only mode the per-chunk `scores.vector`
    must be 0 for every result. Otherwise the ablation is silently
    pulling in the vector signal."""
    pipe = build_pipeline(ablation="bm25-only")
    chunks = _retrieve(pipe, "thickness formula for cylindrical shell")
    assert chunks
    for c in chunks:
        if c.scores is not None:
            assert c.scores.vector == 0.0


def test_vector_only_results_are_pure_vector_signal() -> None:
    """Mirror of the BM25 test: vector_only mode must zero out BM25."""
    pipe = build_pipeline(ablation="vector-only")
    chunks = _retrieve(pipe, "thickness formula for cylindrical shell")
    assert chunks
    for c in chunks:
        if c.scores is not None:
            assert c.scores.bm25 == 0.0


def test_no_graph_mode_skips_graph_expanded_chunks() -> None:
    """`hybrid_no_graph` must not surface chunks whose retrieval_source
    is `graph` — those come from the graph-expansion step."""
    pipe = build_pipeline(ablation="no-graph")
    chunks = _retrieve(pipe, "thickness formula for cylindrical shell")
    assert chunks
    assert all(c.retrieval_source != "graph" for c in chunks)


@pytest.mark.asyncio
async def test_no_pinned_data_drops_calculation_correctness() -> None:
    """REGRESSION (plan §4.4 acceptance): disabling pinned-data MUST
    drop calculation_correctness vs. baseline on calculation queries.

    We don't assert the exact magnitude (it depends on FakeLLMBackend
    behavior) — we assert a strict ordering: baseline >= no-pinned.
    """
    from pathlib import Path as _P

    from anvil.evaluation.dataset import load_golden_dataset
    from anvil.evaluation.metrics import calculation_correctness

    dataset_path = (
        _P(__file__).resolve().parents[2]
        / "tests"
        / "evaluation"
        / "golden_dataset.json"
    )
    examples = [
        e for e in load_golden_dataset(dataset_path) if e.expected_values
    ][:5]
    assert examples

    async def _score(ablation: str) -> list[float]:
        pipe = build_pipeline(ablation=ablation)
        scores: list[float] = []
        for ex in examples:
            outcome = await pipe.generator.generate(ex.query, top_k=10)
            m = calculation_correctness(
                outcome.response,
                ex.expected_values,
                tolerance=ex.numeric_tolerance,
            )
            scores.append(m.value)
        return scores

    baseline = await _score("baseline")
    no_pinned = await _score("no-pinned")

    assert sum(baseline) > sum(no_pinned), (
        f"baseline calc_correctness {baseline} should beat no-pinned {no_pinned}"
    )
    for b, n in zip(baseline, no_pinned, strict=True):
        assert b >= n, (
            "Per-example regression: no-pinned should not exceed baseline."
        )


@pytest.mark.asyncio
async def test_no_refusal_gate_lets_ood_query_reach_llm() -> None:
    """The OOD query "what is the weather in San Jose today?" with the
    refusal gate ON returns confidence=INSUFFICIENT (a refusal). With
    the gate OFF it must NOT short-circuit on `refusal_reason` — the
    response carries whatever the LLM produced. Proves the gate is what
    controls pre-LLM refusal."""
    from anvil.schemas.generation import ResponseConfidence

    pipe_base = build_pipeline(ablation="baseline")
    pipe_no_gate = build_pipeline(ablation="no-refusal")
    q = "what is the weather in San Jose today?"
    base = (await pipe_base.generator.generate(q, top_k=10)).response
    nog = (await pipe_no_gate.generator.generate(q, top_k=10)).response

    assert base.refusal_reason is not None
    assert base.confidence == ResponseConfidence.INSUFFICIENT
    # The no-gate path may still produce an insufficient response (e.g.
    # because no chunks were retrievable), but it MUST NOT carry the
    # refusal-gate's reason string.
    if nog.refusal_reason:
        assert "out of domain" not in nog.refusal_reason.lower()
        assert "low relevance" not in nog.refusal_reason.lower()
