"""Convenience factory — builds the full pipeline for scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anvil.evaluation.ablation import ABLATIONS, PipelineAblation
from anvil.generation.calculation_engine import CalculationEngine, CitationBuilder
from anvil.generation.generator import AnvilGenerator
from anvil.generation.llm_backend import LLMBackend, get_default_backend
from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.retrieval.embedder import Embedder, get_default_embedder
from anvil.retrieval.hybrid_retriever import HybridRetriever
from anvil.retrieval.vector_store import VectorStore
from anvil.schemas.document import DocumentElement


@dataclass
class Pipeline:
    """A fully-wired pipeline: parsed standard + KG + retriever + generator."""

    elements: list[DocumentElement]
    graph_store: GraphStore
    embedder: Embedder
    vector_store: VectorStore
    retriever: HybridRetriever
    generator: AnvilGenerator


def build_pipeline(
    standard_path: str | Path | None = None,
    backend: LLMBackend | None = None,
    embedder: Embedder | None = None,
    ablation: PipelineAblation | str | None = None,
) -> Pipeline:
    """Build the full pipeline from the synthetic standard.

    Defaults:
        - `standard_path` → data/synthetic/standard.md
        - `backend`       → resolved by `get_default_backend()` —
                            honors `ANVIL_LLM_BACKEND` env var; falls
                            back to FakeLLMBackend with a loud WARNING
                            so prod deploys that forgot to set the env
                            see it immediately.
        - `embedder`      → selected by ANVIL_EMBEDDER env var (default: hash)
        - `ablation`      → "baseline" (full hybrid + pinned + gates ON)

    `ablation` may be a `PipelineAblation` instance, the name of a
    catalog entry (`"baseline"`, `"bm25-only"`, `"no-pinned"`, …), or
    None — None resolves to `"baseline"`.
    """
    if ablation is None:
        config = ABLATIONS["baseline"]
    elif isinstance(ablation, str):
        if ablation not in ABLATIONS:
            raise ValueError(
                f"Unknown ablation {ablation!r}. Known: {sorted(ABLATIONS)}."
            )
        config = ABLATIONS[ablation]
    else:
        config = ablation

    path = Path(standard_path) if standard_path else (
        Path(__file__).resolve().parents[2] / "data" / "synthetic" / "standard.md"
    )
    elements = parse_markdown_standard(path)
    graph = build_graph(elements)
    graph_store = GraphStore(graph)

    embedder = embedder or get_default_embedder()
    vector_store = VectorStore(dim=embedder.dim)
    embeddings = embedder.encode([_element_text(e) for e in elements])
    vector_store.add(elements, embeddings)

    retriever = HybridRetriever(
        elements=elements,
        embedder=embedder,
        vector_store=vector_store,
        graph_store=graph_store,
        mode=config.retrieval_mode,
    )
    # Share the parsed elements with the citation builder so it quotes from
    # the live document — no duplicated parsing and no drift if the markdown
    # is swapped out at runtime.
    calc_engine = CalculationEngine(
        citation_builder=CitationBuilder.from_elements(elements)
    )
    generator = AnvilGenerator(
        retriever=retriever,
        backend=backend if backend is not None else get_default_backend(),
        calc_engine=calc_engine,
        # Same parsed elements as the citation builder — gives the
        # citation enforcer the ability to validate canonical-ref
        # quotes against the real document.
        element_index={e.element_id: e for e in elements},
        use_pinned_data=config.use_pinned_data,
        use_refusal_gate=config.use_refusal_gate,
        use_citation_enforcer=config.use_citation_enforcer,
    )
    return Pipeline(
        elements=elements,
        graph_store=graph_store,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator,
    )


def _element_text(el: DocumentElement) -> str:
    parts = [el.title or "", el.paragraph_ref or "", el.content]
    return " ".join(p for p in parts if p)
