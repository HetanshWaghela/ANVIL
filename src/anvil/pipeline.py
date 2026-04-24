"""Convenience factory — builds the full pipeline for scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anvil.generation.calculation_engine import CalculationEngine, CitationBuilder
from anvil.generation.generator import AnvilGenerator
from anvil.generation.llm_backend import FakeLLMBackend, LLMBackend
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
) -> Pipeline:
    """Build the full pipeline from the synthetic standard.

    Defaults:
        - `standard_path` → data/synthetic/standard.md
        - `backend`       → FakeLLMBackend (deterministic, no network)
        - `embedder`      → selected by ANVIL_EMBEDDER env var (default: hash)
    """
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
    )
    # Share the parsed elements with the citation builder so it quotes from
    # the live document — no duplicated parsing and no drift if the markdown
    # is swapped out at runtime.
    calc_engine = CalculationEngine(
        citation_builder=CitationBuilder.from_elements(elements)
    )
    generator = AnvilGenerator(
        retriever=retriever,
        backend=backend or FakeLLMBackend(),
        calc_engine=calc_engine,
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
