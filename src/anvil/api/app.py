"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from anvil.api.middleware import structured_log_middleware
from anvil.api.routes import build_router
from anvil.generation.calculation_engine import CalculationEngine, CitationBuilder
from anvil.generation.generator import AnvilGenerator
from anvil.generation.llm_backend import get_default_backend
from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.retrieval.embedder import get_default_embedder
from anvil.retrieval.hybrid_retriever import HybridRetriever
from anvil.retrieval.vector_store import VectorStore


def create_app(standard_path: str | Path | None = None) -> FastAPI:
    """Build a fully-wired FastAPI app ready to serve requests."""
    path = Path(standard_path) if standard_path else (
        Path(__file__).resolve().parents[3] / "data" / "synthetic" / "standard.md"
    )
    elements = parse_markdown_standard(path)
    graph = build_graph(elements)
    graph_store = GraphStore(graph)

    embedder = get_default_embedder()
    vector_store = VectorStore(dim=embedder.dim)
    embeddings = embedder.encode([el.content for el in elements])
    vector_store.add(elements, embeddings)

    retriever = HybridRetriever(
        elements=elements,
        embedder=embedder,
        vector_store=vector_store,
        graph_store=graph_store,
    )

    backend = get_default_backend()
    calc_engine = CalculationEngine(
        citation_builder=CitationBuilder.from_elements(elements)
    )
    generator = AnvilGenerator(
        retriever=retriever,
        backend=backend,
        calc_engine=calc_engine,
        # Lets the citation enforcer validate canonical-ref quotes
        # against the parsed standard even when retrieval misses the
        # cited element (e.g. pinned Table M-1 lookups).
        element_index={e.element_id: e for e in elements},
    )

    app = FastAPI(
        title="anvil",
        version="0.1.0",
        description=(
            "Compliance-grade retrieval-augmented reasoning over the "
            "Synthetic Pressure Equipment Standard (SPES-1)."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BaseHTTPMiddleware, dispatch=structured_log_middleware)
    app.include_router(
        build_router(
            generator=generator, graph_store=graph_store, calc_engine=calc_engine
        )
    )
    return app
