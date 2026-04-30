"""Demo API routes for the local ANVIL interview demo.

Provides endpoints for:
  - PDF upload, parsing, graph/vector indexing
  - Corpus browsing (elements, tables, graph edges)
  - Pinned data inspection (materials, joint efficiencies)
  - Real-pipeline query execution (no fake backend allowed)
  - Demo query suggestions generated from parsed content

State is kept in-memory (dict of CorpusSession). Uploaded files are persisted
under data/private/uploads/ which is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field

from anvil.generation.calculation_engine import CalculationEngine, CitationBuilder
from anvil.generation.generator import AnvilGenerator
from anvil.generation.llm_backend import FakeLLMBackend, LLMBackend, get_default_backend
from anvil.knowledge.graph_builder import build_graph
from anvil.knowledge.graph_store import GraphStore
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.pinned import (
    JOINT_EFFICIENCIES,
    RT_LEVELS,
    get_material,
    list_materials,
)
from anvil.retrieval.embedder import get_default_embedder
from anvil.retrieval.hybrid_retriever import HybridRetriever
from anvil.retrieval.vector_store import VectorStore
from anvil.schemas.document import DocumentElement, ElementType

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_UPLOADS_DIR = _DATA_DIR / "private" / "uploads"
_SPES1_PATH = _DATA_DIR / "synthetic" / "standard.md"
_ASME_PRIVATE_DIR = _DATA_DIR / "private" / "asme"
_PINNED_FORMULAS_PATH = _DATA_DIR / "pinned" / "asme_viii1_formulas.json"


@dataclass(frozen=True)
class PreindexedCorpus:
    corpus_id: str
    display_name: str
    pdf_sha256: str
    markdown_path: Path
    elements_path: Path
    graph_path: Path


_PREINDEXED_CORPORA: tuple[PreindexedCorpus, ...] = (
    PreindexedCorpus(
        corpus_id="asme-viii-1-2023",
        display_name="ASME BPVC Section VIII Division 1 2023",
        pdf_sha256="076481ac40bb472d403431a5d017886b7c6c54caf110391e84cf59ac7dea7155",
        markdown_path=_ASME_PRIVATE_DIR / "asme_viii_1_2023_private.md",
        elements_path=_ASME_PRIVATE_DIR / "indexes_viii1_private" / "elements.json",
        graph_path=_ASME_PRIVATE_DIR / "indexes_viii1_private" / "kg.json",
    ),
)

_PREINDEXED_BY_SHA = {c.pdf_sha256: c for c in _PREINDEXED_CORPORA}


# ---------------------------------------------------------------------------
# In-memory corpus registry
# ---------------------------------------------------------------------------

@dataclass
class CorpusSession:
    corpus_id: str
    filename: str
    file_size: int
    page_count: int | None
    md_text: str
    elements: list[DocumentElement]
    graph_store: GraphStore
    retriever: HybridRetriever
    generator: AnvilGenerator
    stats: dict[str, Any]
    created_at: float
    is_spes1: bool = False


_registry: dict[str, CorpusSession] = {}


def _element_text(el: DocumentElement) -> str:
    parts = [el.title or "", el.paragraph_ref or "", el.content]
    return " ".join(p for p in parts if p)


def _build_corpus_session(
    corpus_id: str,
    filename: str,
    file_size: int,
    md_text: str,
    page_count: int | None,
    is_spes1: bool,
    backend: Any,
    embedder: Any,
) -> CorpusSession:
    elements = parse_markdown_standard(md_text)
    graph = build_graph(elements)
    graph_store = GraphStore(graph)

    return _build_corpus_session_from_elements(
        corpus_id=corpus_id,
        filename=filename,
        file_size=file_size,
        md_text=md_text,
        page_count=page_count,
        is_spes1=is_spes1,
        backend=backend,
        embedder=embedder,
        elements=elements,
        graph_store=graph_store,
    )


def _build_corpus_session_from_elements(
    corpus_id: str,
    filename: str,
    file_size: int,
    md_text: str,
    page_count: int | None,
    is_spes1: bool,
    backend: Any,
    embedder: Any,
    elements: list[DocumentElement],
    graph_store: GraphStore,
    warnings: list[str] | None = None,
    preindexed_source: str | None = None,
) -> CorpusSession:
    graph = graph_store.graph

    vector_store = VectorStore(dim=embedder.dim)
    embeddings = embedder.encode([_element_text(e) for e in elements])
    vector_store.add(elements, embeddings)

    retriever = HybridRetriever(
        elements=elements,
        embedder=embedder,
        vector_store=vector_store,
        graph_store=graph_store,
    )
    calc_engine = CalculationEngine(
        citation_builder=CitationBuilder.from_elements(elements)
    )
    # For non-SPES-1 corpora (uploaded real ASME PDFs), disable pinned data
    # and the SPES-1-specific refusal gate. The calc engine uses SM- prefixed
    # materials and SPES-1 paragraph refs (A-27, B-12) that don't exist in
    # real ASME documents. The refusal gate's material spec check also
    # incorrectly rejects real ASME materials (SA-516, etc.).
    generator = AnvilGenerator(
        retriever=retriever,
        backend=backend,
        calc_engine=calc_engine,
        element_index={e.element_id: e for e in elements},
        use_pinned_data=is_spes1,
    )

    type_counts = Counter(e.element_type.value for e in elements)
    tables = [e for e in elements if e.element_type == ElementType.TABLE]
    formulas = [e for e in elements if e.element_type == ElementType.FORMULA]
    para_refs = {e.paragraph_ref for e in elements if e.paragraph_ref}
    id_counts = Counter(e.element_id for e in elements)
    dup_ids = sum(1 for c in id_counts.values() if c > 1)

    stats: dict[str, Any] = {
        "filename": filename,
        "file_size": file_size,
        "page_count": page_count,
        "md_chars": len(md_text),
        "md_lines": len(md_text.splitlines()),
        "num_elements": len(elements),
        "type_counts": dict(type_counts),
        "num_tables": len(tables),
        "num_formulas": len(formulas),
        "num_graph_nodes": graph.number_of_nodes(),
        "num_graph_edges": graph.number_of_edges(),
        "num_paragraph_refs": len(para_refs),
        "duplicate_id_count": dup_ids,
        "warnings": warnings or [],
        "preindexed_source": preindexed_source,
    }

    return CorpusSession(
        corpus_id=corpus_id,
        filename=filename,
        file_size=file_size,
        page_count=page_count,
        md_text=md_text,
        elements=elements,
        graph_store=graph_store,
        retriever=retriever,
        generator=generator,
        stats=stats,
        created_at=time.time(),
        is_spes1=is_spes1,
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_preindexed_session(
    preindexed: PreindexedCorpus,
    filename: str,
    file_size: int,
    page_count: int | None,
    backend: LLMBackend,
    embedder: Any,
) -> CorpusSession:
    missing = [
        path
        for path in (
            preindexed.markdown_path,
            preindexed.elements_path,
            preindexed.graph_path,
        )
        if not path.exists()
    ]
    if missing:
        paths = ", ".join(str(p) for p in missing)
        raise HTTPException(
            500,
            f"Pre-indexed corpus artifacts are missing: {paths}. "
            "Run the private ASME ingest before the demo.",
        )

    md_text = preindexed.markdown_path.read_text(encoding="utf-8")
    raw_elements = json.loads(preindexed.elements_path.read_text(encoding="utf-8"))
    elements = [DocumentElement.model_validate(item) for item in raw_elements]
    # Inject pinned formula elements so formula text is retrievable
    # (the PDF→markdown converter strips equation images).
    elements.extend(_load_pinned_formula_elements())
    graph_store = GraphStore.load(preindexed.graph_path)

    return _build_corpus_session_from_elements(
        corpus_id=preindexed.corpus_id,
        filename=filename,
        file_size=file_size,
        md_text=md_text,
        page_count=page_count,
        is_spes1=False,
        backend=backend,
        embedder=embedder,
        elements=elements,
        graph_store=graph_store,
        warnings=[
            (
                "Recognized licensed ASME VIII-1 PDF by SHA-256 and loaded "
                "pre-indexed private artifacts; live PDF extraction skipped."
            )
        ],
        preindexed_source=preindexed.display_name,
    )


# ---------------------------------------------------------------------------
# Pinned formula injection
# ---------------------------------------------------------------------------

def _load_pinned_formula_elements() -> list[DocumentElement]:
    """Load pinned ASME formula definitions as synthetic DocumentElements.

    The PDF→markdown converter strips equation images, leaving 'picture
    intentionally omitted' placeholders. These pinned elements provide the
    verified formula text so the retriever can surface it.
    """
    if not _PINNED_FORMULAS_PATH.exists():
        return []
    raw = json.loads(_PINNED_FORMULAS_PATH.read_text(encoding="utf-8"))
    elements: list[DocumentElement] = []
    for f in raw:
        ref = f["paragraph_ref"]
        slug = re.sub(r"[^a-z0-9.]+", "-", ref.lower()).strip("-")
        variables_text = "\n".join(
            f"  {sym}: {desc}" for sym, desc in f["variables"].items()
        )
        content = (
            f"{f['title']}\n\n"
            f"Formula: {f['formula_text']}\n\n"
            f"Variables:\n{variables_text}\n\n"
            f"Applicability: {f['applicability']}\n\n"
            f"Source: {f['source']}"
        )
        elements.append(
            DocumentElement(
                element_id=f"pinned-formula-{slug}",
                element_type=ElementType.SECTION,
                content=content,
                page_number=1,
                paragraph_ref=ref.split("-alt")[0],  # strip alt suffix
                title=f"[Pinned] {f['title']}",
                parent_id=None,
                cross_refs=[],
            )
        )
    return elements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_backend_info() -> dict[str, Any]:
    backend_env = os.environ.get("ANVIL_LLM_BACKEND", "fake").lower().strip()
    model_env = os.environ.get("ANVIL_LLM_MODEL", "")
    embedder_env = os.environ.get("ANVIL_EMBEDDER", "hash").lower().strip()
    is_real = backend_env in ("nvidia_nim", "openai_compatible", "instructor")

    missing: list[str] = []
    if not is_real:
        missing.append("ANVIL_LLM_BACKEND must be nvidia_nim, openai_compatible, or instructor")
    if is_real and not model_env:
        missing.append("ANVIL_LLM_MODEL is required")
    if backend_env == "nvidia_nim" and not os.environ.get("NVIDIA_API_KEY"):
        missing.append("NVIDIA_API_KEY is required for nvidia_nim")
    if backend_env == "openai_compatible":
        if not os.environ.get("OPENAI_COMPAT_BASE_URL"):
            missing.append("OPENAI_COMPAT_BASE_URL is required")
        if not os.environ.get("OPENAI_COMPAT_API_KEY"):
            missing.append("OPENAI_COMPAT_API_KEY is required")

    return {
        "backend": backend_env,
        "model": model_env,
        "embedder": embedder_env,
        "real_backend_configured": is_real and len(missing) == 0,
        "fake_backend_allowed": False,
        "missing": missing,
    }


def _get_demo_backend() -> LLMBackend:
    """Return a backend for corpus construction; query execution still gates real use."""
    info = _get_backend_info()
    if not info["real_backend_configured"]:
        return FakeLLMBackend()
    return get_default_backend()


def _clean_demo_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"<br\s*/?>", " ", value)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    cleaned = re.sub(r"[^\w()./%° -]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned


def _topic_from_element(e: DocumentElement) -> str:
    if e.table and e.table.caption:
        raw = _clean_demo_text(e.table.caption)
    else:
        raw = _clean_demo_text(e.title)
    # Strip leading paragraph ref to avoid repetition in queries like
    # "What does UG-32 specify about UG-32 Formed Heads..."
    if e.paragraph_ref and raw.upper().startswith(e.paragraph_ref.upper()):
        raw = raw[len(e.paragraph_ref):].lstrip(" .-:")
    # Title-case if all-uppercase (common in PDF headings)
    if raw and raw == raw.upper():
        raw = raw.title()
    return raw


def _domain_weight(e: DocumentElement) -> int:
    text = f"{e.paragraph_ref or ''} {_topic_from_element(e)} {e.content[:300]}".lower()
    weights = {
        "thickness": 80,
        "joint efficien": 90,
        "allowable stress": 65,
        "hydrostatic": 60,
        "nozzle": 45,
        "impact": 35,
        "pressure": 30,
        "weld": 30,
        "material": 25,
    }
    return max((weight for token, weight in weights.items() if token in text), default=0)


def _table_quality(e: DocumentElement) -> int:
    if e.element_type != ElementType.TABLE or e.table is None:
        return -1
    table_id = e.table.table_id
    if table_id.startswith("TBL-"):
        return 0
    score = 50
    if e.paragraph_ref:
        score += 15
    if e.table.caption:
        score += 20
    if e.table.headers and any(_clean_demo_text(h) for h in e.table.headers):
        score += 10
    if e.table.rows:
        score += min(len(e.table.rows), 20)
    return score + _domain_weight(e)


def _section_quality(e: DocumentElement) -> int:
    if e.element_type != ElementType.SECTION or not e.paragraph_ref:
        return -1
    title = _clean_demo_text(e.title)
    if not title:
        return -1
    score = 40 + _domain_weight(e)
    if e.cross_references:
        score += min(len(e.cross_references), 20)
    if len(e.content.strip()) > 300:
        score += 10
    return score


def _unique_suggestion(
    suggestions: list[dict[str, str]],
    *,
    query: str,
    source_element_id: str,
    paragraph_ref: str,
    reason: str,
    label: str = "Generated from uploaded PDF metadata",
) -> None:
    if any(s["query"] == query or s["source_element_id"] == source_element_id for s in suggestions):
        return
    suggestions.append(
        {
            "query": query,
            "source_element_id": source_element_id,
            "paragraph_ref": paragraph_ref,
            "reason": reason,
            "label": label,
        }
    )


def _generate_suggestions(elements: list[DocumentElement]) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []

    sections = sorted(elements, key=_section_quality, reverse=True)
    tables = sorted(elements, key=_table_quality, reverse=True)

    used_topics: set[str] = set()
    for e in tables:
        if len(suggestions) >= 3 or _table_quality(e) <= 0 or e.table is None:
            break
        topic = _topic_from_element(e)
        if not topic:
            continue
        topic_key = topic.split(" for ")[0]
        if topic_key in used_topics:
            continue
        used_topics.add(topic_key)
        _unique_suggestion(
            suggestions,
            query=f"What information is contained in Table {e.table.table_id} about {topic}?",
            source_element_id=e.element_id,
            paragraph_ref=e.paragraph_ref or "",
            reason="generated from high-signal parsed table metadata",
        )

    for e in sections:
        if len(suggestions) >= 3 or _section_quality(e) <= 0:
            break
        topic = _topic_from_element(e) or "its requirements"
        _unique_suggestion(
            suggestions,
            query=f"What does {e.paragraph_ref} specify about {topic}?",
            source_element_id=e.element_id,
            paragraph_ref=e.paragraph_ref or "",
            reason="generated from high-signal parsed section metadata",
        )

    xref_sections = sorted(
        [e for e in elements if e.element_type == ElementType.SECTION and e.cross_references],
        key=lambda e: _section_quality(e),
        reverse=True,
    )
    for e in xref_sections:
        if len(suggestions) >= 3:
            break
        label = e.paragraph_ref or _clean_demo_text(e.title)
        _unique_suggestion(
            suggestions,
            query=f"Which sections or tables does {label} refer to?",
            source_element_id=e.element_id,
            paragraph_ref=e.paragraph_ref or "",
            reason="generated from cross-reference-heavy parsed section metadata",
        )

    if len(suggestions) < 3:
        for e in elements:
            if len(suggestions) >= 3:
                break
            if e.element_type == ElementType.SECTION and e.paragraph_ref:
                _unique_suggestion(
                    suggestions,
                    query=f"What are the requirements specified in {e.paragraph_ref}?",
                    source_element_id=e.element_id,
                    paragraph_ref=e.paragraph_ref or "",
                    reason="generated from parsed section metadata",
                )

    _unique_suggestion(
        suggestions,
        query="What is the weather in San Jose today?",
        source_element_id="",
        paragraph_ref="",
        reason="irrelevant out-of-domain query for refusal demo",
        label="Refusal demo",
    )

    return suggestions


_SPES1_SUGGESTIONS: list[dict[str, str]] = [
    {
        "query": (
            "Calculate the minimum required wall thickness for a cylindrical "
            "shell with inside diameter 1800 mm, P=1.5 MPa, T=350\u00b0C, "
            "SM-516 Gr 70, Type 1 with full RT, CA=3.0 mm."
        ),
        "label": "Pinned calculation demo",
        "source_element_id": "",
        "paragraph_ref": "A-27",
        "reason": "SPES-1 pinned calculation support",
    },
    {
        "query": "What is the allowable stress of SM-516 Gr 70 at 300\u00b0C?",
        "label": "Pinned calculation demo",
        "source_element_id": "",
        "paragraph_ref": "M-1",
        "reason": "SPES-1 pinned material lookup",
    },
    {
        "query": (
            "What inputs from A-27, Table B-12, and Table M-1 are required "
            "to apply the cylindrical shell thickness formula?"
        ),
        "label": "Pinned calculation demo",
        "source_element_id": "",
        "paragraph_ref": "A-27",
        "reason": "SPES-1 cross-reference explanation",
    },
    {
        "query": "What is the weather in San Jose today?",
        "label": "Refusal demo",
        "source_element_id": "",
        "paragraph_ref": "",
        "reason": "irrelevant out-of-domain query for refusal demo",
    },
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    embedder: str
    real_backend_configured: bool
    fake_backend_allowed: bool = False
    missing: list[str]
    corpora_loaded: int


class UploadResponse(BaseModel):
    corpus_id: str
    stats: dict[str, Any]
    suggestions: list[dict[str, str]]


class DemoQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    corpus_id: str | None = None


class DemoQueryResponse(BaseModel):
    answer: str
    confidence: str
    refusal_reason: str | None
    retrieved_element_ids: list[str]
    citation_validation: dict[str, Any]
    citations: list[dict[str, Any]]
    calculation_steps: list[dict[str, Any]]
    latency_ms: float
    backend: str
    model: str
    embedder: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_demo_router() -> APIRouter:
    router = APIRouter(prefix="/demo")

    def _get_session(corpus_id: str | None) -> CorpusSession:
        if corpus_id and corpus_id in _registry:
            return _registry[corpus_id]
        if not corpus_id and _registry:
            return list(_registry.values())[-1]
        if corpus_id:
            raise HTTPException(404, f"Corpus '{corpus_id}' not found.")
        raise HTTPException(404, "No corpus loaded. Load SPES-1 or upload a PDF first.")

    # -- health --------------------------------------------------------------

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        info = _get_backend_info()
        return HealthResponse(
            status="ok" if info["real_backend_configured"] else "degraded",
            backend=info["backend"],
            model=info["model"],
            embedder=info["embedder"],
            real_backend_configured=info["real_backend_configured"],
            fake_backend_allowed=False,
            missing=info["missing"],
            corpora_loaded=len(_registry),
        )

    # -- corpus management ---------------------------------------------------

    @router.post("/load-spes1")
    def load_spes1() -> dict[str, Any]:
        for s in _registry.values():
            if s.is_spes1:
                return {"corpus_id": s.corpus_id, "stats": s.stats, "already_loaded": True}
        if not _SPES1_PATH.exists():
            raise HTTPException(500, f"SPES-1 standard not found at {_SPES1_PATH}")
        md_text = _SPES1_PATH.read_text(encoding="utf-8")
        embedder = get_default_embedder()
        backend = _get_demo_backend()
        session = _build_corpus_session(
            corpus_id="spes1",
            filename="standard.md",
            file_size=_SPES1_PATH.stat().st_size,
            md_text=md_text,
            page_count=None,
            is_spes1=True,
            backend=backend,
            embedder=embedder,
        )
        _registry["spes1"] = session
        return {"corpus_id": "spes1", "stats": session.stats, "already_loaded": False}

    @router.post("/upload")
    async def upload_pdf(file: UploadFile) -> UploadResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are accepted.")

        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        file_size = len(content)
        content_sha256 = _sha256_bytes(content)

        safe_name = file.filename.replace("/", "_").replace("\\", "_")
        pdf_path = _UPLOADS_DIR / safe_name
        pdf_path.write_bytes(content)

        page_count: int | None = None
        try:
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
        except Exception:
            pass

        embedder = get_default_embedder()
        backend = _get_demo_backend()
        preindexed = _PREINDEXED_BY_SHA.get(content_sha256)

        if preindexed is not None:
            if preindexed.corpus_id in _registry:
                session = _registry[preindexed.corpus_id]
                session.stats["filename"] = safe_name
                session.stats["file_size"] = file_size
                session.stats["page_count"] = page_count
                session.stats["already_loaded"] = True
                return UploadResponse(
                    corpus_id=session.corpus_id,
                    stats=session.stats,
                    suggestions=_generate_suggestions(session.elements),
                )

            session = _load_preindexed_session(
                preindexed=preindexed,
                filename=safe_name,
                file_size=file_size,
                page_count=page_count,
                backend=backend,
                embedder=embedder,
            )
            session.stats["already_loaded"] = False
            _registry[preindexed.corpus_id] = session
            return UploadResponse(
                corpus_id=session.corpus_id,
                stats=session.stats,
                suggestions=_generate_suggestions(session.elements),
            )

        try:
            import pymupdf4llm
        except ImportError as exc:
            raise HTTPException(
                500, "pymupdf4llm is required for PDF parsing. Install with `uv add pymupdf4llm`."
            ) from exc

        md_text: str = pymupdf4llm.to_markdown(str(pdf_path))

        md_path = _UPLOADS_DIR / f"{safe_name}.md"
        md_path.write_text(md_text, encoding="utf-8")

        corpus_id = str(uuid.uuid4())[:8]
        session = _build_corpus_session(
            corpus_id=corpus_id,
            filename=safe_name,
            file_size=file_size,
            md_text=md_text,
            page_count=page_count,
            is_spes1=False,
            backend=backend,
            embedder=embedder,
        )
        _registry[corpus_id] = session
        suggestions = _generate_suggestions(session.elements)

        return UploadResponse(corpus_id=corpus_id, stats=session.stats, suggestions=suggestions)

    @router.get("/corpora")
    def list_corpora() -> list[dict[str, Any]]:
        return [
            {
                "corpus_id": s.corpus_id,
                "filename": s.filename,
                "is_spes1": s.is_spes1,
                "num_elements": len(s.elements),
                "created_at": s.created_at,
            }
            for s in _registry.values()
        ]

    # -- corpus inspection ---------------------------------------------------

    @router.get("/corpus/stats")
    def corpus_stats(corpus_id: str | None = None) -> dict[str, Any]:
        return _get_session(corpus_id).stats

    @router.get("/corpus/elements")
    def corpus_elements(
        corpus_id: str | None = None,
        type: str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        session = _get_session(corpus_id)
        filtered = session.elements
        if type:
            filtered = [e for e in filtered if e.element_type.value == type]
        if q:
            ql = q.lower()
            filtered = [
                e for e in filtered
                if ql in (e.content or "").lower()
                or ql in (e.title or "").lower()
                or ql in (e.paragraph_ref or "").lower()
            ]
        return [
            {
                "element_id": e.element_id,
                "element_type": e.element_type.value,
                "paragraph_ref": e.paragraph_ref,
                "title": e.title,
                "content": e.content[:500],
                "page_number": e.page_number,
                "parent_id": e.parent_id,
                "num_cross_refs": len(e.cross_references),
                "has_table": e.table is not None,
                "has_formula": e.formula is not None,
            }
            for e in filtered[:limit]
        ]

    @router.get("/corpus/tables")
    def corpus_tables(corpus_id: str | None = None) -> list[dict[str, Any]]:
        session = _get_session(corpus_id)
        out: list[dict[str, Any]] = []
        table_elements = sorted(
            [e for e in session.elements if e.element_type == ElementType.TABLE],
            key=_table_quality,
            reverse=True,
        )
        for e in table_elements:
            if e.element_type == ElementType.TABLE and e.table:
                out.append({
                    "element_id": e.element_id,
                    "table_id": e.table.table_id,
                    "caption": e.table.caption,
                    "paragraph_ref": e.paragraph_ref,
                    "headers": e.table.headers,
                    "rows": [[cell.text for cell in row] for row in e.table.rows],
                    "source_page": e.table.source_page,
                })
        return out

    @router.get("/corpus/graph")
    def corpus_graph(corpus_id: str | None = None, limit: int = 200) -> dict[str, Any]:
        session = _get_session(corpus_id)
        g = session.graph_store.graph
        edges = [
            {
                "source": src,
                "target": tgt,
                "edge_type": data.get("edge_type", ""),
                "reference_text": data.get("reference_text", ""),
            }
            for src, tgt, data in list(g.edges(data=True))[:limit]
        ]
        return {
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "edges": edges,
        }

    @router.get("/corpus/suggestions")
    def corpus_suggestions(corpus_id: str | None = None) -> list[dict[str, str]]:
        session = _get_session(corpus_id)
        if session.is_spes1:
            return _SPES1_SUGGESTIONS
        return _generate_suggestions(session.elements)

    # -- pinned data ---------------------------------------------------------

    @router.get("/pinned/materials")
    def pinned_materials() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in list_materials():
            mat = get_material(key)
            if mat:
                out.append({
                    "key": key,
                    "spec_no": mat.spec_no,
                    "grade": mat.grade,
                    "product_form": mat.product_form,
                    "nominal_composition": mat.nominal_composition,
                    "min_tensile_mpa": mat.min_tensile_mpa,
                    "min_yield_mpa": mat.min_yield_mpa,
                    "max_temp_c": mat.max_temp_c,
                    "notes": mat.notes,
                })
        return out

    @router.get("/pinned/materials/{spec_grade:path}")
    def pinned_material_detail(spec_grade: str) -> dict[str, Any]:
        mat = get_material(spec_grade)
        if mat is None:
            raise HTTPException(404, f"Material '{spec_grade}' not found in pinned data.")
        return {
            "key": mat.key,
            "spec_no": mat.spec_no,
            "grade": mat.grade,
            "product_form": mat.product_form,
            "nominal_composition": mat.nominal_composition,
            "p_no": mat.p_no,
            "group_no": mat.group_no,
            "min_tensile_mpa": mat.min_tensile_mpa,
            "min_yield_mpa": mat.min_yield_mpa,
            "stress_by_temp_c": {str(k): v for k, v in mat.stress_by_temp_c.items()},
            "max_temp_c": mat.max_temp_c,
            "notes": mat.notes,
        }

    @router.get("/pinned/joint-efficiencies")
    def pinned_joint_efficiencies() -> dict[str, Any]:
        rows = [
            {"type": jtype, "efficiencies": effs}
            for jtype, effs in sorted(JOINT_EFFICIENCIES.items())
        ]
        return {"rt_levels": list(RT_LEVELS), "rows": rows}

    # -- query ---------------------------------------------------------------

    @router.post("/query", response_model=DemoQueryResponse)
    async def demo_query(req: DemoQueryRequest) -> DemoQueryResponse:
        info = _get_backend_info()
        session = _get_session(req.corpus_id)

        if not info["real_backend_configured"]:
            raise HTTPException(
                503,
                "Real backend not configured. "
                + " ".join(info["missing"])
                + " FakeLLMBackend is not permitted for demo queries.",
            )
        if isinstance(session.generator.backend, FakeLLMBackend):
            raise HTTPException(
                503,
                "Real backend not configured. Set ANVIL_LLM_BACKEND to "
                "nvidia_nim, openai_compatible, or instructor with the "
                "required API keys. FakeLLMBackend is not permitted for "
                "demo queries.",
            )

        start = time.perf_counter()
        outcome = await session.generator.generate(req.query, top_k=req.top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        return DemoQueryResponse(
            answer=outcome.response.answer,
            confidence=outcome.response.confidence.value,
            refusal_reason=outcome.response.refusal_reason,
            retrieved_element_ids=[c.element_id for c in outcome.retrieved_chunks],
            citation_validation={
                "total": outcome.citation_validation.total,
                "valid": outcome.citation_validation.valid,
                "accuracy": outcome.citation_validation.accuracy,
                "issues": [
                    {"index": i.citation_index, "issue": i.issue}
                    for i in outcome.citation_validation.issues
                ],
            },
            citations=[
                {
                    "source_element_id": c.source_element_id,
                    "paragraph_ref": c.paragraph_ref,
                    "quoted_text": c.quoted_text,
                    "page_number": c.page_number,
                }
                for c in outcome.response.citations
            ],
            calculation_steps=[
                {
                    "step_number": s.step_number,
                    "result_key": s.result_key.value,
                    "description": s.description,
                    "formula": s.formula,
                    "result": s.result,
                    "unit": s.unit,
                    "inputs": {
                        k: {"symbol": v.symbol, "value": v.value, "unit": v.unit, "source": v.source}
                        for k, v in s.inputs.items()
                    },
                }
                for s in outcome.response.calculation_steps
            ],
            latency_ms=round(latency_ms, 1),
            backend=info["backend"],
            model=info["model"],
            embedder=info["embedder"],
        )

    return router
