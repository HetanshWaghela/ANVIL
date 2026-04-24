"""Phase 4 tests: citation enforcement."""

from __future__ import annotations

from anvil.generation.citation_enforcer import validate_citations
from anvil.schemas.generation import AnvilResponse, Citation, ResponseConfidence
from anvil.schemas.retrieval import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        element_id="sec-a-27",
        paragraph_ref="A-27",
        element_type="section",
        content="The cylindrical shell thickness formula is t = (P × R) / (S × E − 0.6 × P).",
        page_number=1,
        score=0.9,
    )


def test_citation_valid() -> None:
    chunk = _chunk()
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="sec-a-27",
                paragraph_ref="A-27",
                quoted_text="cylindrical shell thickness formula",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [chunk])
    assert result.passed
    assert result.accuracy == 1.0


def test_citation_detects_paragraph_mismatch() -> None:
    chunk = _chunk()
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="sec-a-27",
                paragraph_ref="B-12",  # wrong para
                quoted_text="cylindrical shell thickness",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [chunk])
    assert not result.passed
    assert "paragraph_ref mismatch" in result.issues[0].issue


def test_citation_detects_fabricated_quote() -> None:
    chunk = _chunk()
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="sec-a-27",
                paragraph_ref="A-27",
                quoted_text="quantum chromodynamics of pressure vessels",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [chunk])
    assert not result.passed


def test_citation_accepts_canonical_ref_without_retrieved_chunk() -> None:
    """Pinned-data citations to canonical SPES-1 refs are accepted even when
    the RAG retriever did not include the source element."""
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="tbl-m-1",  # not in retrieved context
                paragraph_ref="Table M-1",
                quoted_text="SM-516 Gr 70 at 350°C",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    # No retrieved chunks
    result = validate_citations(resp, [])
    assert result.passed
