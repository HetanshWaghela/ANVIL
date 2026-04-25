"""Phase 4 tests: citation enforcement."""

from __future__ import annotations

from anvil.generation.citation_enforcer import validate_citations
from anvil.schemas.document import DocumentElement, ElementType
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


def _canonical_index() -> dict[str, DocumentElement]:
    """Minimal element_index covering the canonical SPES-1 elements.

    These are stand-ins that exercise the canonical-ref validation path
    without parsing the whole standard. The `content` strings hold the
    sentinel text the tests below quote (or attempt to quote).
    """
    elements = [
        DocumentElement(
            element_id="sec-a-27",
            element_type=ElementType.SECTION,
            paragraph_ref="A-27",
            title="A-27 Thickness of Shells Under Internal Pressure",
            content=(
                "The minimum required thickness of shells subjected to "
                "internal pressure shall not be less than that computed "
                "by t = (P × R) / (S × E − 0.6 × P)."
            ),
            page_number=1,
        ),
        DocumentElement(
            element_id="tbl-m-1",
            element_type=ElementType.TABLE,
            paragraph_ref="M-1",
            title="Table M-1",
            content="| SM-516 | Gr 70 | Plate | 138 | 138 | 138 | 134 | 127 | 121 | 114 |",
            page_number=4,
        ),
    ]
    return {e.element_id: e for e in elements}


def test_canonical_ref_branch_rejects_fabricated_quote() -> None:
    """REGRESSION (audit A1): a citation to a canonical SPES-1 ref whose
    `quoted_text` does NOT appear in the actual standard element MUST be
    rejected even when the source_element_id isn't in retrieved context.

    The previous implementation accepted any quote on the canonical-ref
    branch, which is precisely the hallucination vector this system exists
    to prevent.
    """
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="tbl-m-1",
                paragraph_ref="Table M-1",
                quoted_text="quantum chromodynamics of pressure vessels",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [], element_index=_canonical_index())
    assert not result.passed
    assert any("not present in canonical element" in i.issue for i in result.issues)


def test_canonical_ref_branch_accepts_real_quote_from_standard() -> None:
    """A canonical-ref citation whose quote IS present in the resolved
    element passes — this is the legitimate pinned-data path."""
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="tbl-m-1",
                paragraph_ref="Table M-1",
                quoted_text="SM-516 | Gr 70 | Plate",
                page_number=4,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [], element_index=_canonical_index())
    assert result.passed


def test_canonical_ref_branch_fails_closed_without_element_index() -> None:
    """REGRESSION (audit A1): with no `element_index` we cannot validate
    `quoted_text`, so the citation MUST be flagged. The old behavior
    silently accepted such citations and was a hallucination escape hatch.
    """
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="tbl-m-1",
                paragraph_ref="Table M-1",
                quoted_text="any text whatsoever",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [])
    assert not result.passed
    assert any("element_index" in i.issue for i in result.issues)


def test_canonical_ref_branch_walks_up_subparagraphs() -> None:
    """`A-27(c)(1)` should resolve to `A-27` when no sub-paragraph
    element exists — same boundary rule as the CitationBuilder."""
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[
            Citation(
                source_element_id="fml-a-27-f0",  # not in retrieved
                paragraph_ref="A-27(c)(1)",
                quoted_text="t = (P × R) / (S × E − 0.6 × P)",
                page_number=1,
            )
        ],
        confidence=ResponseConfidence.HIGH,
    )
    result = validate_citations(resp, [], element_index=_canonical_index())
    assert result.passed
