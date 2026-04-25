"""Phase 4 tests: refusal gate."""

from __future__ import annotations

from anvil.generation.refusal_gate import (
    check_calculation_evidence,
    is_calculation_query,
    should_refuse,
)
from anvil.schemas.retrieval import RetrievedChunk


def _chunk(
    score: float,
    content: str = "A-27 thickness formula",
    paragraph_ref: str = "A-27",
    element_id: str = "x",
) -> RetrievedChunk:
    return RetrievedChunk(
        element_id=element_id,
        paragraph_ref=paragraph_ref,
        element_type="section",
        content=content,
        page_number=1,
        score=score,
    )


def _calc_context() -> list[RetrievedChunk]:
    """The minimum retrieved-context a calculation query needs to satisfy
    the required-elements refusal check: A-27 + M-1 + B-12."""
    return [
        _chunk(0.9, paragraph_ref="A-27", element_id="sec-a-27"),
        _chunk(0.8, paragraph_ref="M-1", element_id="sec-m-1"),
        _chunk(0.8, paragraph_ref="B-12", element_id="sec-b-12"),
    ]


def test_is_calculation_query_positive() -> None:
    assert is_calculation_query("calculate the minimum wall thickness")
    assert is_calculation_query("compute MAWP")


def test_is_calculation_query_negative() -> None:
    assert not is_calculation_query("what is the weather today")


def test_refuses_on_empty_chunks() -> None:
    d = should_refuse("thickness", [])
    assert d.should_refuse
    assert "relevant" in (d.reason or "").lower()


def test_refuses_on_low_relevance() -> None:
    d = should_refuse("thickness", [_chunk(0.001)])
    assert d.should_refuse


def test_refuses_on_unknown_material() -> None:
    d = should_refuse(
        "compute thickness for SM-999 Gr XYZ at 300°C",
        [_chunk(0.8, content="SM-999 is not in our tables")],
    )
    assert d.should_refuse
    assert "SM-999" in (d.reason or "")


def test_refuses_on_over_max_temp() -> None:
    d = should_refuse(
        "compute thickness for SM-516 Gr 70 at 700°C",
        [_chunk(0.8)],
    )
    assert d.should_refuse
    assert "700" in (d.reason or "")


def test_refuses_incomplete_calculation_query() -> None:
    """Free-text calculations must include every required design input."""
    d = should_refuse(
        "compute thickness for SM-516 Gr 70 at 350°C",
        _calc_context(),
    )
    assert d.should_refuse
    assert "missing required input" in (d.reason or "")


def test_accepts_complete_in_domain_query() -> None:
    """In-domain calc query with full context (A-27 + M-1 + B-12) → no refusal."""
    d = should_refuse(
        "compute thickness for ID=1800 mm, P=1.5 MPa, SM-516 Gr 70 at 350°C, "
        "Type 1 Full RT",
        _calc_context(),
    )
    assert not d.should_refuse


def test_evidence_check_passes_with_full_context() -> None:
    """Full A-27 + M-1 + B-12 retrieval → evidence check passes."""
    d = check_calculation_evidence(_calc_context())
    assert not d.should_refuse


def test_evidence_check_refuses_missing_stress_table() -> None:
    """Spec §Refusal Gate: refuse if the stress-table evidence is missing
    from retrieval even when the formula and joint table are present."""
    d = check_calculation_evidence(
        [
            _chunk(0.9, paragraph_ref="A-27", element_id="sec-a-27"),
            _chunk(0.8, paragraph_ref="B-12", element_id="sec-b-12"),
        ]
    )
    assert d.should_refuse
    assert "M-1" in (d.reason or "")


def test_evidence_check_refuses_missing_joint_table() -> None:
    d = check_calculation_evidence(
        [
            _chunk(0.9, paragraph_ref="A-27", element_id="sec-a-27"),
            _chunk(0.8, paragraph_ref="M-1", element_id="sec-m-1"),
        ]
    )
    assert d.should_refuse
    assert "B-12" in (d.reason or "")


def test_evidence_check_refuses_missing_formula() -> None:
    d = check_calculation_evidence(
        [
            _chunk(0.9, paragraph_ref="M-1", element_id="sec-m-1"),
            _chunk(0.8, paragraph_ref="B-12", element_id="sec-b-12"),
        ]
    )
    assert d.should_refuse
    assert "A-27" in (d.reason or "")


def test_evidence_check_accepts_subparagraph_formula() -> None:
    """A-27(c)(1) should count as covering 'A-27' (sub-paragraph boundary)."""
    d = check_calculation_evidence(
        [
            _chunk(0.9, paragraph_ref="A-27(c)(1)", element_id="fml-a-27-f0"),
            _chunk(0.8, paragraph_ref="M-1", element_id="sec-m-1"),
            _chunk(0.8, paragraph_ref="B-12", element_id="sec-b-12"),
        ]
    )
    assert not d.should_refuse
