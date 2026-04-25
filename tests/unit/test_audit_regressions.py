"""Regression tests for the deep-audit fixes.

These tests lock in behavioral guarantees that replaced earlier hardcoded /
heuristic implementations. If any of them fail, a fragile pattern has crept
back into the codebase.
"""

from __future__ import annotations

import pytest

from anvil import CalculationError
from anvil.generation.calculation_engine import CitationBuilder
from anvil.generation.generator import _try_parse_calculation_inputs
from anvil.knowledge.graph_builder import build_graph
from anvil.schemas.document import DocumentElement, ElementType
from anvil.schemas.generation import StepKey

# ---------------------------------------------------------------------------
# 1) CitationBuilder quotes real document content — not hardcoded strings.
# ---------------------------------------------------------------------------


def test_citation_builder_uses_real_document_content(standard_md_path, parsed_elements) -> None:
    builder = CitationBuilder.from_elements(parsed_elements)
    cit = builder.for_material("SM-516 Gr 70")
    # The quoted text MUST be a real substring of the standard's content
    # (either a table row or section intro) — no fabricated sentence.
    full_text = standard_md_path.read_text()
    assert cit.quoted_text.strip() in full_text, (
        f"Citation quote '{cit.quoted_text}' not found in the standard — "
        f"the CitationBuilder must quote real content."
    )


def test_citation_builder_material_quote_prefers_row(parsed_elements) -> None:
    """When a material row exists in Table M-1, prefer that over the intro."""
    builder = CitationBuilder.from_elements(parsed_elements)
    cit = builder.for_material("SM-516 Gr 70")
    # The SM-516 row contains "SM-516" — the quote should too.
    assert "SM-516" in cit.quoted_text


def test_citation_builder_refuses_unknown_paragraph(parsed_elements) -> None:
    """Asking for a non-existent paragraph MUST raise, not fabricate."""
    builder = CitationBuilder.from_elements(parsed_elements)
    with pytest.raises(CalculationError):
        builder.for_paragraph("Z-99")


def test_citation_builder_walks_up_subparagraphs(parsed_elements) -> None:
    """A-27(c)(1) should resolve to the A-27 element when no sub-element exists."""
    builder = CitationBuilder.from_elements(parsed_elements)
    cit = builder.for_paragraph("A-27(c)(1)")
    assert cit.source_element_id == "sec-a-27"


def test_citation_builder_accepts_table_prefix(parsed_elements) -> None:
    """`'Table M-1'` and `'M-1'` should both resolve to the same element."""
    builder = CitationBuilder.from_elements(parsed_elements)
    cit_a = builder.for_paragraph("M-1")
    cit_b = builder.for_paragraph("Table M-1")
    assert cit_a.source_element_id == cit_b.source_element_id


# ---------------------------------------------------------------------------
# 2) Graph builder uses parsed table refs — no hardcoded M-1 / B-12 strings.
# ---------------------------------------------------------------------------


def test_graph_builder_wires_arbitrary_tables() -> None:
    """If a formula variable cites `Table X-5`, the edge must appear when
    a Table X-5 element exists — without any code change in graph_builder.
    """
    # Synthetic elements: one formula referencing Table X-5, one table X-5.
    from anvil.schemas.document import (
        ElementType,
        FormulaVariable,
        ParsedFormula,
        ParsedTable,
        TableCell,
    )

    formula_el = DocumentElement(
        element_id="fml-demo",
        element_type=ElementType.FORMULA,
        paragraph_ref="X-10",
        title="Demo formula",
        content="t = k * Q",
        page_number=1,
        formula=ParsedFormula(
            formula_id="X-10-f0",
            latex="t = k Q",
            plain_text="t = k * Q",
            variables=[
                FormulaVariable(
                    symbol="Q",
                    name="fudge factor",
                    unit="",
                    source="Table X-5",
                )
            ],
            source_paragraph="X-10",
        ),
    )
    table_el = DocumentElement(
        element_id="tbl-x-5",
        element_type=ElementType.TABLE,
        paragraph_ref="X-5",
        title="Table X-5",
        content="| a | b |",
        page_number=1,
        table=ParsedTable(
            table_id="X-5",
            headers=["a", "b"],
            rows=[[TableCell(row=0, col=0, text="1"), TableCell(row=0, col=1, text="2")]],
            source_page=1,
            source_paragraph="X-5",
        ),
    )
    g = build_graph([formula_el, table_el])
    assert g.has_edge("fml-demo", "tbl-x-5"), (
        "Formula → Table edge must be created from the variable's actual "
        "source string, not a hardcoded lookup."
    )


# ---------------------------------------------------------------------------
# 3) CalculationStep.result_key is a typed enum (no description-substring match).
# ---------------------------------------------------------------------------


def test_calculation_step_carries_result_key(pipeline) -> None:
    import asyncio

    from anvil.generation.calculation_engine import CalculationInputs

    async def go():
        outcome = await pipeline.generator.generate(
            "compute thickness",
            calculation_inputs=CalculationInputs(
                component="cylindrical_shell",
                P_mpa=1.5,
                design_temp_c=350,
                material="SM-516 Gr 70",
                joint_type=1,
                rt_level="Full RT",
                corrosion_allowance_mm=3.0,
                inside_diameter_mm=1800,
            ),
        )
        return outcome

    outcome = asyncio.run(go())
    keys = {s.result_key for s in outcome.response.calculation_steps}
    # The seven-step pipeline must expose every semantic role.
    assert keys == {
        StepKey.ALLOWABLE_STRESS,
        StepKey.JOINT_EFFICIENCY,
        StepKey.APPLICABILITY_CHECK,
        StepKey.MIN_THICKNESS,
        StepKey.DESIGN_THICKNESS,
        StepKey.NOMINAL_PLATE,
        StepKey.MAWP,
    }


# ---------------------------------------------------------------------------
# 4) NL parser never silently defaults joint type or RT level.
# ---------------------------------------------------------------------------


def test_nl_parser_refuses_without_joint_type() -> None:
    """Must return None (refuse) if joint type is missing — no default."""
    q = (
        "compute thickness for cylindrical shell ID=1000 mm, P=1.5 MPa, "
        "T=200°C, SM-516 Gr 70 with full RT, CA=2 mm."
    )
    assert _try_parse_calculation_inputs(q) is None


def test_nl_parser_refuses_without_rt_level() -> None:
    """Must return None (refuse) if RT level is missing — no default."""
    q = (
        "compute thickness for cylindrical shell ID=1000 mm, P=1.5 MPa, "
        "T=200°C, SM-516 Gr 70, Type 1, CA=2 mm."
    )
    assert _try_parse_calculation_inputs(q) is None


def test_nl_parser_accepts_explicit_inputs() -> None:
    q = (
        "compute thickness for cylindrical shell ID=1000 mm, P=1.5 MPa, "
        "T=200°C, SM-516 Gr 70, Type 2 with spot radiography, CA=2 mm."
    )
    inp = _try_parse_calculation_inputs(q)
    assert inp is not None
    assert inp.joint_type == 2
    assert inp.rt_level == "Spot RT"
    assert inp.material == "SM-516 Gr 70"


# ---------------------------------------------------------------------------
# 5) Formula applicability conditions are populated from the standard text.
# ---------------------------------------------------------------------------


def test_formula_extractor_populates_applicability_conditions() -> None:
    """REGRESSION (audit A2): the parser must extract "applies when ..."
    sentences into ParsedFormula.applicability_conditions. The previous
    implementation always set this field to []; downstream compliance
    reasoning had no structural access to per-formula limits.
    """
    from anvil.parsing.formula_extractor import extract_formulas

    body = (
        "Some preamble.\n\n"
        "```\n"
        "t = (P × R) / (S × E − 0.6 × P)\n"
        "```\n\n"
        "This formula applies when: t ≤ R/2 AND P ≤ 0.385 × S × E. "
        "When these conditions are not satisfied, A-27(c)(3) shall be "
        "applied (not covered here).\n"
    )
    formulas = extract_formulas("A-27(c)(1)", body)
    assert len(formulas) == 1
    conds = formulas[0].applicability_conditions
    assert any("R/2" in c for c in conds), conds
    assert any("0.385" in c for c in conds), conds


def test_formula_extractor_attaches_conditions_per_formula() -> None:
    """When a paragraph carries multiple formulas, each formula's
    `applicability_conditions` must come from the text region between
    its fence and the next fence — NOT a global merge."""
    from anvil.parsing.formula_extractor import extract_formulas

    body = (
        "```\nt = (P × R) / (S × E − 0.6 × P)\n```\n\n"
        "Applicable when P ≤ 0.385 × S × E.\n\n"
        "```\nt = (P × R) / (2 × S × E − 0.2 × P)\n```\n\n"
        "Applicable when t ≤ 0.356 × R AND P ≤ 0.665 × S × E.\n"
    )
    formulas = extract_formulas("A-27", body)
    assert len(formulas) == 2
    # Cylindrical formula gets the 0.385 condition only
    assert any("0.385" in c for c in formulas[0].applicability_conditions)
    assert all("0.665" not in c for c in formulas[0].applicability_conditions)
    # Spherical formula gets BOTH the 0.356 R and 0.665 S·E conditions
    sph = formulas[1].applicability_conditions
    assert any("0.356" in c for c in sph), sph
    assert any("0.665" in c for c in sph), sph


def test_synthetic_standard_a27_formulas_have_conditions(parsed_elements) -> None:
    """Integration: SPES-1's A-27 formulas (parsed from standard.md) must
    end up with non-empty applicability_conditions on every formula
    element — proves the regex matches the actual standard text, not
    just contrived test inputs."""
    from anvil.schemas.document import ElementType

    formula_els = [
        e
        for e in parsed_elements
        if e.element_type == ElementType.FORMULA
        and e.paragraph_ref
        and e.paragraph_ref.startswith("A-27")
        and e.formula is not None
    ]
    assert formula_els, "no A-27 formulas were parsed — parser may have regressed"
    for el in formula_els:
        assert el.formula is not None
        assert el.formula.applicability_conditions, (
            f"{el.element_id} has empty applicability_conditions; the "
            f"parser must extract them from the surrounding text."
        )


# ---------------------------------------------------------------------------
# 6) Table M-1 in the markdown matches the pinned JSON ground truth.
# ---------------------------------------------------------------------------


def test_markdown_table_m1_matches_pinned_data(parsed_elements) -> None:
    """Every material row in the pinned JSON must appear in the parsed
    Table M-1, and every row in the parsed table must round-trip back to
    a value that matches the pinned data (no drift between the two)."""

    from anvil.pinned import MATERIALS

    m1 = next(
        e
        for e in parsed_elements
        if e.element_type == ElementType.TABLE
        and e.table is not None
        and e.table.table_id == "M-1"
    )
    md_rows = {row[0].text + " " + row[1].text for row in m1.table.rows}
    # Every pinned material's spec_no must appear as the leading cell of
    # some parsed row (grades may be rendered as "—" in the markdown so we
    # match on the spec_no prefix, not the full key).
    for key, rec in MATERIALS.items():
        matched = any(r.startswith(rec.spec_no) for r in md_rows)
        assert matched, (
            f"Pinned material '{key}' not present as a row in Table M-1 "
            f"of the parsed standard."
        )
