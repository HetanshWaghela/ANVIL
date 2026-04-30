"""Phase 1 tests: parsing the synthetic standard."""

from __future__ import annotations

from anvil.parsing import (
    MATERIAL_SPEC_PATTERN,
    TABLE_REF_PATTERN,
    XREF_PATTERN,
    detect_cross_references,
    extract_formulas,
    extract_markdown_table,
    parse_markdown_standard,
)
from anvil.schemas.document import ElementType


def test_parse_synthetic_standard_produces_elements(parsed_elements) -> None:
    assert len(parsed_elements) > 5


def test_all_section_refs_present(parsed_elements) -> None:
    refs = {e.paragraph_ref for e in parsed_elements if e.paragraph_ref}
    assert {"A-23", "A-25", "A-27", "A-32", "B-12", "M-23"}.issubset(refs)


def test_formula_extraction_finds_a27(parsed_elements) -> None:
    formulas = [e for e in parsed_elements if e.element_type == ElementType.FORMULA]
    a27_formulas = [
        f for f in formulas if f.paragraph_ref and f.paragraph_ref.startswith("A-27")
    ]
    assert len(a27_formulas) >= 3, f"Expected A-27 formulas, got {len(a27_formulas)}"
    # First formula should have variables P, R, S, E, t
    f = a27_formulas[0]
    assert f.formula is not None
    symbols = {v.symbol for v in f.formula.variables}
    assert {"P", "S", "E"}.issubset(symbols)


def test_table_extraction_b12(parsed_elements) -> None:
    tables = [e for e in parsed_elements if e.element_type == ElementType.TABLE]
    b12 = [t for t in tables if t.table and t.table.table_id == "B-12"]
    assert len(b12) == 1
    tbl = b12[0].table
    assert tbl is not None
    # Should have 6 joint rows
    assert len(tbl.rows) == 6
    # Headers include Full RT
    assert any("Full" in h for h in tbl.headers)


def test_table_extraction_m1(parsed_elements) -> None:
    tables = [e for e in parsed_elements if e.element_type == ElementType.TABLE]
    m1 = [t for t in tables if t.table and t.table.table_id == "M-1"]
    assert len(m1) == 1
    tbl = m1[0].table
    assert tbl is not None
    assert len(tbl.rows) >= 3
    # First column should contain a SM- spec
    first_cells = [row[0].text for row in tbl.rows]
    assert any(c.startswith("SM-") for c in first_cells)


def test_cross_references_detected_a27(parsed_elements) -> None:
    a27 = next(e for e in parsed_elements if e.paragraph_ref == "A-27")
    # A-27 mentions both B-12 and M-1 in its body — must detect them
    target_ids = {x.target_id for x in a27.cross_references}
    tables_targeted = [t for t in target_ids if t.startswith("tbl-")]
    assert len(tables_targeted) >= 1, f"A-27 xrefs: {a27.cross_references}"


def test_xref_pattern_matches_see_pattern() -> None:
    text = "allowable stress shall be obtained from Table M-1"
    assert XREF_PATTERN.search(text) is not None


def test_table_ref_pattern() -> None:
    assert TABLE_REF_PATTERN.search("see Table B-12") is not None
    assert TABLE_REF_PATTERN.search("see Table UW-12") is not None
    assert TABLE_REF_PATTERN.search("see Table 13-13(c)") is not None
    assert TABLE_REF_PATTERN.search("no table here") is None


def test_material_spec_pattern() -> None:
    assert MATERIAL_SPEC_PATTERN.search("Use SM-516 Gr 70 plate") is not None
    assert MATERIAL_SPEC_PATTERN.search("Type 304 stainless SM-240 Type 304") is not None


def test_detect_cross_references_with_index() -> None:
    idx = {"A-27": "sec-a-27", "TABLE M-1": "tbl-m-1"}
    refs = detect_cross_references(
        "sec-caller",
        "See A-27 for thickness and refer to Table M-1 for stress values.",
        idx,
    )
    target_ids = {r.target_id for r in refs}
    assert "sec-a-27" in target_ids
    assert "tbl-m-1" in target_ids


def test_detect_cross_references_with_real_asme_refs() -> None:
    idx = {"UG-27": "sec-ug-27", "TABLE UW-12": "tbl-uw-12"}
    refs = detect_cross_references(
        "sec-caller",
        "Thickness is designed per UG-27 and joint efficiency is obtained from Table UW-12.",
        idx,
    )
    target_ids = {r.target_id for r in refs}

    assert "sec-ug-27" in target_ids
    assert "tbl-uw-12" in target_ids


def test_extract_formulas_from_code_block() -> None:
    text = "Some text\n\n```\nt = (P × R) / (S × E − 0.6 × P)\n```\n\nMore text."
    formulas = extract_formulas("A-27(c)(1)", text)
    assert len(formulas) == 1
    assert formulas[0].source_paragraph == "A-27(c)(1)"
    assert "=" in formulas[0].plain_text


def test_extract_markdown_table_roundtrip() -> None:
    md = """
| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |
"""
    tbl = extract_markdown_table("T-1", "para", 1, md)
    assert tbl is not None
    assert tbl.headers == ["A", "B"]
    assert len(tbl.rows) == 2
    assert tbl.rows[1][1].text == "4"


def test_extract_markdown_table_repairs_wrapped_wide_rows() -> None:
    md = """
|||Product|Product||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Spec|Grade|Form|40C|100C|150C|200C|250C|300C|350C|400C|450C|500C|
|SM-|Gr|Plate|138|138|138|134|127|121|114|103|73|49|
|516|70||||||||||||
|SM-|—|Forging138||138|138|133|125|119|112|100|71|47|
|105||||||||||||
"""
    tbl = extract_markdown_table("M-1", "M-1", 3, md)

    assert tbl is not None
    assert tbl.headers[:4] == ["Spec", "Grade", "Product Form", "40C"]
    assert [cell.text for cell in tbl.rows[0][:4]] == ["SM-516", "Gr 70", "Plate", "138"]
    assert [cell.text for cell in tbl.rows[1][:5]] == [
        "SM-105",
        "—",
        "Forging",
        "138",
        "138",
    ]


def test_parse_markdown_inline() -> None:
    md = """# Title

## Part A

### A-27 Thickness

t = (P × R) / (S × E − 0.6 × P)

```
t = (P × R) / (S × E − 0.6 × P)
```
"""
    elements = parse_markdown_standard(md)
    assert any(e.paragraph_ref == "A-27" for e in elements)


def test_parse_real_asme_style_refs_and_table_ids() -> None:
    md = """# ASME VIII

## **UG-27 THICKNESS OF SHELLS UNDER INTERNAL PRESSURE**

See Table UW-12 for joint efficiencies.

## **Table UW-12 Maximum Allowable Joint Efficiencies for Welded Joints**

| Joint Type | Full RT | Spot RT |
|---|---|---|
| Type No. 1 butt joint | 1.00 | 0.85 |
| Type No. 2 butt joint | 0.90 | 0.80 |

## ð **23** Þ **UG-99 STANDARD HYDROSTATIC TEST**

Test paragraph.
"""
    elements = parse_markdown_standard(md)
    refs = {e.paragraph_ref for e in elements if e.paragraph_ref}

    assert "UG-27" in refs
    assert "UG-99" in refs

    table = next(e.table for e in elements if e.table and e.table.table_id == "UW-12")
    assert table is not None
    assert table.caption == "Maximum Allowable Joint Efficiencies for Welded Joints"
    assert table.source_paragraph == "UW-12"
    table_el = next(e for e in elements if e.table and e.table.table_id == "UW-12")
    assert table_el.paragraph_ref == "UW-12"


def test_duplicate_headings_get_stable_unique_ids() -> None:
    md = """# Title

## Appendix

### Notes

First note.

### Notes

Second note.

### A-27 Thickness

```
t = (P × R) / (S × E − 0.6 × P)
```

### A-27 Thickness

```
t = (P × R) / (S × E − 0.6 × P)
```
"""
    elements = parse_markdown_standard(md)
    ids = [e.element_id for e in elements]

    assert len(ids) == len(set(ids))
    assert "sec-notes" in ids
    assert "sec-notes-2" in ids
    assert "sec-a-27" in ids
    assert "sec-a-27-2" in ids
    assert "fml-a-27-f0" in ids
    assert "fml-a-27-f0-2" in ids
