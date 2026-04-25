"""Unit tests for parser benchmark metric functions.

These tests use synthetic ParsedTable / ParsedFormula / DocumentElement objects
directly — no PDFs, no APIs. They verify the mathematical behavior of each
metric function.
"""

from __future__ import annotations

import pytest

from anvil.evaluation.parser_metrics import (
    score_formula_fidelity,
    score_paragraph_ref_recall,
    score_section_recall,
    score_table_recovery_f1,
)
from anvil.schemas.document import (
    DocumentElement,
    ElementType,
    FormulaVariable,
    ParsedFormula,
    ParsedTable,
    TableCell,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(
    table_id: str,
    headers: list[str],
    data: list[list[str]],
    source_page: int = 1,
    source_paragraph: str = "A-1",
) -> ParsedTable:
    """Build a ParsedTable from simple lists."""
    rows = []
    for r_idx, row_data in enumerate(data):
        cells = [TableCell(row=r_idx, col=c_idx, text=t) for c_idx, t in enumerate(row_data)]
        rows.append(cells)
    return ParsedTable(
        table_id=table_id,
        headers=headers,
        rows=rows,
        source_page=source_page,
        source_paragraph=source_paragraph,
    )


def _make_formula(
    formula_id: str,
    plain_text: str,
    source_paragraph: str = "A-27",
) -> ParsedFormula:
    """Build a minimal ParsedFormula."""
    return ParsedFormula(
        formula_id=formula_id,
        latex="",
        plain_text=plain_text,
        variables=[
            FormulaVariable(symbol="t", name="thickness", unit="mm", source="calculated")
        ],
        source_paragraph=source_paragraph,
    )


def _make_element(
    element_id: str,
    paragraph_ref: str | None = None,
    title: str | None = None,
    content: str = "",
) -> DocumentElement:
    """Build a minimal DocumentElement."""
    return DocumentElement(
        element_id=element_id,
        element_type=ElementType.SECTION,
        paragraph_ref=paragraph_ref,
        title=title,
        content=content,
        page_number=1,
    )


# ---------------------------------------------------------------------------
# Table Recovery F1
# ---------------------------------------------------------------------------


class TestTableRecoveryF1:
    def test_perfect_match(self) -> None:
        """Identical tables → F1 = 1.0."""
        gt = _make_table("M-1", ["A", "B"], [["1", "2"], ["3", "4"]])
        pred = _make_table("M-1", ["A", "B"], [["1", "2"], ["3", "4"]])
        assert score_table_recovery_f1([pred], [gt]) == 1.0

    def test_empty_both(self) -> None:
        """No tables on either side → 1.0."""
        assert score_table_recovery_f1([], []) == 1.0

    def test_empty_predicted(self) -> None:
        """GT has tables but prediction is empty → 0.0."""
        gt = _make_table("M-1", ["A"], [["1"]])
        assert score_table_recovery_f1([], [gt]) == 0.0

    def test_empty_ground_truth(self) -> None:
        """No GT tables → 1.0 regardless of predictions."""
        pred = _make_table("M-1", ["A"], [["1"]])
        assert score_table_recovery_f1([pred], []) == 1.0

    def test_partial_match(self) -> None:
        """Overlapping but not identical cells → F1 between 0 and 1."""
        gt = _make_table("M-1", ["A", "B"], [["1", "2"], ["3", "4"]])
        # Same headers but different data in second row
        pred = _make_table("M-1", ["A", "B"], [["1", "2"], ["X", "Y"]])
        f1 = score_table_recovery_f1([pred], [gt])
        assert 0.0 < f1 < 1.0

    def test_extra_predicted_cells_lower_precision(self) -> None:
        """Extra cells in prediction lowers precision but recall stays high."""
        gt = _make_table("M-1", ["A"], [["1"]])
        pred = _make_table("M-1", ["A"], [["1"], ["EXTRA"]])
        f1 = score_table_recovery_f1([pred], [gt])
        # All GT cells found (recall=1), but precision < 1 due to extra cell
        assert 0.5 < f1 < 1.0

    def test_multiple_tables(self) -> None:
        """Multiple GT tables — each scored independently, macro-averaged."""
        gt1 = _make_table("M-1", ["A"], [["1"]], source_page=1)
        gt2 = _make_table("B-12", ["X"], [["Y"]], source_page=2)
        pred1 = _make_table("M-1", ["A"], [["1"]], source_page=1)
        # pred2 is completely wrong
        pred2 = _make_table("B-12", ["Z"], [["W"]], source_page=2)
        f1 = score_table_recovery_f1([pred1, pred2], [gt1, gt2])
        # gt1 matched perfectly (1.0), gt2 matched with 0 overlap (0.0)
        # macro avg = 0.5
        assert 0.4 <= f1 <= 0.6

    def test_whitespace_normalization(self) -> None:
        """Cell text is normalized: extra whitespace doesn't cause mismatch."""
        gt = _make_table("M-1", ["Header A"], [["  value  1  "]])
        pred = _make_table("M-1", ["Header A"], [["value 1"]])
        assert score_table_recovery_f1([pred], [gt]) == 1.0


# ---------------------------------------------------------------------------
# Formula Fidelity
# ---------------------------------------------------------------------------


class TestFormulaFidelity:
    def test_perfect_match(self) -> None:
        """All formulas match → 1.0."""
        gt = [_make_formula("f1", "t = (P × R) / (S × E − 0.6 × P)")]
        pred = [_make_formula("f1", "t = (P × R) / (S × E − 0.6 × P)")]
        assert score_formula_fidelity(pred, gt) == 1.0

    def test_no_ground_truth(self) -> None:
        """No GT formulas → 1.0."""
        assert score_formula_fidelity([], []) == 1.0

    def test_empty_predicted(self) -> None:
        """GT has formulas but predicted is empty → 0.0."""
        gt = [_make_formula("f1", "t = P × R")]
        assert score_formula_fidelity([], gt) == 0.0

    def test_partial_match(self) -> None:
        """Some formulas match, some don't."""
        gt = [
            _make_formula("f1", "t = (P × R) / (S × E − 0.6 × P)"),
            _make_formula("f2", "t = (P × Ro) / (S × E + 0.4 × P)"),
        ]
        pred = [
            _make_formula("f1", "t = (P × R) / (S × E − 0.6 × P)"),
            _make_formula("f2", "WRONG FORMULA"),
        ]
        fid = score_formula_fidelity(pred, gt)
        assert fid == pytest.approx(0.5)

    def test_whitespace_normalization(self) -> None:
        """Extra whitespace in formulas doesn't cause mismatch."""
        gt = [_make_formula("f1", "t = P × R")]
        pred = [_make_formula("f1", "  t  =  P  ×  R  ")]
        assert score_formula_fidelity(pred, gt) == 1.0

    def test_compressed_whitespace_match(self) -> None:
        """pymupdf4llm strips all spaces from formulas — should still match.

        PDF round-trip: 't = (P × R) / (S × E − 0.6 × P)' becomes
        't=(P×R)/(S×E−0.6×P)'. These are semantically identical.
        """
        gt = [_make_formula("f1", "t = (P × R) / (S × E − 0.6 × P)")]
        pred = [_make_formula("f1", "t=(P×R)/(S×E−0.6×P)")]
        assert score_formula_fidelity(pred, gt) == 1.0


# ---------------------------------------------------------------------------
# Paragraph Reference Recall
# ---------------------------------------------------------------------------


class TestParagraphRefRecall:
    def test_perfect_recall(self) -> None:
        elements = [
            _make_element("e1", paragraph_ref="A-27"),
            _make_element("e2", paragraph_ref="B-12"),
        ]
        gt_refs = ["A-27", "B-12"]
        assert score_paragraph_ref_recall(elements, gt_refs) == 1.0

    def test_empty_gt(self) -> None:
        elements = [_make_element("e1", paragraph_ref="A-27")]
        assert score_paragraph_ref_recall(elements, []) == 1.0

    def test_empty_predicted(self) -> None:
        assert score_paragraph_ref_recall([], ["A-27"]) == 0.0

    def test_partial_recall(self) -> None:
        elements = [_make_element("e1", paragraph_ref="A-27")]
        gt_refs = ["A-27", "B-12"]
        assert score_paragraph_ref_recall(elements, gt_refs) == pytest.approx(0.5)

    def test_case_insensitive(self) -> None:
        elements = [_make_element("e1", paragraph_ref="a-27")]
        gt_refs = ["A-27"]
        assert score_paragraph_ref_recall(elements, gt_refs) == 1.0


# ---------------------------------------------------------------------------
# Section Recall
# ---------------------------------------------------------------------------


class TestSectionRecall:
    def test_perfect_recall(self) -> None:
        elements = [
            _make_element("e1", title="A-27 Thickness of Shells Under Internal Pressure"),
            _make_element("e2", title="B-12 Joint Efficiency"),
        ]
        gt_headings = ["A-27 Thickness of Shells Under Internal Pressure", "B-12 Joint Efficiency"]
        assert score_section_recall(elements, gt_headings) == 1.0

    def test_empty_gt(self) -> None:
        elements = [_make_element("e1", title="Some heading")]
        assert score_section_recall(elements, []) == 1.0

    def test_empty_predicted(self) -> None:
        assert score_section_recall([], ["Some heading"]) == 0.0

    def test_partial_recall(self) -> None:
        elements = [_make_element("e1", title="A-27 Thickness of Shells")]
        gt_headings = ["A-27 Thickness of Shells", "B-12 Joint Efficiency"]
        assert score_section_recall(elements, gt_headings) == pytest.approx(0.5)

    def test_substring_match(self) -> None:
        """Heading is found as substring in element title (handles formatting diff)."""
        elements = [
            _make_element("e1", title="### A-27 Thickness of Shells Under Internal Pressure ###")
        ]
        gt_headings = ["A-27 Thickness of Shells Under Internal Pressure"]
        assert score_section_recall(elements, gt_headings) == 1.0

    def test_case_insensitive(self) -> None:
        elements = [_make_element("e1", title="a-27 thickness of shells")]
        gt_headings = ["A-27 Thickness of Shells"]
        assert score_section_recall(elements, gt_headings) == 1.0
