"""Parser benchmark metrics — quantitative comparison of PDF parsing systems.

Each metric function takes predicted and ground-truth objects and returns a
float in [0, 1]. Higher is better for all metrics.

Mathematical definitions are documented inline per the project guardrail
that every metric must have a clear mathematical definition in code comments.
"""

from __future__ import annotations

import re

from anvil.schemas.document import DocumentElement, ParsedFormula, ParsedTable


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", text.strip())


def _normalize_cell_text(text: str) -> str:
    """Normalize a table cell value for comparison.

    Strips whitespace, lowercases, and removes trailing/leading punctuation
    that varies across parsers (e.g. trailing periods, leading dashes).
    """
    s = _normalize_whitespace(text).lower()
    # Remove leading/trailing hyphens and periods that are layout artifacts
    s = s.strip(".-– ")
    return s


# ---------------------------------------------------------------------------
# Table Recovery F1
# ---------------------------------------------------------------------------
# Mathematical definition:
#   For each ground-truth table, find the best-matching predicted table
#   (by source_page proximity and caption/id overlap). Then compute
#   cell-level exact-match metrics:
#
#     precision = |predicted_cells ∩ gt_cells| / |predicted_cells|
#     recall    = |predicted_cells ∩ gt_cells| / |gt_cells|
#     F1        = 2 × precision × recall / (precision + recall)
#
#   Each cell is identified by (row_index, col_index, normalized_text).
#   Header rows are included (row=-1 for header cells).
#   The final score is the macro-average F1 across all GT tables.


def _table_to_cell_set(table: ParsedTable) -> set[tuple[int, int, str]]:
    """Convert a ParsedTable to a set of (row, col, text) tuples.

    Header cells use row = -1 to distinguish them from data cells.
    """
    cells: set[tuple[int, int, str]] = set()
    # Header row
    for col, h in enumerate(table.headers):
        cells.add((-1, col, _normalize_cell_text(h)))
    # Data rows
    for row in table.rows:
        for cell in row:
            cells.add((cell.row, cell.col, _normalize_cell_text(cell.text)))
    return cells


def _match_tables(
    predicted: list[ParsedTable], ground_truth: list[ParsedTable]
) -> list[tuple[ParsedTable, ParsedTable | None]]:
    """Match GT tables to predicted tables using table_id and source_page.

    Returns a list of (gt_table, matched_pred_table_or_None) pairs.
    """
    available = list(predicted)
    matches: list[tuple[ParsedTable, ParsedTable | None]] = []

    for gt in ground_truth:
        best: ParsedTable | None = None
        best_score = -1.0

        for pred in available:
            score = 0.0
            # Same table_id is a strong signal
            if pred.table_id.upper() == gt.table_id.upper():
                score += 10.0
            # Same source page
            if pred.source_page == gt.source_page:
                score += 2.0
            # Caption overlap (if both have captions)
            if pred.caption and gt.caption:
                pred_words = set(pred.caption.lower().split())
                gt_words = set(gt.caption.lower().split())
                if pred_words & gt_words:
                    score += 1.0
            # Header overlap
            pred_headers = {h.lower().strip() for h in pred.headers}
            gt_headers = {h.lower().strip() for h in gt.headers}
            if pred_headers & gt_headers:
                overlap = len(pred_headers & gt_headers) / max(len(gt_headers), 1)
                score += overlap * 3.0

            if score > best_score:
                best_score = score
                best = pred

        if best is not None and best_score > 0:
            available.remove(best)
        else:
            best = None
        matches.append((gt, best))

    return matches


def score_table_recovery_f1(
    predicted: list[ParsedTable], ground_truth: list[ParsedTable]
) -> float:
    """Cell-level exact-match F1 for table recovery.

    Tables are matched by source_page + caption/id proximity, then cells
    are compared by (row, col, normalized_text). Header rows are included.

    Returns macro-average F1 across all GT tables. Returns 1.0 if both
    lists are empty (no tables to compare). Returns 0.0 if GT has tables
    but predicted is empty.
    """
    if not ground_truth:
        return 1.0
    if not predicted:
        return 0.0

    matches = _match_tables(predicted, ground_truth)
    f1_scores: list[float] = []

    for gt, pred in matches:
        if pred is None:
            f1_scores.append(0.0)
            continue

        gt_cells = _table_to_cell_set(gt)
        pred_cells = _table_to_cell_set(pred)

        if not gt_cells and not pred_cells:
            f1_scores.append(1.0)
            continue
        if not gt_cells or not pred_cells:
            f1_scores.append(0.0)
            continue

        intersection = gt_cells & pred_cells
        precision = len(intersection) / len(pred_cells)
        recall = len(intersection) / len(gt_cells)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


# ---------------------------------------------------------------------------
# Formula Fidelity
# ---------------------------------------------------------------------------
# Mathematical definition:
#   fidelity = |{f ∈ GT_formulas : ∃ p ∈ predicted_formulas s.t.
#                normalize(p.plain_text) == normalize(f.plain_text)}|
#              / |GT_formulas|
#
#   normalize() strips ALL whitespace so that semantically identical formulas
#   like 't = (P × R) / (S × E)' and 't=(P×R)/(S×E)' are treated as equal.
#   Returns 1.0 if GT has no formulas, 0.0 if GT has formulas but none match.


def _normalize_formula(text: str) -> str:
    """Normalize a formula for comparison by stripping ALL whitespace.

    PDF parsers often compress or expand whitespace around operators during
    round-trip (e.g. 't = (P × R) / ...' ↔ 't=(P×R)/...'). Since the
    mathematical content is identical, we strip all whitespace for comparison.
    """
    return re.sub(r"\s+", "", text.strip())


def score_formula_fidelity(
    predicted: list[ParsedFormula], ground_truth: list[ParsedFormula]
) -> float:
    """Fraction of GT formulas correctly extracted (plain_text match).

    After stripping all whitespace, each GT formula is checked for an exact
    match in the predicted formulas. This tolerates the whitespace differences
    introduced by PDF round-trips while still requiring the mathematical
    content to be identical.
    """
    if not ground_truth:
        return 1.0
    if not predicted:
        return 0.0

    pred_texts = {_normalize_formula(f.plain_text) for f in predicted}
    matched = sum(
        1
        for gt_f in ground_truth
        if _normalize_formula(gt_f.plain_text) in pred_texts
    )
    return matched / len(ground_truth)


# ---------------------------------------------------------------------------
# Paragraph Reference Recall
# ---------------------------------------------------------------------------
# Mathematical definition:
#   recall = |{r ∈ gt_refs : ∃ e ∈ predicted_elements s.t.
#              e.paragraph_ref is not None AND
#              normalize(e.paragraph_ref) == normalize(r)}|
#            / |gt_refs|
#
#   normalize() upper-cases the reference.
#   Returns 1.0 if gt_refs is empty.


def score_paragraph_ref_recall(
    predicted_elements: list[DocumentElement], gt_refs: list[str]
) -> float:
    """Fraction of expected paragraph refs found in any element.paragraph_ref.

    Each GT ref is checked (case-insensitively) against the set of
    paragraph_ref values across all predicted elements.
    """
    if not gt_refs:
        return 1.0
    if not predicted_elements:
        return 0.0

    pred_refs = {
        el.paragraph_ref.upper()
        for el in predicted_elements
        if el.paragraph_ref is not None
    }
    matched = sum(1 for r in gt_refs if r.upper() in pred_refs)
    return matched / len(gt_refs)


# ---------------------------------------------------------------------------
# Section Recall
# ---------------------------------------------------------------------------
# Mathematical definition:
#   recall = |{h ∈ gt_headings : ∃ e ∈ predicted_elements s.t.
#              e.title is not None AND
#              normalize(h) is a substring of normalize(e.title)}|
#            / |gt_headings|
#
#   normalize() lowercases and collapses whitespace.
#   Substring matching allows for minor formatting differences.
#   Returns 1.0 if gt_headings is empty.


def score_section_recall(
    predicted_elements: list[DocumentElement], gt_headings: list[str]
) -> float:
    """Fraction of expected section headings found in any element.title.

    Each GT heading is checked via case-insensitive substring matching
    against all element titles. This tolerates minor formatting differences
    (e.g. extra whitespace, numbering prefixes).
    """
    if not gt_headings:
        return 1.0
    if not predicted_elements:
        return 0.0

    pred_titles = [
        _normalize_whitespace(el.title).lower()
        for el in predicted_elements
        if el.title is not None
    ]

    matched = 0
    for h in gt_headings:
        norm_h = _normalize_whitespace(h).lower()
        if any(norm_h in t for t in pred_titles):
            matched += 1

    return matched / len(gt_headings)
