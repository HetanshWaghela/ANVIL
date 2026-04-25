"""Markdown table extraction into the ParsedTable schema."""

from __future__ import annotations

import re

from anvil.schemas.document import ParsedTable, TableCell


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def _looks_like_header_row(cells: list[str]) -> bool:
    """Return True for label rows, False for data-heavy rows.

    This is intentionally structural rather than domain-specific: a row is
    header-like when most non-empty cells contain alphabetic labels, units, or
    short column tokens, and not mostly numeric values.
    """
    non_empty = [c for c in cells if c.strip()]
    if not non_empty:
        return False
    numeric = sum(1 for c in non_empty if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", c.strip()))
    alpha_or_unit = sum(
        1
        for c in non_empty
        if re.search(r"[A-Za-z°%/()]", c) or re.search(r"\d+\s*[A-Za-z°%]", c)
    )
    return alpha_or_unit >= max(1, len(non_empty) - 1) and numeric <= len(non_empty) // 3


def _merge_header_rows(top: list[str], lower: list[str]) -> list[str]:
    """Merge stacked table headers without forcing group labels everywhere."""
    merged: list[str] = []
    top_counts = {cell: top.count(cell) for cell in set(top) if cell}
    for top_cell, lower_cell in zip(top, lower, strict=True):
        top_clean = top_cell.strip()
        lower_clean = lower_cell.strip()
        if not top_clean:
            merged.append(lower_clean)
        elif not lower_clean:
            merged.append(top_clean)
        elif top_clean.lower() == lower_clean.lower():
            merged.append(lower_clean)
        elif top_counts.get(top_clean, 0) > 1 and re.search(r"\d", lower_clean):
            # A repeated group label over numeric/unit columns adds noise.
            merged.append(lower_clean)
        else:
            merged.append(f"{top_clean} {lower_clean}")
    return merged


def _split_fused_alpha_numeric(cells: list[str]) -> list[str]:
    """Repair cells like 'Forging138' when the next column is blank.

    PDF table extraction often fuses the end of a text cell with the first
    numeric value in the next cell. Split only when the next cell is empty and
    the token boundary is unambiguous.
    """
    repaired = list(cells)
    for i in range(len(repaired) - 1):
        if repaired[i + 1].strip():
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z /().-]*?)([-+]?\d+(?:\.\d+)?)", repaired[i].strip())
        if match is None:
            continue
        repaired[i] = match.group(1).strip()
        repaired[i + 1] = match.group(2).strip()
    return repaired


def _is_continuation_row(cells: list[str], width: int) -> bool:
    """Detect wrapped row fragments emitted by PDF-to-Markdown tools."""
    if not cells:
        return False
    non_empty_indexes = [i for i, cell in enumerate(cells) if cell.strip()]
    if not non_empty_indexes:
        return False
    if max(non_empty_indexes) > min(2, width - 1):
        return False
    return len(non_empty_indexes) <= max(1, min(2, width // 4))


def _join_wrapped_fragment(prefix: str, suffix: str) -> str:
    left = prefix.strip()
    right = suffix.strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return f"{left}{right}"
    return f"{left} {right}"


def _repair_wrapped_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    """Merge sparse continuation rows into the preceding logical row."""
    repaired: list[list[str]] = []
    for raw_row in rows:
        row = _split_fused_alpha_numeric((raw_row + [""] * width)[:width])
        if repaired and _is_continuation_row(row, width):
            prev = repaired[-1]
            for idx, cell in enumerate(row):
                if cell.strip():
                    prev[idx] = _join_wrapped_fragment(prev[idx], cell)
            continue
        repaired.append(row)
    return repaired


def extract_markdown_table(
    table_id: str,
    source_paragraph: str,
    source_page: int,
    md_block: str,
    caption: str | None = None,
) -> ParsedTable | None:
    """Parse a markdown pipe-table into a ParsedTable.

    A valid markdown table has:
        | h1 | h2 | h3 |
        |----|----|----|
        | c1 | c2 | c3 |

    Returns None if the block does not look like a table.
    """
    lines = [ln.rstrip() for ln in md_block.splitlines() if ln.strip()]
    # Keep only lines that look like table rows
    table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(table_lines) < 2:
        return None

    sep_idx: int | None = None
    for i, ln in enumerate(table_lines):
        stripped = ln.strip().strip("|")
        if re.fullmatch(r"[\s\-\|:]+", stripped):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return None

    def split_row(ln: str) -> list[str]:
        inner = ln.strip()
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        return [c.strip() for c in inner.split("|")]

    raw_headers = split_row(table_lines[sep_idx - 1])
    width = len(raw_headers)
    raw_data_rows = [split_row(ln) for ln in table_lines[sep_idx + 1 :]]

    headers = raw_headers
    data_rows = raw_data_rows
    if data_rows:
        first_row = (data_rows[0] + [""] * width)[:width]
        sparse_header = sum(1 for h in headers if h.strip()) <= max(1, width // 3)
        if sparse_header and _looks_like_header_row(first_row):
            headers = _merge_header_rows(headers, first_row)
            data_rows = data_rows[1:]
    if _is_separator_row(headers):
        return None

    data_rows = _repair_wrapped_rows(data_rows, len(headers))

    rows: list[list[TableCell]] = []
    for r_idx, cells_text in enumerate(data_rows):
        # Pad/truncate to header width
        cells_text = (cells_text + [""] * len(headers))[: len(headers)]
        row_cells = [
            TableCell(row=r_idx, col=c_idx, text=txt) for c_idx, txt in enumerate(cells_text)
        ]
        rows.append(row_cells)

    return ParsedTable(
        table_id=table_id,
        caption=caption,
        headers=headers,
        rows=rows,
        source_page=source_page,
        source_paragraph=source_paragraph,
    )
