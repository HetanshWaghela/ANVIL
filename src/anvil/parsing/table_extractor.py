"""Markdown table extraction into the ParsedTable schema."""

from __future__ import annotations

import re

from anvil.schemas.document import ParsedTable, TableCell


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

    headers = split_row(table_lines[sep_idx - 1])
    data_lines = table_lines[sep_idx + 1 :]

    rows: list[list[TableCell]] = []
    for r_idx, ln in enumerate(data_lines):
        cells_text = split_row(ln)
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
