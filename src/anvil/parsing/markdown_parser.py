"""Markdown parser for the SPES-1 synthetic standard.

The synthetic standard is distributed as Markdown to keep the repo simple
and avoid a binary PDF dependency for tests. A PDF parser is provided in
`pdf_parser.py` for real-world documents; both pipelines emit the same
DocumentElement schema.

Parsing strategy:
1. Split on heading lines (`##` = Part, `###` = Section, numbered like `A-27`)
2. For each section, classify nested content:
   - Fenced code blocks containing `=` → formulas
   - Pipe tables → tables
   - Everything else → paragraph text
3. Identify the paragraph reference from the heading (e.g. "### A-27 Thickness...")
4. Emit DocumentElements with stable IDs, then run section_linker to resolve
   cross-references.
"""

from __future__ import annotations

import re
from pathlib import Path

from anvil.parsing.formula_extractor import extract_formulas
from anvil.parsing.section_linker import link_elements
from anvil.parsing.table_extractor import extract_markdown_table
from anvil.schemas.document import DocumentElement, ElementType

# Matches headings like "### A-27 Thickness of Shells" or "### B-12 Joint Efficiency"
_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)
_CODE_REF_PATTERN = (
    r"(?:[A-Z]{1,5}-\d+(?:\.\d+)?(?:-\d+)?(?:\([a-z0-9]+\))?"
    r"|[0-9]{1,2}-[0-9]+(?:\.\d+)?(?:\([a-z]\))?)"
)
_PARA_REF_IN_HEADING = re.compile(rf"\b({_CODE_REF_PATTERN})\b", re.IGNORECASE)
_TABLE_ID_RE = re.compile(rf"\bTable\s+({_CODE_REF_PATTERN})\b", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_BLOCK_RE = re.compile(r"(?:^\s*\|.*\|\s*$\n?)+", re.MULTILINE)


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9.]+", "-", s)
    return s.strip("-")


def _unique_id(base: str, seen: dict[str, int]) -> str:
    """Return a stable unique id, preserving the first occurrence unchanged."""
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return base
    return f"{base}-{count + 1}"


def _strip_md_formatting(text: str) -> str:
    """Strip markdown bold/italic markers (**, *, __) from text."""
    s = text.strip()
    s = s.replace("<br>", " ")
    s = s.replace("<br/>", " ")
    # Strip bold (**text**) and italic (*text*) markers
    s = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", s)
    # Strip underscore-style bold/italic (__text__, _text_)
    s = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", s)
    # Drop common extraction artifacts while preserving ASME paragraph refs.
    s = re.sub(r"[^\w().\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_paragraph_ref(heading_text: str) -> str | None:
    # Strip bold/italic markers so pymupdf4llm headings like
    # "**A-27 Thickness...**" match the paragraph-ref regex.
    cleaned = _strip_md_formatting(heading_text)
    if cleaned.lower().startswith("table "):
        return None
    m = _PARA_REF_IN_HEADING.match(cleaned)
    if m is None:
        m = _PARA_REF_IN_HEADING.search(cleaned)
    return m.group(1).upper() if m else None


def parse_markdown_standard(source: str | Path) -> list[DocumentElement]:
    """Parse a markdown standard into a flat list of DocumentElements.

    Args:
        source: Either a filesystem path to a .md file, or raw markdown text.

    Returns:
        A list of DocumentElements. The list is ordered as in the source.
        Cross-references are populated via section_linker.
    """
    text = _load_text(source)
    # Simulate pagination: ~40 lines per page (for provenance)
    lines = text.splitlines()
    line_to_page = {i: (i // 40) + 1 for i in range(len(lines))}

    elements: list[DocumentElement] = []
    parent_stack: list[tuple[int, str]] = []  # (heading_level, element_id)
    seen_ids: dict[str, int] = {}

    # Find all heading spans; for each, the "content" runs until the next heading.
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return elements

    for i, m in enumerate(headings):
        level = len(m.group(1))
        title_full = m.group(2).strip()
        start_line = text[: m.start()].count("\n")
        page = line_to_page.get(start_line, 1)

        # Content extends to the next heading (or end of text)
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end() : end].strip("\n")

        para_ref = _extract_paragraph_ref(title_full)
        element_id = _unique_id(f"sec-{_slugify(para_ref or title_full)}", seen_ids)

        # Maintain parent stack
        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1][1] if parent_stack else None

        elements.append(
            DocumentElement(
                element_id=element_id,
                element_type=ElementType.SECTION,
                paragraph_ref=para_ref,
                title=title_full,
                content=body,
                page_number=page,
                parent_id=parent_id,
            )
        )
        parent_stack.append((level, element_id))

        # Extract child tables
        for t_idx, tm in enumerate(_TABLE_BLOCK_RE.finditer(body)):
            table_block = tm.group(0)
            table_id = _detect_table_id(title_full, body, para_ref, t_idx)
            table_ref = para_ref or (None if table_id.startswith("TBL-") else table_id)
            table = extract_markdown_table(
                table_id=table_id,
                source_paragraph=table_ref or element_id,
                source_page=page,
                md_block=table_block,
                caption=_detect_table_caption(title_full, table_id),
            )
            if table is None:
                continue
            el_id = _unique_id(f"tbl-{_slugify(table_id)}", seen_ids)
            elements.append(
                DocumentElement(
                    element_id=el_id,
                    element_type=ElementType.TABLE,
                    paragraph_ref=table_ref,
                    title=f"Table {table_id}",
                    content=table_block.strip(),
                    page_number=page,
                    parent_id=element_id,
                    table=table,
                )
            )

        # Extract child formulas
        if para_ref:
            formulas = extract_formulas(para_ref, body)
            for f in formulas:
                formula_id = _unique_id(f"fml-{_slugify(f.formula_id)}", seen_ids)
                elements.append(
                    DocumentElement(
                        element_id=formula_id,
                        element_type=ElementType.FORMULA,
                        paragraph_ref=para_ref,
                        title=f.formula_id,
                        content=f.plain_text,
                        page_number=page,
                        parent_id=element_id,
                        formula=f,
                    )
                )

    return link_elements(elements)


def _detect_table_id(
    heading: str, body: str, para_ref: str | None, index: int
) -> str:
    """Derive a canonical table id like 'M-1' or 'B-12' from surrounding text."""
    # Trust explicit table headings. Avoid scanning the whole body for
    # "Table X" because normal paragraphs often cite other tables; using those
    # citations as the current table id creates false labels.
    m = _TABLE_ID_RE.search(_strip_md_formatting(heading))
    if m:
        return m.group(1).upper()
    # In SPES-1, a section like "B-12 Joint Efficiency" contains the table
    # itself without a separate "Table B-12" heading.
    if para_ref is not None:
        return para_ref if index == 0 else f"{para_ref}-{index}"
    return f"TBL-{index}"


def _detect_table_caption(heading: str, table_id: str) -> str | None:
    """Return text after headings like `Table UW-12 Joint Efficiencies`."""
    cleaned = _strip_md_formatting(heading)
    patterns = [
        rf"\bTable\s+{re.escape(table_id)}\b\s*(.*)$",
        rf"\b{re.escape(table_id)}\b\s*(.*)$",
    ]
    caption = ""
    for pattern in patterns:
        m = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if m is not None:
            caption = m.group(1).strip(" -:;")
            break
    if not caption:
        return None
    caption = re.sub(r"\bCont(?:inued)?'?d\b\.?", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\(\s*\)", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip(" -:;")
    return caption or None


def _load_text(source: str | Path) -> str:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        return Path(source).read_text(encoding="utf-8")
    if isinstance(source, str):
        return source
    raise TypeError(f"Unsupported source type: {type(source)}")
