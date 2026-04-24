"""Document parsing layer — raw document → typed DocumentElements."""

from __future__ import annotations

from anvil.parsing.formula_extractor import extract_formulas
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.parsing.pdf_parser import parse_pdf
from anvil.parsing.section_linker import (
    MATERIAL_SPEC_PATTERN,
    TABLE_REF_PATTERN,
    XREF_PATTERN,
    detect_cross_references,
    link_elements,
)
from anvil.parsing.table_extractor import extract_markdown_table

__all__ = [
    "MATERIAL_SPEC_PATTERN",
    "TABLE_REF_PATTERN",
    "XREF_PATTERN",
    "detect_cross_references",
    "extract_formulas",
    "extract_markdown_table",
    "link_elements",
    "parse_markdown_standard",
    "parse_pdf",
]
