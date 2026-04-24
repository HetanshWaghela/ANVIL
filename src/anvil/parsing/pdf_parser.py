"""PDF parser wrapper around pymupdf4llm.

For real engineering standards distributed as PDF. The synthetic standard
is in markdown, so this is rarely exercised in tests — but the interface
matches `parse_markdown_standard` so downstream code does not care.
"""

from __future__ import annotations

from pathlib import Path

from anvil import ParsingError
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.schemas.document import DocumentElement


def parse_pdf(pdf_path: str | Path) -> list[DocumentElement]:
    """Parse a PDF into DocumentElements by delegating to pymupdf4llm.

    The PDF is first converted to markdown (preserving layout, headings, and
    tables via pymupdf4llm's to_markdown), then handed to the markdown parser.
    This keeps the DocumentElement emission in a single place.
    """
    try:
        import pymupdf4llm  # imported lazily — heavy dependency
    except ImportError as e:  # pragma: no cover
        raise ParsingError(
            "pymupdf4llm is required for PDF parsing. Install with `uv add pymupdf4llm`."
        ) from e

    path = Path(pdf_path)
    if not path.exists():
        raise ParsingError(f"PDF not found: {path}")

    md_text = pymupdf4llm.to_markdown(str(path))
    return parse_markdown_standard(md_text)
