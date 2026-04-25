"""Parser benchmark schemas — typed models for M7 parser comparison."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anvil.schemas.document import DocumentElement, ParsedFormula, ParsedTable


class ParserOutput(BaseModel):
    """Raw output of a single parser run on a single PDF."""

    system: str = Field(
        description="Parser system identifier: 'pymupdf4llm', 'naive_pdfminer', 'reducto', 'azure_di'"
    )
    pdf_path: str = Field(description="Path to the source PDF")
    page_count: int = Field(description="Total pages in the PDF")
    elements: list[DocumentElement] = Field(
        default_factory=list, description="Parsed document elements"
    )
    latency_ms: float = Field(description="Wall-clock parsing time in milliseconds")
    cost_usd: float | None = Field(
        None, description="API cost in USD (None for local parsers)"
    )
    pdf_mtime: float = Field(
        0.0, description="mtime of the source PDF at parse time, for cache validation"
    )


class GroundTruthAnnotation(BaseModel):
    """Ground truth annotation for a benchmark PDF."""

    pdf_path: str = Field(description="Path to the source PDF")
    tables: list[ParsedTable] = Field(default_factory=list)
    formulas: list[ParsedFormula] = Field(default_factory=list)
    paragraph_refs: list[str] = Field(
        default_factory=list, description="Expected paragraph refs, e.g. ['A-27', 'B-12']"
    )
    section_headings: list[str] = Field(
        default_factory=list, description="Expected section headings"
    )


class ParserMetricResult(BaseModel):
    """Metrics for a single parser × PDF evaluation."""

    system: str
    pdf_path: str
    table_recovery_f1: float = Field(
        description="Cell-level exact-match F1 for table recovery"
    )
    formula_fidelity: float = Field(
        description="Fraction of GT formulas correctly extracted (plain_text match)"
    )
    paragraph_ref_recall: float = Field(
        description="Fraction of expected paragraph refs recovered"
    )
    section_recall: float = Field(
        description="Fraction of expected section headings recovered"
    )
    latency_ms_per_page: float = Field(
        description="Parsing latency per page in milliseconds"
    )
    cost_usd_per_page: float | None = Field(
        None, description="API cost per page in USD (None for local parsers)"
    )
