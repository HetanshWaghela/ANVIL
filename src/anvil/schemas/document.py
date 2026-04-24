"""Document element schemas — the output of the parsing layer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    """Coarse classification of a parsed document element."""

    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORMULA = "formula"
    DEFINITION = "definition"
    CONDITION = "condition"
    NOTE = "note"
    FIGURE = "figure"
    LIST_ITEM = "list_item"


class CrossReference(BaseModel):
    """A typed cross-reference between two document elements."""

    source_id: str = Field(description="ID of the element containing the reference")
    target_id: str = Field(description="ID of the referenced element")
    reference_text: str = Field(description="The actual reference text, e.g. 'see A-23'")
    reference_type: str = Field(
        description="Type: 'defines', 'references', 'constrains', 'requires'"
    )


class TableCell(BaseModel):
    """A single cell in a parsed table."""

    row: int
    col: int
    text: str
    is_header: bool = False
    spans_rows: int = 1
    spans_cols: int = 1


class ParsedTable(BaseModel):
    """A structured representation of a parsed table."""

    table_id: str
    caption: str | None = None
    headers: list[str]
    rows: list[list[TableCell]]
    source_page: int
    source_paragraph: str = Field(
        description="The paragraph this table belongs to, e.g. B-12"
    )


class FormulaVariable(BaseModel):
    """A named variable appearing in a formula."""

    symbol: str = Field(description="e.g. 'S'")
    name: str = Field(description="e.g. 'maximum allowable stress'")
    unit: str = Field(description="e.g. 'MPa'")
    source: str = Field(description="Where to find this value, e.g. 'Table M-1'")


class ParsedFormula(BaseModel):
    """A structured representation of a parsed formula/equation."""

    formula_id: str
    latex: str = Field(description="LaTeX representation of the formula")
    plain_text: str = Field(
        description="Plain text form, e.g. 't = (P × R) / (S × E − 0.6 × P)'"
    )
    variables: list[FormulaVariable]
    applicability_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions under which this formula applies, e.g. 'when t ≤ R/2'",
    )
    source_paragraph: str


class DocumentElement(BaseModel):
    """A single parsed element of a document (section, table, formula, etc.)."""

    element_id: str
    element_type: ElementType
    paragraph_ref: str | None = Field(
        None, description="Paragraph reference, e.g. 'A-27(c)(1)'"
    )
    title: str | None = None
    content: str
    page_number: int
    parent_id: str | None = None
    cross_references: list[CrossReference] = Field(default_factory=list)
    table: ParsedTable | None = None
    formula: ParsedFormula | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
