"""Knowledge graph node/edge schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Types of nodes in the anvil knowledge graph."""

    SECTION = "section"
    TABLE = "table"
    FORMULA = "formula"
    DEFINITION = "definition"
    CONDITION = "condition"
    NOTE = "note"
    MATERIAL = "material"


class EdgeType(StrEnum):
    """Typed edges in the knowledge graph — these encode semantic relationships."""

    REFERENCES = "references"  # A document-level reference (soft)
    REQUIRES = "requires"  # Formula requires a value from this table/definition
    DEFINES = "defines"  # Paragraph defines a variable or concept
    CONSTRAINS = "constrains"  # Condition constrains formula applicability
    SUPERSEDES = "supersedes"  # Later revision supersedes earlier
    CONTAINS = "contains"  # Parent section contains child element


class GraphNode(BaseModel):
    """A typed knowledge graph node (Pydantic mirror of NetworkX attrs)."""

    node_id: str
    node_type: NodeType
    paragraph_ref: str | None = None
    title: str | None = None
    content_summary: str = Field(default="", description="Short summary for display")
    source_element_id: str | None = Field(
        None, description="ID of the DocumentElement this node came from"
    )
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A typed edge between two graph nodes."""

    source: str
    target: str
    edge_type: EdgeType
    reference_text: str = Field(
        default="", description="Human-readable rationale for this edge"
    )
    weight: float = Field(default=1.0, ge=0.0)
