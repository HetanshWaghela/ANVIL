"""Translate parsed DocumentElements into typed GraphNodes."""

from __future__ import annotations

from anvil.schemas.document import DocumentElement, ElementType
from anvil.schemas.knowledge_graph import GraphNode, NodeType

_ELEMENT_TO_NODE: dict[ElementType, NodeType] = {
    ElementType.SECTION: NodeType.SECTION,
    ElementType.PARAGRAPH: NodeType.SECTION,
    ElementType.TABLE: NodeType.TABLE,
    ElementType.FORMULA: NodeType.FORMULA,
    ElementType.DEFINITION: NodeType.DEFINITION,
    ElementType.CONDITION: NodeType.CONDITION,
    ElementType.NOTE: NodeType.NOTE,
    ElementType.FIGURE: NodeType.SECTION,
    ElementType.LIST_ITEM: NodeType.SECTION,
}


def document_element_to_node(el: DocumentElement) -> GraphNode:
    """Convert a DocumentElement into a typed GraphNode."""
    summary = el.content[:200].replace("\n", " ").strip()
    return GraphNode(
        node_id=el.element_id,
        node_type=_ELEMENT_TO_NODE.get(el.element_type, NodeType.SECTION),
        paragraph_ref=el.paragraph_ref,
        title=el.title,
        content_summary=summary,
        source_element_id=el.element_id,
        metadata={"page": str(el.page_number)},
    )
