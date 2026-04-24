"""Edge type rules and weights for the knowledge graph."""

from __future__ import annotations

from anvil.schemas.knowledge_graph import EdgeType

# Edge weights influence graph expansion — higher-weight edges are preferred
# when ranking the expanded context. These are deliberately interpretable.
EDGE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.REQUIRES: 1.0,    # Formula requires a table/value — highest priority
    EdgeType.REFERENCES: 0.8,  # Document-level pointer
    EdgeType.DEFINES: 0.9,     # Definition of a variable/concept
    EdgeType.CONSTRAINS: 0.85, # Applicability condition
    EdgeType.CONTAINS: 0.5,    # Hierarchical containment
    EdgeType.SUPERSEDES: 0.3,  # Version relationship (rare in SPES-1)
}


# Reference-text keywords → edge type classifier. The parser's raw
# reference_text ("shall be obtained from Table M-1") determines the
# edge semantics.
def classify_edge(reference_text: str, target_is_table: bool) -> EdgeType:
    """Classify a cross-reference into an EdgeType."""
    s = reference_text.lower()
    if "obtained from" in s or "taken from" in s or target_is_table:
        return EdgeType.REQUIRES
    if "defined" in s or "defines" in s:
        return EdgeType.DEFINES
    if "applies when" in s or "applicable" in s or "shall satisfy" in s:
        return EdgeType.CONSTRAINS
    return EdgeType.REFERENCES
