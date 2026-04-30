"""Cross-reference detection for SPES-1 paragraph refs and tables."""

from __future__ import annotations

import re

from anvil.schemas.document import CrossReference, DocumentElement

_CODE_REF_PATTERN = (
    r"(?:[A-Z]{1,5}-\d+(?:\.\d+)?(?:-\d+)?(?:\([a-z0-9]+\))?"
    r"|[0-9]{1,2}-[0-9]+(?:\.\d+)?(?:\([a-z]\))?)"
)

# Patterns tuned for SPES-1 and real ASME-style references:
#   A-27, B-12, M-1, UG-27, UW-12, UCS-23, Table UG-43, Table 13-13(c)
XREF_PATTERN = re.compile(
    r"(?:see|per|as\s+specified\s+in|in\s+accordance\s+with|refer\s+to|from|"
    r"obtained\s+from|consult|shall\s+be\s+(?:obtained|taken)\s+from)\s+"
    rf"(Table\s+{_CODE_REF_PATTERN}|{_CODE_REF_PATTERN})",
    re.IGNORECASE,
)

# Standalone reference to a Table X-N anywhere in text
TABLE_REF_PATTERN = re.compile(rf"Table\s+({_CODE_REF_PATTERN})", re.IGNORECASE)

# Paragraph-like references without leading verbs.
PARA_REF_PATTERN = re.compile(rf"\b({_CODE_REF_PATTERN})\b", re.IGNORECASE)

# Material specifications: SM-516 Gr 70, SM-240 Type 304
MATERIAL_SPEC_PATTERN = re.compile(
    r"SM-\d+(?:\s+(?:Gr(?:ade)?|Type)\s+\w+)?",
    re.IGNORECASE,
)


def _normalize_ref(ref_text: str) -> str:
    """Normalize reference string ('table m-1' → 'Table M-1')."""
    s = ref_text.strip()
    if s.lower().startswith("table"):
        rest = s.split(None, 1)[1] if len(s.split(None, 1)) > 1 else ""
        return f"Table {rest.upper()}"
    return s.upper()


def detect_cross_references(
    source_id: str,
    text: str,
    element_ids_by_ref: dict[str, str],
) -> list[CrossReference]:
    """Extract cross-references from a block of text.

    Args:
        source_id: element_id of the source paragraph/element
        text: the text to scan
        element_ids_by_ref: map from normalized paragraph ref (e.g. 'A-27(C)(1)',
                            'TABLE M-1') → element_id of the target element

    Returns:
        List of CrossReference objects for all detected references where a
        matching target element exists. Unresolved references are skipped.
    """
    refs: list[CrossReference] = []
    seen: set[tuple[str, str]] = set()

    # 1) Explicit "see X" style references
    for m in XREF_PATTERN.finditer(text):
        raw = m.group(1)
        norm = _normalize_ref(raw)
        target = element_ids_by_ref.get(norm.upper())
        if target and (source_id, target) not in seen:
            refs.append(
                CrossReference(
                    source_id=source_id,
                    target_id=target,
                    reference_text=m.group(0),
                    reference_type="references",
                )
            )
            seen.add((source_id, target))

    # 2) Standalone table references (e.g. "from Table B-12")
    for m in TABLE_REF_PATTERN.finditer(text):
        norm = _normalize_ref(f"Table {m.group(1)}")
        target = element_ids_by_ref.get(norm.upper())
        if target and (source_id, target) not in seen:
            refs.append(
                CrossReference(
                    source_id=source_id,
                    target_id=target,
                    reference_text=m.group(0),
                    reference_type="requires",
                )
            )
            seen.add((source_id, target))

    # 3) Bare paragraph refs appearing in the text
    for m in PARA_REF_PATTERN.finditer(text):
        norm = m.group(1).upper()
        target = element_ids_by_ref.get(norm)
        if target and target != source_id and (source_id, target) not in seen:
            refs.append(
                CrossReference(
                    source_id=source_id,
                    target_id=target,
                    reference_text=m.group(0),
                    reference_type="references",
                )
            )
            seen.add((source_id, target))

    return refs


def link_elements(elements: list[DocumentElement]) -> list[DocumentElement]:
    """Populate `cross_references` for every element, returning the same list.

    Builds an index of paragraph_ref → element_id, then scans each element's
    content for references. Mutates elements in place for simplicity, then
    returns them for chaining.
    """
    index: dict[str, str] = {}
    for el in elements:
        if el.paragraph_ref:
            index[el.paragraph_ref.upper()] = el.element_id
        if el.table and el.table.table_id:
            index[f"TABLE {el.table.table_id.upper()}"] = el.element_id

    for el in elements:
        refs = detect_cross_references(el.element_id, el.content, index)
        el.cross_references = refs
    return elements
