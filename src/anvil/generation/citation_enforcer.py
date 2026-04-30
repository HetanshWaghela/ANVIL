"""Post-generation citation validation.

For every citation in an `AnvilResponse`, verify:

  1. The `source_element_id` exists in the retrieved context.
  2. The `quoted_text` actually appears in (or is strongly similar to) the
     cited source content.
  3. The `paragraph_ref` matches the source element's paragraph.

Pinned-data citations (e.g. the calculation engine quoting Table M-1) often
point at canonical SPES-1 elements that retrieval did not happen to surface
in the top-K. Those are still legitimate, but the validator MUST NOT trust
them blindly — that would let a hallucinated `quoted_text` slip through. To
validate them, callers thread an `element_index` (parsed-element-id → element)
into `validate_citations`. When the canonical branch is taken, we resolve the
ref against the index and require `quoted_text` to be a substring of the
real element's content. With no index available, the canonical branch fails
closed — fabricating "Table M-1" citations is the highest-risk failure mode
in this system, and a silent permissive fallback is exactly what we removed.

If any citation fails validation, the response should be downgraded to
`MEDIUM` confidence (or rejected) by the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvil.schemas.document import DocumentElement, ElementType
from anvil.schemas.generation import AnvilResponse, Citation
from anvil.schemas.retrieval import RetrievedChunk

# Strips a trailing sub-paragraph marker like "(c)", "(1)", "(iv)" — same
# rule as `CitationBuilder._SUBPARA_STRIP`. Kept as a local helper so the
# enforcer has no dependency on generation-layer internals.
_SUBPARA_STRIP = re.compile(r"\([A-Z0-9]+\)\s*$", re.IGNORECASE)


@dataclass
class CitationIssue:
    citation_index: int
    citation: Citation
    issue: str


@dataclass
class CitationValidationResult:
    total: int
    valid: int
    issues: list[CitationIssue]

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 1.0
        return self.valid / self.total

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _substring_match(quote: str, content: str, min_overlap: float = 0.6) -> bool:
    """Permissive match: exact substring OR ≥60% word overlap for the quote."""
    q = _normalize(quote)
    c = _normalize(content)
    if not q:
        return False
    if q in c:
        return True
    q_words = set(q.split())
    c_words = set(c.split())
    if not q_words:
        return False
    overlap = len(q_words & c_words) / len(q_words)
    return overlap >= min_overlap


def validate_citations(
    response: AnvilResponse,
    retrieved_chunks: list[RetrievedChunk],
    element_index: dict[str, DocumentElement] | None = None,
) -> CitationValidationResult:
    """Check every citation against the retrieved context.

    Args:
        response: the AnvilResponse to validate.
        retrieved_chunks: the RAG context the response was conditioned on.
        element_index: optional id→DocumentElement map of the FULL parsed
            standard. Used to validate `quoted_text` for canonical-ref
            citations (e.g. pinned-data Table M-1 quotes) when the cited
            element wasn't surfaced in retrieval. With no index available,
            such citations fail closed — fabricating a "Table M-1" quote is
            the worst failure mode in a compliance system, so silently
            trusting `quoted_text` here is not acceptable.
    """
    chunks_by_id: dict[str, RetrievedChunk] = {c.element_id: c for c in retrieved_chunks}
    # Also index retrieved chunks by paragraph_ref for fallback resolution.
    # LLMs often emit the paragraph ref (e.g. "UG-32") as source_element_id
    # instead of the internal element id ("sec-ug-32").
    chunks_by_para_ref: dict[str, RetrievedChunk] = {}
    for c in retrieved_chunks:
        if c.paragraph_ref:
            key = c.paragraph_ref.upper()
            if key not in chunks_by_para_ref:
                chunks_by_para_ref[key] = c
    issues: list[CitationIssue] = []

    all_citations: list[Citation] = list(response.citations)
    for step in response.calculation_steps:
        all_citations.append(step.citation)
        for iv in step.inputs.values():
            if iv.citation is not None:
                all_citations.append(iv.citation)

    canonical_index = (
        _build_paragraph_ref_index(element_index) if element_index else None
    )

    valid = 0
    for i, cit in enumerate(all_citations):
        source = chunks_by_id.get(cit.source_element_id)
        # Fallback: LLM often uses paragraph ref as source_element_id
        if source is None and cit.source_element_id:
            source = chunks_by_para_ref.get(cit.source_element_id.upper())
        # Fallback: try citation's paragraph_ref against retrieved chunks
        if source is None and cit.paragraph_ref:
            pr_key = cit.paragraph_ref.upper().strip()
            if pr_key.startswith("TABLE "):
                pr_key = pr_key[len("TABLE "):].strip()
            source = chunks_by_para_ref.get(pr_key)
            # Also try stripping sub-paragraph markers: UG-32(A) -> UG-32
            if source is None:
                stripped = _SUBPARA_STRIP.sub("", pr_key).strip()
                source = chunks_by_para_ref.get(stripped)
        if source is None:
            # Pinned-data citation: the cited element exists in the parsed
            # standard but wasn't in the top-K retrieval. We accept it ONLY
            # if (a) the paragraph_ref is canonical, AND (b) the
            # quoted_text is substantively present in the resolved
            # element's content. Without an `element_index` to check
            # against, we have no way to verify (b) — fail closed.
            if not (cit.paragraph_ref and _looks_like_spes1_ref(cit.paragraph_ref)):
                issues.append(
                    CitationIssue(
                        citation_index=i,
                        citation=cit,
                        issue=(
                            f"source_element_id '{cit.source_element_id}' not in "
                            f"retrieved context and paragraph_ref is not canonical."
                        ),
                    )
                )
                continue
            if canonical_index is None:
                issues.append(
                    CitationIssue(
                        citation_index=i,
                        citation=cit,
                        issue=(
                            f"source_element_id '{cit.source_element_id}' not in "
                            f"retrieved context; cannot validate canonical-ref "
                            f"citation against the standard because no "
                            f"element_index was provided to validate_citations()."
                        ),
                    )
                )
                continue
            canonical_el = _resolve_canonical_ref(
                cit.paragraph_ref, canonical_index
            )
            if canonical_el is None:
                issues.append(
                    CitationIssue(
                        citation_index=i,
                        citation=cit,
                        issue=(
                            f"paragraph_ref '{cit.paragraph_ref}' does not "
                            f"resolve to any element in the parsed standard."
                        ),
                    )
                )
                continue
            if not _substring_match(cit.quoted_text, canonical_el.content):
                issues.append(
                    CitationIssue(
                        citation_index=i,
                        citation=cit,
                        issue=(
                            f"quoted_text not present in canonical element "
                            f"'{canonical_el.element_id}' for paragraph "
                            f"'{cit.paragraph_ref}'."
                        ),
                    )
                )
                continue
            valid += 1
            continue
        if source.paragraph_ref and not _paragraph_refs_compatible(
            cit.paragraph_ref, source.paragraph_ref
        ):
            issues.append(
                CitationIssue(
                    citation_index=i,
                    citation=cit,
                    issue=(
                        f"paragraph_ref mismatch: citation says "
                        f"'{cit.paragraph_ref}' but source is "
                        f"'{source.paragraph_ref}'."
                    ),
                )
            )
            continue
        if not _substring_match(cit.quoted_text, source.content):
            issues.append(
                CitationIssue(
                    citation_index=i,
                    citation=cit,
                    issue=(
                        "quoted_text not substantially present in cited source content."
                    ),
                )
            )
            continue
        valid += 1

    return CitationValidationResult(
        total=len(all_citations), valid=valid, issues=issues
    )


def _build_paragraph_ref_index(
    element_index: dict[str, DocumentElement],
) -> dict[str, DocumentElement]:
    """Build a paragraph_ref → element map for canonical-ref lookups.

    When multiple elements share a paragraph_ref (e.g. a SECTION and the
    TABLE/FORMULA child it contains), prefer the SECTION because it carries
    the narrative text used for prose quotes, while TABLE/FORMULA elements
    are still reachable via their own dedicated content for row quotes.
    Mirrors the preference used by `CitationBuilder.from_elements`.
    """
    by_ref: dict[str, DocumentElement] = {}
    for el in element_index.values():
        if not el.paragraph_ref:
            continue
        key = el.paragraph_ref.upper()
        existing = by_ref.get(key)
        if existing is None or (
            existing.element_type != ElementType.SECTION
            and el.element_type == ElementType.SECTION
        ):
            by_ref[key] = el
    return by_ref


def _resolve_canonical_ref(
    paragraph_ref: str,
    by_ref: dict[str, DocumentElement],
) -> DocumentElement | None:
    """Resolve a (possibly sub-paragraph) ref to an element, walking up.

    `A-27(c)(1)` resolves to `A-27` if no sub-paragraph element exists.
    `Table M-1` is normalized to `M-1`. Returns None if no element matches.
    """
    key = paragraph_ref.upper().strip()
    if key.startswith("TABLE "):
        key = key[len("TABLE ") :].strip()
    if key in by_ref:
        return by_ref[key]
    stripped = _SUBPARA_STRIP.sub("", key).strip()
    while stripped and stripped != key:
        if stripped in by_ref:
            return by_ref[stripped]
        key = stripped
        stripped = _SUBPARA_STRIP.sub("", key).strip()
    return None


# Matches both SPES-1 refs (A-27, B-12, M-1) and real ASME refs
# (UG-27, UW-12, UCS-23, UHT-18.2, Table UG-43, 13-13(c)).
_CANONICAL_REF = re.compile(
    r"^(?:Table\s+)?(?:[A-Z]{1,5}-\d+(?:\.\d+)?(?:-\d+)?|\d{1,2}-\d+(?:\.\d+)?)"
    r"(?:\([a-z0-9]+\)(?:\(\d+\))?)?$",
    re.IGNORECASE,
)


def _looks_like_spes1_ref(ref: str) -> bool:
    """Return True if ref looks like a valid code paragraph reference.

    Despite the name (kept for backward compatibility), this now recognizes
    both SPES-1 style (A-27) and real ASME style (UG-27, UW-12, UCS-23).
    """
    return bool(_CANONICAL_REF.match(ref.strip()))


def _paragraph_refs_compatible(citation_ref: str, source_ref: str) -> bool:
    """Return True if the citation's paragraph_ref is consistent with the source.

    Sub-paragraph citations are accepted when their parent appears in the
    source (e.g. citation `A-27(c)(1)` is compatible with source `A-27`,
    and vice versa). "Table X-N" variants match "X-N" directly.

    Boundary rule: a prefix match only counts when the character after the
    prefix in the longer ref begins a sub-paragraph marker — `(` or end-of-
    string. That prevents `A-2` from spuriously matching `A-27`.
    """
    def norm(r: str) -> str:
        r = r.upper().strip()
        r = re.sub(r"^TABLE\s+", "", r)
        return r.replace(" ", "")

    a = norm(citation_ref)
    b = norm(source_ref)
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    if not longer.startswith(shorter):
        return False
    # The character right after the shorter ref must be a sub-paragraph
    # boundary. Paragraph refs like "A-27(c)(1)" use "(" as the marker; any
    # alphanumeric character at that position means we matched a different
    # paragraph entirely (e.g. "A-2" should not match "A-27").
    next_char = longer[len(shorter) : len(shorter) + 1]
    return next_char == "("
