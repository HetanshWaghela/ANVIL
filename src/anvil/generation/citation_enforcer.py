"""Post-generation citation validation.

For every citation in an `AnvilResponse`, verify:

  1. The `source_element_id` exists in the retrieved context.
  2. The `quoted_text` actually appears in (or is strongly similar to) the
     cited source content.
  3. The `paragraph_ref` matches the source element's paragraph.

If any citation fails validation, the response should be downgraded to
`MEDIUM` confidence (or rejected) by the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvil.schemas.generation import AnvilResponse, Citation
from anvil.schemas.retrieval import RetrievedChunk


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
) -> CitationValidationResult:
    """Check every citation against the retrieved context."""
    chunks_by_id: dict[str, RetrievedChunk] = {c.element_id: c for c in retrieved_chunks}
    issues: list[CitationIssue] = []

    all_citations: list[Citation] = list(response.citations)
    for step in response.calculation_steps:
        all_citations.append(step.citation)
        for iv in step.inputs.values():
            if iv.citation is not None:
                all_citations.append(iv.citation)

    valid = 0
    for i, cit in enumerate(all_citations):
        source = chunks_by_id.get(cit.source_element_id)
        if source is None:
            # Accept pinned-data citations that reference canonical paragraphs/tables
            # even when they aren't present in retrieval — but only for valid refs.
            if cit.paragraph_ref and _looks_like_spes1_ref(cit.paragraph_ref):
                valid += 1
                continue
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


_CANONICAL_REF = re.compile(
    r"^(?:Table\s+)?[A-M]-\d+(?:\([a-z]\)(?:\(\d+\))?)?$",
    re.IGNORECASE,
)


def _looks_like_spes1_ref(ref: str) -> bool:
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
