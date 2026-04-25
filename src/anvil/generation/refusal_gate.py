"""Pre-generation refusal gate.

If the retrieved context has low relevance to the query OR critical pieces
required for a calculation are missing, we refuse BEFORE invoking the LLM.
This eliminates hallucinated stress values at their root cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvil.parsing.section_linker import MATERIAL_SPEC_PATTERN
from anvil.pinned import get_material
from anvil.schemas.retrieval import RetrievedChunk

# Generic "X Grade/Gr/Type Y" pattern — catches mentions of non-SPES-1
# material specs like "Titanium Grade 5" or "Inconel 625" so we can refuse
# cleanly instead of returning a spurious answer.
_GENERIC_MATERIAL_PATTERN = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:Grade|Gr|Type)\s+\w+\b"
)
_UNSUPPORTED_MATERIAL_PATTERN = re.compile(
    r"\b(SA-\d+(?:\s+(?:Gr(?:ade)?|Type)\s+\w+)?|Inconel\s+\d+)\b",
    re.IGNORECASE,
)

RELEVANCE_THRESHOLD: float = 0.05


@dataclass
class RefusalDecision:
    """Outcome of the refusal gate."""

    should_refuse: bool
    reason: str | None = None


# A calculation query asks the system to produce a number (thickness, MAWP,
# ...). Purely conceptual questions about formulas (e.g. "what inputs does
# A-27(c)(1) need?") are NOT calculation queries and should route to
# retrieval-only answering. We require both a calc verb AND a domain noun
# OR an explicit numeric parameter to avoid false positives on questions
# like "what is joint efficiency?".
_CALC_VERB = re.compile(
    r"\b(calculate|compute|determine|find|back-?calculate)\b", re.IGNORECASE
)
_CALC_NOUN = re.compile(
    r"\b(thickness|mawp|(?:design\s+)?pressure|wall|plate)\b", re.IGNORECASE
)
_NUMERIC_PARAM = re.compile(
    r"\b(?:P|T|R|ID|OD|CA)\s*=?\s*[\d.]+|"
    r"\b[\d.]+\s*(?:MPa|mm|°C)\b",
    re.IGNORECASE,
)
_TEMPERATURE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*C", re.IGNORECASE)
_PRESSURE_PARAM = re.compile(
    r"(?:\bP\s*=?\s*|design\s+pressure(?:\s*=)?\s*)[\d.]+\s*MPa",
    re.IGNORECASE,
)
_DIAMETER_PARAM = re.compile(
    r"(?:inside\s+diameter|outside\s+diameter|ID|OD)\s*=?\s*[\d.]+",
    re.IGNORECASE,
)
_JOINT_TYPE_PARAM = re.compile(r"\bType\s+\d\b", re.IGNORECASE)
_RT_PARAM = re.compile(
    r"\b(full\s+(?:RT|radiography)|spot\s+(?:RT|radiography)|no\s+(?:RT|radiography))\b",
    re.IGNORECASE,
)


def is_calculation_query(query: str) -> bool:
    """True when the query asks the system to *produce a number*, not merely
    discuss a formula or look up a single value. Heuristic but conservative:
    we require a calculation verb plus either a domain noun or an explicit
    numeric parameter (`P=1.5 MPa`, `ID=1800 mm`, …).
    """
    has_verb = bool(_CALC_VERB.search(query))
    if not has_verb:
        return False
    return bool(_CALC_NOUN.search(query) or _NUMERIC_PARAM.search(query))


# Required-element categories for calculation queries. Per spec §Refusal
# Gate — if any of these are missing from retrieval, refuse: we cannot
# responsibly compute thickness without the formula paragraph, the stress
# lookup, and the joint-efficiency lookup all surfaced in the context.
# Each entry is (human-readable label, tuple of acceptable paragraph_refs).
_REQUIRED_REFS_FOR_CALC: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("thickness formula (A-27)", ("A-27",)),
    ("allowable-stress table (M-1)", ("M-1", "Table M-1")),
    ("joint-efficiency table (B-12)", ("B-12", "Table B-12")),
)


def _chunk_covers(chunk: RetrievedChunk, acceptable: tuple[str, ...]) -> bool:
    """Return True if the chunk's paragraph_ref is (a prefix match for)
    one of the acceptable refs. Mirrors the same boundary rule used by
    the citation enforcer so `A-27(c)(1)` counts as covering `A-27`.
    """
    ref = (chunk.paragraph_ref or "").upper().replace(" ", "")
    if ref.startswith("TABLE"):
        ref = ref[5:]
    for a in acceptable:
        aa = a.upper().replace(" ", "")
        if aa.startswith("TABLE"):
            aa = aa[5:]
        if ref == aa:
            return True
        # Prefix match on sub-paragraph boundaries only
        longer, shorter = (ref, aa) if len(ref) > len(aa) else (aa, ref)
        if longer.startswith(shorter) and longer[len(shorter) : len(shorter) + 1] in ("", "("):
            return True
    return False


def _mentions_material(query: str) -> str | None:
    m = MATERIAL_SPEC_PATTERN.search(query)
    if not m:
        return None
    return m.group(0)


def _extract_design_temp(query: str) -> float | None:
    m = _TEMPERATURE_PATTERN.search(query)
    if not m:
        return None
    return float(m.group(1))


def should_refuse(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> RefusalDecision:
    """Determine if the system should refuse the query.

    Refuses when:
      1. No chunks retrieved, or max relevance score is below the threshold.
      2. The query mentions an unsupported material spec (`Titanium Grade 5`,
         `SA-516 Gr 70`, `Inconel 625`) that is not an SM- SPES-1 spec.
      3. The query mentions an SM- spec not present in pinned data.
      4. The query mentions a design temperature outside the tabulated
         range for the mentioned material (per M-23).
      5. The query is a calculation query but one of the required evidence
         categories (formula paragraph, stress table, joint efficiency
         table) is missing from the retrieved context.

    Rules (1)–(4) catch obvious OOD / hallucination-bait queries before the
    LLM is invoked. Rule (5) is the compliance-grade requirement: refuse
    to produce a numeric answer if we can't cite every piece of the
    calculation chain.
    """
    # 1) Relevance threshold
    max_relevance = max((c.score for c in retrieved_chunks), default=0.0)
    if not retrieved_chunks or max_relevance < relevance_threshold:
        return RefusalDecision(
            should_refuse=True,
            reason=(
                "No sufficiently relevant content found in the standard "
                f"(max relevance {max_relevance:.3f} < threshold "
                f"{relevance_threshold:.3f})."
            ),
        )

    # 2a) Material mention that is NOT a SM- spec (e.g., "Titanium Grade 5",
    # "SA-516 Gr 70", "Inconel 625") — SPES-1 only covers SM- specs in M-1.
    sm_mat = _mentions_material(query)
    unsupported_match = _UNSUPPORTED_MATERIAL_PATTERN.search(query)
    generic_match = _GENERIC_MATERIAL_PATTERN.search(query)
    if sm_mat is None and unsupported_match is not None:
        material_text = unsupported_match.group(0)
        return RefusalDecision(
            should_refuse=True,
            reason=(
                f"Material '{material_text}' is not an SM- "
                "specification covered by SPES-1 Table M-1. Please verify "
                "the specification or consult the supplementary material tables."
            ),
        )
    if sm_mat is None and generic_match is not None:
        material_text = generic_match.group(0)
        return RefusalDecision(
            should_refuse=True,
            reason=(
                f"Material '{material_text}' is not an SM- "
                "specification covered by SPES-1 Table M-1. Please verify "
                "the specification or consult the supplementary material tables."
            ),
        )

    # 2b) Unknown material (only check if explicitly mentioned)
    mat_text = sm_mat
    if mat_text is not None:
        # Normalize 'SM-516 Grade 70' → 'SM-516 Gr 70'
        norm = re.sub(r"Grade", "Gr", mat_text, flags=re.IGNORECASE)
        norm = re.sub(r"\s+", " ", norm).strip()
        mat = get_material(norm)
        if mat is None:
            return RefusalDecision(
                should_refuse=True,
                reason=(
                    f"Material '{mat_text}' is not in the SPES-1 pinned "
                    "materials table. Please verify the specification or "
                    "add the material to Table M-1."
                ),
            )
        # 3) Out-of-range temperature for this material
        temp = _extract_design_temp(query)
        if temp is not None and temp > mat.max_temp_c:
            return RefusalDecision(
                should_refuse=True,
                reason=(
                    f"Design temperature {temp}°C exceeds the maximum "
                    f"tabulated temperature for {norm} "
                    f"({mat.max_temp_c}°C per M-23)."
                ),
            )

    if is_calculation_query(query):
        missing_inputs: list[str] = []
        if not _PRESSURE_PARAM.search(query):
            missing_inputs.append("design pressure")
        if not _DIAMETER_PARAM.search(query):
            missing_inputs.append("inside or outside diameter")
        if sm_mat is None:
            missing_inputs.append("supported SM material")
        if not _JOINT_TYPE_PARAM.search(query):
            missing_inputs.append("joint type")
        if not _RT_PARAM.search(query):
            missing_inputs.append("radiography extent")
        if missing_inputs:
            return RefusalDecision(
                should_refuse=True,
                reason=(
                    "Cannot complete calculation: missing required input(s): "
                    f"{', '.join(missing_inputs)}."
                ),
            )

    return RefusalDecision(should_refuse=False)


def check_calculation_evidence(
    retrieved_chunks: list[RetrievedChunk],
) -> RefusalDecision:
    """Compliance-grade required-elements check per spec §Refusal Gate.

    A calculation-grade answer is only defensible when the retrieval layer
    has surfaced every piece of the evidence chain: the formula paragraph,
    the allowable-stress table, and the joint-efficiency table. If any is
    missing, refuse — we cannot cite the computation honestly.

    This is separate from `should_refuse()` because the generator only
    wants to apply this check when the calculation's inputs would
    otherwise be *derived from the LLM's reading of the retrieved
    context*. When an API caller passes explicit `CalculationInputs`, the
    engine uses pinned data directly and retrieval is used only to
    surface the citation text — a different failure mode with different
    containment logic, handled by the `CitationBuilder` (which raises
    `CalculationError` if it cannot find a real citation quote).
    """
    missing: list[str] = []
    for label, acceptable in _REQUIRED_REFS_FOR_CALC:
        if not any(_chunk_covers(c, acceptable) for c in retrieved_chunks):
            missing.append(label)
    if missing:
        return RefusalDecision(
            should_refuse=True,
            reason=(
                "Cannot complete calculation: retrieval did not surface "
                f"required evidence for {', '.join(missing)}. "
                "Broaden the query or verify the ingestion pipeline."
            ),
        )
    return RefusalDecision(should_refuse=False)
