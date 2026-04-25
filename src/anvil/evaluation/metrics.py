"""Evaluation metrics — each with an explicit mathematical definition."""

from __future__ import annotations

import re

from anvil.generation.citation_enforcer import validate_citations
from anvil.schemas.document import DocumentElement
from anvil.schemas.evaluation import GoldenExample, MetricScore
from anvil.schemas.generation import AnvilResponse, ResponseConfidence, StepKey
from anvil.schemas.retrieval import RetrievedChunk

# ----------------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------------


def retrieval_precision_at_k(
    retrieved: list[RetrievedChunk],
    expected_refs: list[str],
    k: int,
) -> MetricScore:
    """Precision@K = |{retrieved ∩ relevant}| / |retrieved[:k]|.

    A retrieved chunk counts as relevant if its paragraph_ref matches any
    expected_ref (as a prefix match so `A-27(c)(1)` counts for expected `A-27`).

    The pass threshold is computed dynamically as
    `0.8 * len(expected_refs) / k`: a query with 2 expected refs in top-10
    needs to hit at least 1.6 → 2 relevant chunks (≥20%) to pass; a query
    with 1 expected ref needs ≥8% precision (at least 1 of 10). This avoids
    penalizing specific-lookup queries where only one chunk is the target.
    """
    top = retrieved[:k]
    if not top:
        return MetricScore(name="retrieval_precision_at_k", value=0.0, passed=False)
    expected_norm = {r.upper() for r in expected_refs}
    relevant = sum(1 for c in top if _ref_matches_any(c.paragraph_ref, expected_norm))
    value = relevant / len(top)
    threshold = min(0.3, max(0.08, 0.8 * len(expected_refs) / k))
    return MetricScore(
        name="retrieval_precision_at_k",
        value=value,
        passed=value >= threshold,
        threshold=threshold,
        details={"k": k, "relevant": relevant, "total": len(top)},
    )


def retrieval_recall_at_k(
    retrieved: list[RetrievedChunk],
    expected_refs: list[str],
    k: int,
) -> MetricScore:
    """Recall@K = |retrieved ∩ relevant| / |relevant|."""
    if not expected_refs:
        return MetricScore(
            name="retrieval_recall_at_k", value=1.0, passed=True, threshold=1.0
        )
    top = retrieved[:k]
    found_refs: set[str] = set()
    for expected in expected_refs:
        if any(_ref_matches(c.paragraph_ref, expected) for c in top):
            found_refs.add(expected.upper())
    value = len(found_refs) / len(expected_refs)
    return MetricScore(
        name="retrieval_recall_at_k",
        value=value,
        passed=value >= 0.99,
        threshold=0.99,
        details={"k": k, "found": len(found_refs), "expected": len(expected_refs)},
    )


# ----------------------------------------------------------------------------
# Generation metrics
# ----------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def faithfulness(
    response: AnvilResponse,
    retrieved: list[RetrievedChunk],
) -> MetricScore:
    """Faithfulness = supported_claims / total_claims.

    A simplified, deterministic RAGAS-style check: decompose the answer into
    sentences, treat each as a claim, and count as supported if its numeric
    values appear in the retrieved context or the calculation steps.
    Non-numeric claims are supported if any substantive noun phrase appears
    in the retrieved context. This is a proxy — real RAGAS uses an NLI model.
    """
    sentences = _sentences(response.answer)
    if not sentences:
        return MetricScore(name="faithfulness", value=1.0, passed=True, threshold=0.8)

    context_text = " ".join(c.content for c in retrieved).lower()
    # Flatten all rounded forms of every step result (and input values) into
    # a single "trusted numbers" string. A sentence-level number is supported
    # if it appears either in the retrieved context or in this trusted string.
    calc_numbers_parts: list[str] = []
    for step in response.calculation_steps:
        calc_numbers_parts.append(_round_str(step.result))
        for iv in step.inputs.values():
            calc_numbers_parts.append(_round_str(iv.value))
    calc_numbers = " " + " ".join(calc_numbers_parts) + " "

    supported = 0
    for sent in sentences:
        if _sentence_supported(sent, context_text, calc_numbers):
            supported += 1
    value = supported / len(sentences)
    return MetricScore(
        name="faithfulness",
        value=value,
        passed=value >= 0.8,
        threshold=0.8,
        details={"supported": supported, "total": len(sentences)},
    )


def citation_accuracy(
    response: AnvilResponse,
    retrieved: list[RetrievedChunk],
    element_index: dict[str, DocumentElement] | None = None,
) -> MetricScore:
    """Citation Accuracy = valid_citations / total_citations.

    `element_index` is forwarded to `validate_citations` so canonical-ref
    citations (e.g. pinned-data Table M-1 quotes that retrieval did not
    surface) can have their `quoted_text` validated against the parsed
    standard. Without it, those citations fail closed — which is the
    intended safety behavior, not a measurement artifact.
    """
    result = validate_citations(response, retrieved, element_index=element_index)
    return MetricScore(
        name="citation_accuracy",
        value=result.accuracy,
        passed=result.accuracy >= 0.8,
        threshold=0.8,
        details={"valid": result.valid, "total": result.total},
    )


def calculation_correctness(
    response: AnvilResponse,
    expected_values: dict[str, float],
    tolerance: float = 0.02,
) -> MetricScore:
    """Correctness = fraction of expected numeric values matched within tolerance.

    Uses a strict numerical comparison against the calculation steps (which
    are injected by the deterministic engine). For each expected key, we
    look for a matching step by description or by the final result.
    """
    if not expected_values:
        return MetricScore(
            name="calculation_correctness", value=1.0, passed=True, threshold=1.0
        )
    matched = 0
    mismatches: dict[str, str] = {}
    step_values = _extract_named_step_values(response)
    for key, expected in expected_values.items():
        actual = step_values.get(key)
        if actual is None:
            mismatches[key] = "no matching step"
            continue
        if abs(actual - expected) <= max(tolerance, abs(expected) * tolerance):
            matched += 1
        else:
            mismatches[key] = f"expected {expected}, got {actual}"
    value = matched / len(expected_values)
    return MetricScore(
        name="calculation_correctness",
        value=value,
        passed=value == 1.0,
        threshold=1.0,
        details={"matched": matched, "total": len(expected_values), **mismatches},
    )


def refusal_calibration(
    response: AnvilResponse,
    example: GoldenExample,
) -> MetricScore:
    """Refusal calibration: 1 if (actually refused) == (should have refused), else 0."""
    refused = response.confidence == ResponseConfidence.INSUFFICIENT
    correct = refused == example.expected_refusal
    return MetricScore(
        name="refusal_calibration",
        value=1.0 if correct else 0.0,
        passed=correct,
        threshold=1.0,
        details={"expected_refusal": int(example.expected_refusal), "actual_refusal": int(refused)},
    )


def entity_grounding(
    response: AnvilResponse,
    retrieved: list[RetrievedChunk],
) -> MetricScore:
    """Every named entity (paragraph ref, material spec) in the answer should
    appear somewhere in retrieved context or be a canonical SPES-1 reference.
    """
    entities = _extract_entities(response.answer)
    if not entities:
        return MetricScore(
            name="entity_grounding", value=1.0, passed=True, threshold=1.0
        )
    context_text = " ".join(c.content for c in retrieved).upper() + " " + " ".join(
        (c.paragraph_ref or "").upper() for c in retrieved
    )
    grounded = 0
    for ent in entities:
        if ent.upper() in context_text or _is_canonical_ref(ent):
            grounded += 1
    value = grounded / len(entities)
    return MetricScore(
        name="entity_grounding",
        value=value,
        passed=value >= 0.9,
        threshold=0.9,
        details={"grounded": grounded, "total": len(entities)},
    )


def structural_completeness(
    response: AnvilResponse,
    expected_refs: list[str],
    retrieved: list[RetrievedChunk] | None = None,
) -> MetricScore:
    """For calculation / xref queries, every expected paragraph/table ref must
    appear somewhere in either (a) the response's citations, (b) the
    calculation steps' citations, or (c) the retrieved context.

    Rationale: a well-executed query demonstrates access to the required
    provenance either by citing it or (for lookup/xref queries where only
    a subset is quoted) by surfacing the relevant chunks through retrieval.
    """
    if not expected_refs:
        return MetricScore(
            name="structural_completeness",
            value=1.0,
            passed=True,
            threshold=1.0,
        )
    present_refs: set[str] = set()
    for c in response.citations:
        present_refs.add(c.paragraph_ref)
    for step in response.calculation_steps:
        present_refs.add(step.citation.paragraph_ref)
    if retrieved:
        for chunk in retrieved:
            if chunk.paragraph_ref:
                present_refs.add(chunk.paragraph_ref)
    covered = 0
    for expected in expected_refs:
        if any(_ref_matches(p, expected) for p in present_refs):
            covered += 1
    value = covered / len(expected_refs)
    return MetricScore(
        name="structural_completeness",
        value=value,
        passed=value == 1.0,
        threshold=1.0,
        details={"covered": covered, "total": len(expected_refs)},
    )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _ref_matches(actual_ref: str | None, expected_ref: str) -> bool:
    """Match two paragraph refs on sub-paragraph boundaries only.

    `A-27(c)(1)` matches expected `A-27` (next char is `(`, a sub-para
    marker) but `A-2` does NOT match `A-27` (next char is `7`, which means
    we bled into a different paragraph number). Same boundary rule as the
    citation enforcer; kept as a local helper so the evaluation module has
    no dependency on generation-layer internals.
    """
    if not actual_ref:
        return False
    a = _normalize_ref(actual_ref)
    e = _normalize_ref(expected_ref)
    if a == e:
        return True
    longer, shorter = (a, e) if len(a) > len(e) else (e, a)
    if not longer.startswith(shorter):
        return False
    next_char = longer[len(shorter) : len(shorter) + 1]
    return next_char == "("


def _normalize_ref(ref: str) -> str:
    """Normalize for matching: uppercase, strip 'TABLE ' prefix, drop spaces."""
    r = ref.upper().strip()
    if r.startswith("TABLE "):
        r = r[6:]
    return r.replace(" ", "")


def _ref_matches_any(actual_ref: str | None, expected_refs: set[str]) -> bool:
    return any(_ref_matches(actual_ref, e) for e in expected_refs)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def _sentence_supported(sent: str, context: str, calc_numbers: str) -> bool:
    s = sent.lower()
    nums = _NUMERIC_RE.findall(sent)
    if nums:
        # At least one number must appear in context or calc results
        for n in nums:
            # Pad with spaces so "14" doesn't match "140" etc.
            needle = f" {n} "
            if n in context or needle in calc_numbers:
                return True
        return False
    # Non-numeric: require ≥40% word overlap with context
    s_words = {w for w in re.findall(r"[a-z0-9\-]+", s) if len(w) > 3}
    c_words = set(re.findall(r"[a-z0-9\-]+", context.lower()))
    if not s_words:
        return True
    return len(s_words & c_words) / len(s_words) >= 0.4


def _round_str(x: float) -> str:
    """Return several rounded string forms of a float for fuzzy numeric match."""
    forms = {f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}"}
    return " ".join(forms)


_STEP_KEY_TO_METRIC_KEY: dict[StepKey, str] = {
    StepKey.ALLOWABLE_STRESS: "S_mpa",
    StepKey.JOINT_EFFICIENCY: "E",
    StepKey.APPLICABILITY_CHECK: "applicability_rhs",
    StepKey.MIN_THICKNESS: "t_min_mm",
    StepKey.DESIGN_THICKNESS: "t_design_mm",
    StepKey.NOMINAL_PLATE: "t_nominal_mm",
    StepKey.MAWP: "mawp_mpa",
}


def _extract_named_step_values(response: AnvilResponse) -> dict[str, float]:
    """Map canonical metric keys to calculation step results.

    Uses the typed `result_key` on each step — a stable enum that does NOT
    depend on the human-readable description string. This is what the golden
    dataset's `expected_values` is compared against.
    """
    out: dict[str, float] = {}
    for step in response.calculation_steps:
        metric_key = _STEP_KEY_TO_METRIC_KEY.get(step.result_key)
        if metric_key is not None:
            out[metric_key] = step.result
    # Radius is not a step result on its own — it's an input to several steps.
    for step in response.calculation_steps:
        for iv in step.inputs.values():
            if iv.symbol == "R" and "R_mm" not in out:
                out["R_mm"] = iv.value
    return out


_ENTITY_RE = re.compile(
    r"\b(?:Table\s+[A-Z]-\d+[A-Za-z]?|[A-Z]-\d+(?:\([a-z]\)(?:\(\d+\))?)?|SM-\d+(?:\s+(?:Gr|Type)\s+\w+)?)\b"
)
_CANON_RE = re.compile(r"^(?:Table\s+)?[A-M]-\d+(?:\([a-z]\)(?:\(\d+\))?)?$", re.IGNORECASE)


def _extract_entities(text: str) -> list[str]:
    return _ENTITY_RE.findall(text)


def _is_canonical_ref(text: str) -> bool:
    return bool(_CANON_RE.match(text.strip()))
