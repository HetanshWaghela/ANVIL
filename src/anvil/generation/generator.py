"""Orchestrator: RAG + refusal gate + calculation engine + LLM + citation check."""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvil import CalculationError
from anvil.generation.calculation_engine import (
    CalculationEngine,
    CalculationInputs,
    CalculationResult,
)
from anvil.generation.citation_enforcer import (
    CitationValidationResult,
    validate_citations,
)
from anvil.generation.llm_backend import FakeLLMBackend, LLMBackend
from anvil.generation.prompt_builder import build_context_prompt
from anvil.generation.refusal_gate import (
    check_calculation_evidence,
    is_calculation_query,
    should_refuse,
)
from anvil.logging_config import get_logger
from anvil.retrieval.hybrid_retriever import HybridRetriever
from anvil.schemas.generation import AnvilResponse, ResponseConfidence
from anvil.schemas.retrieval import RetrievalQuery, RetrievedChunk

log = get_logger(__name__)


@dataclass
class GenerationOutcome:
    """Result of a single end-to-end generate call."""

    response: AnvilResponse
    retrieved_chunks: list[RetrievedChunk]
    citation_validation: CitationValidationResult
    calculation: CalculationResult | None = None


class AnvilGenerator:
    """End-to-end pipeline: retrieve → gate → calculate → generate → validate."""

    def __init__(
        self,
        retriever: HybridRetriever,
        backend: LLMBackend | None = None,
        calc_engine: CalculationEngine | None = None,
    ) -> None:
        self.retriever = retriever
        self.backend = backend or FakeLLMBackend()
        self.calc_engine = calc_engine or CalculationEngine()

    async def generate(
        self,
        query: str,
        top_k: int = 10,
        calculation_inputs: CalculationInputs | None = None,
    ) -> GenerationOutcome:
        """Run the full pipeline.

        If `calculation_inputs` is provided (or can be parsed from a
        calculation query), the deterministic engine runs first and its
        output is injected into the prompt.
        """
        log.info("generate.start", query=query[:120])

        # 1) Retrieve
        rq = RetrievalQuery(text=query, top_k=top_k, enable_graph_expansion=True)
        chunks = self.retriever.retrieve(rq)

        # 2) Refusal gate
        decision = should_refuse(query, chunks)
        if decision.should_refuse:
            log.info("generate.refusal", reason=decision.reason)
            system, user = build_context_prompt(query, chunks)
            response = await self.backend.generate(
                system_prompt=system,
                user_prompt=user,
                query=query,
                retrieved_chunks=chunks,
                calculation_steps=[],
                refusal_reason=decision.reason,
            )
            return GenerationOutcome(
                response=response,
                retrieved_chunks=chunks,
                citation_validation=CitationValidationResult(total=0, valid=0, issues=[]),
            )

        # 3) Calculation (if applicable).
        #
        # Fail-loud policy: a CalculationError here means an input-driven
        # calculation was requested but could not be produced (unknown
        # material, temperature out of range, degenerate formula). We do NOT
        # silently proceed with an LLM answer that omits the number — that
        # is precisely the hallucination vector we are trying to eliminate.
        # Instead we short-circuit to a refusal response with the calc
        # engine's error as the reason, giving the caller a precise
        # explanation instead of a plausible-but-wrong free-form answer.
        calc_result: CalculationResult | None = None
        calc_steps = []
        inputs = calculation_inputs
        calc_was_requested = inputs is not None
        inputs_came_from_caller = inputs is not None
        if inputs is None and is_calculation_query(query):
            inputs = _try_parse_calculation_inputs(query)
            calc_was_requested = inputs is not None

        # Required-elements evidence check. Only apply when inputs came
        # from the (LLM-driven) NL parser — if the caller supplied explicit
        # CalculationInputs through the API, the engine uses pinned data
        # deterministically and citation integrity is enforced downstream
        # by CitationBuilder.
        if calc_was_requested and not inputs_came_from_caller:
            evidence = check_calculation_evidence(chunks)
            if evidence.should_refuse:
                log.info("generate.evidence_refusal", reason=evidence.reason)
                system, user = build_context_prompt(query, chunks)
                response = await self.backend.generate(
                    system_prompt=system,
                    user_prompt=user,
                    query=query,
                    retrieved_chunks=chunks,
                    calculation_steps=[],
                    refusal_reason=evidence.reason,
                )
                return GenerationOutcome(
                    response=response,
                    retrieved_chunks=chunks,
                    citation_validation=CitationValidationResult(
                        total=0, valid=0, issues=[]
                    ),
                )

        if inputs is not None:
            try:
                calc_result = self.calc_engine.calculate(inputs)
                calc_steps = calc_result.steps
            except CalculationError as exc:
                log.info("generate.calc_refusal", reason=str(exc))
                system, user = build_context_prompt(query, chunks)
                response = await self.backend.generate(
                    system_prompt=system,
                    user_prompt=user,
                    query=query,
                    retrieved_chunks=chunks,
                    calculation_steps=[],
                    refusal_reason=f"Calculation cannot be completed: {exc}",
                )
                return GenerationOutcome(
                    response=response,
                    retrieved_chunks=chunks,
                    citation_validation=CitationValidationResult(
                        total=0, valid=0, issues=[]
                    ),
                )

        # 4) Prompt assembly. If a calc was requested but produced no result,
        # that's an invariant violation — should have refused above.
        if calc_was_requested and calc_result is None:
            raise CalculationError(
                "Calculation was requested but no result was produced. "
                "This is an invariant violation — the engine should either "
                "return a result or raise CalculationError."
            )
        calc_summary = _summarize_calculation(calc_result) if calc_result else None
        system, user = build_context_prompt(query, chunks, calculation_summary=calc_summary)

        # 5) LLM call
        response = await self.backend.generate(
            system_prompt=system,
            user_prompt=user,
            query=query,
            retrieved_chunks=chunks,
            calculation_steps=calc_steps,
        )

        # 6) Citation validation
        validation = validate_citations(response, chunks)
        if not validation.passed and response.confidence == ResponseConfidence.HIGH:
            # Downgrade confidence if citations failed to validate
            response = response.model_copy(
                update={"confidence": ResponseConfidence.MEDIUM}
            )

        log.info(
            "generate.done",
            confidence=response.confidence.value,
            citations=len(response.citations),
            validation_issues=len(validation.issues),
        )
        return GenerationOutcome(
            response=response,
            retrieved_chunks=chunks,
            citation_validation=validation,
            calculation=calc_result,
        )


# ---- input parsing helpers -------------------------------------------------


_MATERIAL_RE = re.compile(
    r"(SM-\d+(?:\s+(?:Gr(?:ade)?|Type)\s+\w+)?)",
    re.IGNORECASE,
)
# Accept "P = 1.5 MPa", "P=1.5 MPa", or "design pressure 1.5 MPa".
_P_RE = re.compile(
    r"(?:\bP\s*=?\s*|design\s+pressure(?:\s*=)?\s*)([\d.]+)\s*MPa",
    re.IGNORECASE,
)
# Accept "inside diameter 1800 mm", "ID=1200", "ID 800 mm" — mm suffix optional.
_ID_RE = re.compile(
    r"(?:inside\s+diameter|ID)\s*=?\s*([\d.]+)(?:\s*mm)?", re.IGNORECASE
)
_OD_RE = re.compile(
    r"(?:outside\s+diameter|OD)\s*=?\s*([\d.]+)(?:\s*mm)?", re.IGNORECASE
)
# Design temperature: "T=350°C", "T 350 C", or "design temperature 350°C"
_T_RE = re.compile(
    r"(?:\bT\s*=?\s*|design\s+temperature(?:\s*=)?\s*)(\d+(?:\.\d+)?)\s*°?\s*C",
    re.IGNORECASE,
)
_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°\s*C", re.IGNORECASE)
_CA_RE = re.compile(
    r"(?:corrosion\s+allowance|CA)\s*=?\s*([\d.]+)(?:\s*mm)?", re.IGNORECASE
)
_TYPE_RE = re.compile(r"Type\s+(\d)\b", re.IGNORECASE)
_RT_RE = re.compile(
    r"(full\s+(?:RT|radiography)|spot\s+(?:RT|radiography)|no\s+(?:RT|radiography))",
    re.IGNORECASE,
)


def _try_parse_calculation_inputs(query: str) -> CalculationInputs | None:
    """Best-effort extraction of calculation parameters from a natural query.

    Returns None if essential parameters are missing — in that case the LLM
    receives retrieved context but no precomputed calculation. Explicit
    inputs (via the API) are always preferred over this parser.
    """
    try:
        P = float(_P_RE.search(query).group(1))  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        return None

    id_match = _ID_RE.search(query)
    od_match = _OD_RE.search(query)
    if id_match is None and od_match is None:
        return None

    temp_match = _T_RE.search(query) or _TEMP_RE.search(query)
    if temp_match is None:
        return None
    T = float(temp_match.group(1))

    mat_match = _MATERIAL_RE.search(query)
    if mat_match is None:
        return None
    material = re.sub(r"\s+", " ", mat_match.group(1)).strip()
    material = re.sub(r"Grade", "Gr", material, flags=re.IGNORECASE)

    # Joint type and RT level MUST be explicitly stated — no silent defaults.
    # A calculation without these is not safe to produce; the caller should
    # use the /calculate endpoint or provide them in the prompt.
    type_match = _TYPE_RE.search(query)
    if type_match is None:
        return None
    joint_type = int(type_match.group(1))
    rt_match = _RT_RE.search(query)
    if rt_match is None:
        return None
    rt_raw = rt_match.group(1).lower()
    if "full" in rt_raw:
        rt_level = "Full RT"
    elif "spot" in rt_raw:
        rt_level = "Spot RT"
    else:
        rt_level = "No RT"

    ca_match = _CA_RE.search(query)
    ca = float(ca_match.group(1)) if ca_match else 0.0

    component = (
        "cylindrical_shell_outside_radius"
        if od_match and not id_match
        else (
            "spherical_shell"
            if re.search(r"spher", query, re.IGNORECASE)
            else "cylindrical_shell"
        )
    )

    return CalculationInputs(
        component=component,  # type: ignore[arg-type]
        P_mpa=P,
        design_temp_c=T,
        material=material,
        joint_type=joint_type,
        rt_level=rt_level,
        corrosion_allowance_mm=ca,
        inside_diameter_mm=float(id_match.group(1)) if id_match else None,
        outside_diameter_mm=float(od_match.group(1)) if od_match else None,
    )


def _summarize_calculation(result: CalculationResult) -> str:
    # Look up the formula-bearing step by its stable result_key rather than
    # by positional index, so reordering steps does not silently break the
    # prompt summary. If the step is missing entirely that is an invariant
    # violation — the engine must not emit a result without its primary
    # formula step — so we raise instead of emitting an empty formula
    # string that would quietly produce a subtly wrong LLM prompt.
    from anvil.schemas.generation import StepKey

    formula_step = next(
        (s for s in result.steps if s.result_key == StepKey.MIN_THICKNESS),
        None,
    )
    if formula_step is None:
        raise CalculationError(
            "CalculationResult is missing the MIN_THICKNESS step. This is an "
            "invariant violation: the calculation engine must always emit the "
            "primary thickness formula. Refusing to produce a degraded summary."
        )
    formula_text = formula_step.formula
    lines = [
        f"formula = {result.formula_ref}: {formula_text}",
        f"S (allowable stress) = {result.S_mpa} MPa",
        f"E (joint efficiency) = {result.E}",
        f"R = {result.R_mm} mm",
        f"applicability: {result.applicability_lhs} ≤ {result.applicability_rhs} → {'OK' if result.applicability_ok else 'FAIL'}",
        f"t_min = {result.t_min_mm:.2f} mm",
        f"t_design (with CA) = {result.t_design_mm:.2f} mm",
        f"t_nominal (plate) = {result.t_nominal_mm} mm",
        f"MAWP = {result.mawp_mpa:.3f} MPa",
    ]
    return "\n".join(lines)
