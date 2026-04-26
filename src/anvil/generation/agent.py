"""M6 — Agentic tool-calling loop.

`AnvilAgent.run(query)` is an alternative to `AnvilGenerator.generate`.
Where the generator uses a fixed retrieve → calculate → generate
pipeline, the agent loop lets a decider (real LLM or a scripted stub)
pick tools turn-by-turn until it issues `finalize` or hits the
budget.

Design contract — what makes this defensible vs. "let an LLM
free-fly":

1. **Mandatory budget.** `AgentBudget` is non-optional; max_steps
   defaults to 8. A runaway loop on real-NIM costs at most
   ceil(8 * avg_call_latency) seconds + tokens.
2. **Fail-soft tools.** Tool adapters never raise; errors land in
   `ToolResult.error` so the loop can react (retry / refuse) instead
   of crashing 27 examples into a 30-example eval.
3. **Hard error budget.** After `max_tool_errors` cumulative
   tool failures, the loop terminates with a refusal — same as if
   the step budget ran out.
4. **Provenance.** Every (call, result) pair lands in the transcript;
   `RunLogger` persists it next to per_example.json so a reviewer can
   replay the agent's decisions deterministically.

This module deliberately does NOT touch citation validation or the
post-processing that `AnvilGenerator` runs — those are concerns of the
*final* `AnvilResponse`, which the decider produces directly via
`FinalAnswer`. Down-stream metrics treat the agent's output the same
as the generator's output.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anvil import RetryableGenerationError
from anvil.generation.agent_backend import AgentBackend
from anvil.generation.agent_tools import ToolRegistry
from anvil.logging_config import get_logger
from anvil.schemas.agent import (
    AgentBudget,
    AgentStep,
    AgentTranscript,
    TerminationReason,
    ToolCall,
    ToolResult,
)
from anvil.schemas.generation import (
    AnvilResponse,
    CalculationStep,
    Citation,
    ResponseConfidence,
)
from anvil.schemas.retrieval import RetrievedChunk

if TYPE_CHECKING:
    from anvil.generation.generator import AnvilGenerator

log = get_logger(__name__)


@dataclass
class AgentOutcome:
    """Result of one `AnvilAgent.run(query)` call.

    Mirrors `GenerationOutcome` so eval framework code can switch
    between agent / non-agent modes by just swapping the producer.
    """

    response: AnvilResponse
    transcript: AgentTranscript


# ---------------------------------------------------------------------------
# Refusal-shaped fallback used when the loop terminates without `finalize`.
# ---------------------------------------------------------------------------


def _budget_refusal(query: str, reason: str) -> AnvilResponse:
    """Produce a refusal response when the loop exits without a final answer.

    We deliberately reuse the existing refusal contract (confidence =
    INSUFFICIENT, populated `refusal_reason`) so all downstream
    metrics treat budget-exhaustion the same as a deliberate refusal —
    no special-casing needed in `refusal_calibration` or
    `faithfulness`.
    """
    return AnvilResponse(
        query=query,
        answer=(
            "Agent loop terminated without producing a final answer. "
            "See refusal_reason for details."
        ),
        citations=[],
        calculation_steps=[],
        confidence=ResponseConfidence.INSUFFICIENT,
        refusal_reason=reason,
        retrieved_context_ids=[],
    )


def _calculation_autofinalize(query: str, output: dict[str, object]) -> AnvilResponse:
    """Finalize immediately after a successful deterministic calculation.

    This keeps the agentic path from burning extra LLM turns after the trusted
    calculation engine has already produced the answer, steps, and citations.
    It is not answer hardcoding: all numbers and provenance come from the live
    `calculate` tool output produced by `CalculationEngine`.
    """
    steps_raw = output.get("calculation_steps")
    citations_raw = output.get("citations")
    if not isinstance(steps_raw, list):
        raise ValueError("calculate output missing calculation_steps list.")
    if not isinstance(citations_raw, list):
        raise ValueError("calculate output missing citations list.")

    steps = [CalculationStep.model_validate(item) for item in steps_raw]
    citations = [Citation.model_validate(item) for item in citations_raw]

    t_min = _required_float(output, "t_min_mm")
    t_design = _required_float(output, "t_design_mm")
    t_nominal = int(_required_float(output, "t_nominal_mm"))
    mawp = _required_float(output, "mawp_mpa")

    return AnvilResponse(
        query=query,
        answer=(
            f"Minimum required thickness t_min = {t_min:.2f} mm; "
            f"design thickness with corrosion allowance = {t_design:.2f} mm; "
            f"selected nominal plate = {t_nominal} mm; "
            f"MAWP back-calculated from the selected plate = {mawp:.3f} MPa."
        ),
        citations=citations,
        calculation_steps=steps,
        confidence=ResponseConfidence.HIGH,
        retrieved_context_ids=sorted({c.source_element_id for c in citations}),
    )


def _is_direct_lookup_query(query: str, kind: str) -> bool:
    """Return true when a pinned lookup result directly answers the query.

    The agent may call `pinned_lookup` as an intermediate step before a
    calculation. In that case auto-finalizing would be wrong. This narrow
    intent check only fires for direct table lookups and deliberately excludes
    calculation/MAWP/thickness requests.
    """
    q = query.lower()
    if any(token in q for token in ("calculate", "compute", "thickness", "mawp")):
        return False
    if kind == "allowable_stress":
        return "allowable stress" in q or "stress" in q
    if kind == "joint_efficiency":
        return "joint efficiency" in q or "efficiency" in q
    if kind == "material":
        return any(
            phrase in q
            for phrase in (
                "product form",
                "max temperature",
                "maximum temperature",
                "p-number",
                "p number",
                "group number",
                "material",
                "listed for",
            )
        )
    return False


def _pinned_lookup_autofinalize(
    query: str,
    output: dict[str, object],
    registry: ToolRegistry,
) -> AnvilResponse | None:
    """Finalize direct pinned-table lookup questions after a successful tool call.

    This mirrors calculation auto-finalization: once a deterministic trusted
    tool has returned the requested table value plus a source row, an extra LLM
    turn is avoidable provider load. It does not invent values; all numbers and
    citations come from pinned data and the parsed standard citation index.
    """
    if output.get("found") is not True:
        return None
    kind = str(output.get("kind") or "")
    if not _is_direct_lookup_query(query, kind):
        return None

    key = str(output.get("key") or "")
    citations: list[Citation]
    answer: str
    try:
        if kind == "allowable_stress":
            stress = _required_float(output, "allowable_stress_mpa")
            temp_c = _required_float(output, "temp_c")
            citations = [registry.calc_engine._citations.for_material(key)]  # noqa: SLF001
            answer = (
                f"The allowable stress of {key} at {temp_c:g}°C is "
                f"{stress:g} MPa per Table M-1."
            )
        elif kind == "joint_efficiency":
            efficiency = _required_float(output, "joint_efficiency")
            joint_type = int(_required_float(output, "joint_type"))
            rt_level = str(output.get("rt_level") or "").strip()
            citations = [
                registry.calc_engine._citations.for_joint_efficiency(joint_type)  # noqa: SLF001
            ]
            answer = (
                f"The joint efficiency for Type {joint_type} with {rt_level} "
                f"is E = {efficiency:g} per Table B-12."
            )
        elif kind == "material":
            citations = [registry.calc_engine._citations.for_material(key)]  # noqa: SLF001
            details: list[str] = []
            product_form = output.get("product_form")
            max_temp_c = output.get("max_temp_c")
            if isinstance(product_form, str) and product_form:
                details.append(f"product form {product_form}")
            if isinstance(max_temp_c, int | float):
                details.append(f"maximum listed temperature {max_temp_c:g}°C")
            detail_text = ", ".join(details) if details else "a pinned material row"
            answer = f"{key} is listed in Table M-1 with {detail_text}."
        else:
            return None
    except Exception as exc:  # noqa: BLE001 - citation lookup must fail closed
        log.warning("agent.pinned_autofinalize.failed", error=repr(exc), kind=kind)
        return None

    return AnvilResponse(
        query=query,
        answer=answer,
        citations=citations,
        calculation_steps=[],
        confidence=ResponseConfidence.HIGH,
        retrieved_context_ids=[c.source_element_id for c in citations],
    )


def _required_float(output: dict[str, object], key: str) -> float:
    value = output.get(key)
    if not isinstance(value, int | float | str):
        raise ValueError(f"calculate output missing numeric {key}.")
    return float(value)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


# Saturation threshold: after this many retrieve_context+graph_lookup calls
# without finalizing AND without firing a deterministic tool (pinned_lookup
# or calculate), the host hands off to the trusted synthesizer. This caps
# the runaway-retrieval failure mode (transcripts showed 8/8 retrieve calls
# in a loop on lookup and cross_reference queries) without blocking genuine
# multi-hop research that legitimately needs more steps.
_RETRIEVAL_SATURATION_THRESHOLD = 2


def _aggregate_retrieved_chunks(steps: list[AgentStep]) -> list[RetrievedChunk]:
    """Union of every retrieve_context chunk in the transcript.

    Mirrors the deduplication policy in `agent_runner._aggregate_retrieval`
    so the saturation auto-finalize sees exactly the chunk set that the
    metric layer will score.
    """
    seen: OrderedDict[str, RetrievedChunk] = OrderedDict()
    for step in steps:
        if step.call.name != "retrieve_context" or not step.result.ok:
            continue
        for raw in step.result.output.get("chunks", []):
            eid = raw.get("element_id")
            if not eid or eid in seen:
                continue
            seen[eid] = RetrievedChunk(
                element_id=eid,
                content=raw.get("content", ""),
                element_type=raw.get("element_type", "paragraph"),
                paragraph_ref=raw.get("paragraph_ref"),
                page_number=int(raw.get("page_number") or 1),
                score=float(raw.get("score", 0.0)),
                retrieval_source=raw.get("retrieval_source", "agent"),
            )
    return list(seen.values())


def _retrieval_saturation_triggered(steps: list[AgentStep]) -> bool:
    """True when the agent has gathered evidence but is looping on retrieval.

    Conditions:
      1. ≥ `_RETRIEVAL_SATURATION_THRESHOLD` successful retrieve_context or
         graph_lookup calls.
      2. No successful pinned_lookup or calculate calls (those have their
         own deterministic auto-finalize paths and must not be pre-empted).
      3. At least one chunk has been gathered.

    The threshold is intentionally low (2). The fixed pipeline does well on
    these queries with a single retrieval; an agent that has called retrieval
    twice without finalizing is effectively confused and will benefit from
    trusted host-side synthesis over the chunks it has already chosen.
    """
    n_retrieval_calls = sum(
        1
        for s in steps
        if s.call.name in ("retrieve_context", "graph_lookup") and s.result.ok
    )
    if n_retrieval_calls < _RETRIEVAL_SATURATION_THRESHOLD:
        return False
    has_deterministic_tool = any(
        s.call.name in ("pinned_lookup", "calculate") and s.result.ok
        for s in steps
    )
    if has_deterministic_tool:
        return False
    return any(
        s.call.name == "retrieve_context"
        and s.result.ok
        and s.result.output.get("chunks")
        for s in steps
    )


class AnvilAgent:
    """Orchestrate the tool-calling loop."""

    def __init__(
        self,
        decider: AgentBackend,
        registry: ToolRegistry,
        budget: AgentBudget | None = None,
        synthesizer: AnvilGenerator | None = None,
    ) -> None:
        self.decider = decider
        self.registry = registry
        self.budget = budget or AgentBudget()
        # Optional trusted host-side answer synthesizer. When attached, the
        # loop falls through to `synthesizer.synthesize_from_chunks(...)`
        # after retrieval saturation, producing a structured AnvilResponse
        # with citation enforcement instead of letting the LLM loop until
        # the step budget is exhausted. Backwards compatible: passing None
        # preserves the previous "only finalize via LLM or auto-finalize on
        # pinned/calc" behavior, which existing unit tests depend on.
        self.synthesizer = synthesizer

    async def run(self, query: str) -> AgentOutcome:
        """Execute the loop until finalize, budget exhaustion, or error limit."""
        log.info("agent.run.start", query_preview=query[:80])
        t0 = time.perf_counter()
        steps: list[AgentStep] = []
        n_errors = 0
        manifest = self.registry.manifest()

        termination: TerminationReason | None = None
        final_response: AnvilResponse | None = None

        while True:
            # ---- step budget ------------------------------------------------
            if len(steps) >= self.budget.max_steps:
                termination = TerminationReason(
                    kind="budget_steps_exhausted",
                    detail=(
                        f"Loop reached max_steps={self.budget.max_steps} "
                        f"without finalizing. Tools used: "
                        f"{[s.call.name for s in steps]}."
                    ),
                )
                break

            # ---- decide -----------------------------------------------------
            try:
                decision = await self.decider.decide(query, steps, manifest)
            except RetryableGenerationError:
                log.warning("agent.decider.retryable_error")
                raise
            except Exception as exc:  # noqa: BLE001 — decider is user-supplied
                log.warning("agent.decider.error", error=repr(exc))
                termination = TerminationReason(
                    kind="decider_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                break

            # ---- finalize? --------------------------------------------------
            if decision.final is not None:
                final_response = decision.final.response
                termination = TerminationReason(
                    kind="finalized",
                    detail=f"Finalized after {len(steps)} tool calls.",
                )
                log.info(
                    "agent.run.finalized",
                    n_steps=len(steps),
                    confidence=final_response.confidence.value,
                )
                break

            # ---- execute tool ----------------------------------------------
            assert decision.tool_call is not None  # XOR invariant on AgentDecision
            call: ToolCall = decision.tool_call

            if call.name == "finalize":
                try:
                    raw_response = call.arguments["response"]
                    final_response = AnvilResponse.model_validate(raw_response)
                    termination = TerminationReason(
                        kind="finalized",
                        detail=(
                            "Finalized via finalize tool call after "
                            f"{len(steps)} tool calls."
                        ),
                    )
                    log.info(
                        "agent.run.finalized",
                        n_steps=len(steps),
                        confidence=final_response.confidence.value,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - model output is untrusted
                    log.warning(
                        "agent.tool.finalize_invalid",
                        error=repr(exc),
                    )
                    steps.append(
                        AgentStep(
                            step_index=len(steps),
                            call=call,
                            result=ToolResult(
                                name=call.name,
                                arguments=call.arguments,
                                error=f"Invalid finalize payload: {exc}",
                            ),
                        )
                    )
                    n_errors += 1
                    if n_errors > self.budget.max_tool_errors:
                        termination = TerminationReason(
                            kind="tool_error_limit",
                            detail=(
                                f"Cumulative tool errors ({n_errors}) exceeded "
                                f"max_tool_errors={self.budget.max_tool_errors}."
                            ),
                        )
                        break
                    continue

            # Compliance guardrail: a calculation-only transcript is not enough
            # evidence for an auditable answer. If the LLM jumps straight to the
            # deterministic calculation tool, hydrate the transcript with one
            # retrieval step first. This is not answer hardcoding — it records
            # the standard evidence the calculation answer is conditioned on.
            if call.name == "calculate" and not any(
                s.call.name == "retrieve_context" for s in steps
            ):
                retrieve_call = ToolCall(
                    name="retrieve_context",
                    arguments={"query": query, "top_k": 10},
                )
                log.info(
                    "agent.tool.auto_retrieve_before_calculation",
                    step_index=len(steps),
                    tool=retrieve_call.name,
                    arg_keys=list(retrieve_call.arguments),
                )
                retrieve_result = self.registry.execute(retrieve_call)
                steps.append(
                    AgentStep(
                        step_index=len(steps),
                        call=retrieve_call,
                        result=retrieve_result,
                    )
                )
                if not retrieve_result.ok:
                    n_errors += 1
                    if n_errors > self.budget.max_tool_errors:
                        termination = TerminationReason(
                            kind="tool_error_limit",
                            detail=(
                                f"Cumulative tool errors ({n_errors}) exceeded "
                                f"max_tool_errors={self.budget.max_tool_errors}."
                            ),
                        )
                        break

            log.info(
                "agent.tool.call",
                step_index=len(steps),
                tool=call.name,
                arg_keys=list(call.arguments),
            )
            result = self.registry.execute(call)
            steps.append(AgentStep(step_index=len(steps), call=call, result=result))
            if call.name == "pinned_lookup" and result.ok:
                final_response = _pinned_lookup_autofinalize(
                    query,
                    result.output,
                    self.registry,
                )
                if final_response is not None:
                    termination = TerminationReason(
                        kind="finalized",
                        detail=("Auto-finalized after successful deterministic pinned lookup."),
                    )
                    log.info(
                        "agent.run.auto_finalized_after_pinned_lookup",
                        n_steps=len(steps),
                        confidence=final_response.confidence.value,
                    )
                    break
            if call.name == "calculate" and result.ok and result.output.get("ok") is True:
                final_response = _calculation_autofinalize(query, result.output)
                termination = TerminationReason(
                    kind="finalized",
                    detail=("Auto-finalized after successful deterministic calculation tool call."),
                )
                log.info(
                    "agent.run.auto_finalized_after_calculation",
                    n_steps=len(steps),
                    confidence=final_response.confidence.value,
                )
                break
            # Retrieval saturation: the agent has gathered evidence but is
            # looping on retrieval. Hand off to the trusted synthesizer to
            # produce a structured AnvilResponse with full citation
            # enforcement, using the chunks the agent itself chose. This is
            # not a "give up" path — it is the same answer-synthesis code
            # the fixed pipeline uses, executed over the agent-curated
            # context. Only triggers when a synthesizer is attached, so the
            # standalone agent semantics are unchanged for callers that
            # don't opt in.
            if (
                self.synthesizer is not None
                and call.name in ("retrieve_context", "graph_lookup")
                and result.ok
                and _retrieval_saturation_triggered(steps)
            ):
                aggregated = _aggregate_retrieved_chunks(steps)
                log.info(
                    "agent.run.retrieval_saturation_synthesize",
                    n_steps=len(steps),
                    n_chunks=len(aggregated),
                )
                synth_outcome = await self.synthesizer.synthesize_from_chunks(
                    query, aggregated
                )
                final_response = synth_outcome.response
                termination = TerminationReason(
                    kind="finalized",
                    detail=(
                        f"Auto-finalized via host synthesizer after "
                        f"{len(steps)} tool calls — retrieval saturation "
                        f"on {len(aggregated)} unique chunks."
                    ),
                )
                log.info(
                    "agent.run.auto_finalized_after_retrieval_saturation",
                    n_steps=len(steps),
                    n_chunks=len(aggregated),
                    confidence=final_response.confidence.value,
                )
                break
            if not result.ok:
                n_errors += 1
                if n_errors > self.budget.max_tool_errors:
                    termination = TerminationReason(
                        kind="tool_error_limit",
                        detail=(
                            f"Cumulative tool errors ({n_errors}) exceeded "
                            f"max_tool_errors={self.budget.max_tool_errors}."
                        ),
                    )
                    break

        # --- assemble output -------------------------------------------------
        if final_response is None:
            final_response = _budget_refusal(
                query,
                reason=(
                    termination.detail
                    if termination is not None
                    else "Agent terminated unexpectedly."
                ),
            )
        # `termination` is guaranteed non-None: every path that exits
        # the loop sets it. mypy can't see this so we narrow explicitly.
        assert termination is not None

        transcript = AgentTranscript(
            query=query,
            steps=steps,
            final_response=final_response if termination.kind == "finalized" else None,
            termination=termination,
            total_duration_ms=(time.perf_counter() - t0) * 1000,
            n_tool_calls=len(steps),
            n_tool_errors=n_errors,
        )
        log.info("agent.run.done", **transcript.to_summary())
        return AgentOutcome(response=final_response, transcript=transcript)


__all__ = ["AgentOutcome", "AnvilAgent"]
