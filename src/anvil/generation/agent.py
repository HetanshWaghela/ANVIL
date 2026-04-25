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
from dataclasses import dataclass

from anvil.generation.agent_backend import AgentBackend
from anvil.generation.agent_tools import ToolRegistry
from anvil.logging_config import get_logger
from anvil.schemas.agent import (
    AgentBudget,
    AgentStep,
    AgentTranscript,
    TerminationReason,
    ToolCall,
)
from anvil.schemas.generation import AnvilResponse, ResponseConfidence

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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class AnvilAgent:
    """Orchestrate the tool-calling loop."""

    def __init__(
        self,
        decider: AgentBackend,
        registry: ToolRegistry,
        budget: AgentBudget | None = None,
    ) -> None:
        self.decider = decider
        self.registry = registry
        self.budget = budget or AgentBudget()

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
            log.info(
                "agent.tool.call",
                step_index=len(steps),
                tool=call.name,
                arg_keys=list(call.arguments),
            )
            result = self.registry.execute(call)
            steps.append(
                AgentStep(step_index=len(steps), call=call, result=result)
            )
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
