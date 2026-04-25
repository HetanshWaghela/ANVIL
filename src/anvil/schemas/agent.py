"""Pydantic models for the M6 agentic tool-calling loop.

The agent loop is an *alternative* generation strategy to
`AnvilGenerator.generate`: instead of a fixed retrieve → calculate →
generate pipeline, the LLM (or a scripted decider, in tests) picks
tools turn by turn until it issues a `finalize` call or exhausts the
budget.

Why a separate code path:

1. **Empirical comparison.** The headline plan claims the
   "tool-calling loop" is a pluggable alternative; M6 makes that
   claim measurable. We can run both pipelines on the same golden
   dataset and report whether the agent loop wins on any metric.
2. **Provenance.** Every tool call + result is recorded in
   `AgentTranscript`, which the run-logger persists next to the rest
   of the run artifacts. A reviewer can replay the agent's reasoning.
3. **Budget enforcement.** Free-tier NIM has a hard RPM cap; an
   uncapped agent loop is a footgun. `AgentBudget` enforces a
   per-query ceiling on calls AND tokens, fail-closed.

This module intentionally has no runtime dependencies beyond Pydantic
+ the existing schemas — `agent.py` and `agent_tools.py` import these
shapes, never the other way around.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from anvil.schemas.generation import AnvilResponse

# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One tool invocation the agent (or scripted decider) wants to make.

    `name` must match a key in the `ToolRegistry`. `arguments` is a
    free-form JSON-compatible dict; per-tool argument validation is
    performed by the tool itself when it executes — that keeps the
    Pydantic schema for `ToolCall` tool-agnostic and lets us add new
    tools without bumping a versioned union.
    """

    name: str = Field(..., description="Tool name; must be in the registry.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-compatible kwargs for the tool.",
    )


class ToolResult(BaseModel):
    """Outcome of executing a `ToolCall`.

    Tools never raise out of `execute()` — every failure mode is
    captured here as `error`. This is essential for transcript
    integrity (a raised exception would tear down the loop and lose
    all preceding turns) and for making error-handling decisions
    visible in the scored eval.
    """

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class AgentStep(BaseModel):
    """A single (call, result) pair in the transcript."""

    step_index: int
    call: ToolCall
    result: ToolResult


# ---------------------------------------------------------------------------
# Termination signals
# ---------------------------------------------------------------------------


class FinalAnswer(BaseModel):
    """Wrapper around an `AnvilResponse` produced by the agent.

    The agent must invoke the special `finalize` tool with this
    payload to terminate. Wrapping `AnvilResponse` (rather than
    inheriting) keeps the existing schema unchanged for the
    non-agentic path while letting the agent loop emit identical
    downstream artifacts.
    """

    response: AnvilResponse


class TerminationReason(BaseModel):
    """Why the agent loop ended.

    `kind` is the operationally important field; `detail` is free-form
    for logs and reports. We pin the literal set of values so a typo
    can't silently slip through.
    """

    kind: Literal[
        "finalized",
        "budget_steps_exhausted",
        "budget_tokens_exhausted",
        "tool_error_limit",
        "decider_error",
    ]
    detail: str = ""


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class AgentBudget(BaseModel):
    """Hard caps on the agent loop. All caps are inclusive.

    `max_steps` is the number of tool invocations (excluding the
    finalize call). 8 is the default — empirically enough for a
    multi-hop calculation query and tight enough that a runaway loop
    on real-NIM costs at most ~8 calls (~24s on free tier).
    """

    max_steps: int = 8
    max_total_prompt_tokens: int = 200_000
    max_tool_errors: int = 3

    @model_validator(mode="after")
    def _validate(self) -> AgentBudget:
        if self.max_steps < 1:
            raise ValueError("max_steps must be ≥ 1")
        if self.max_total_prompt_tokens < 1:
            raise ValueError("max_total_prompt_tokens must be ≥ 1")
        if self.max_tool_errors < 0:
            raise ValueError("max_tool_errors must be ≥ 0")
        return self


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class AgentTranscript(BaseModel):
    """Complete record of one agent run.

    Persisted by `RunLogger` next to `per_example.json` /
    `raw_responses.jsonl` for every agentic eval run, so a reviewer
    can answer "what did the model decide and why?" without re-running.
    """

    query: str
    steps: list[AgentStep] = Field(default_factory=list)
    final_response: AnvilResponse | None = None
    termination: TerminationReason
    total_duration_ms: float = 0.0
    n_tool_calls: int = 0
    n_tool_errors: int = 0

    def to_summary(self) -> dict[str, Any]:
        """Compact summary suitable for logging / report tables."""
        return {
            "query_preview": self.query[:80],
            "n_steps": self.n_tool_calls,
            "n_errors": self.n_tool_errors,
            "termination": self.termination.kind,
            "tools_used": [s.call.name for s in self.steps],
            "duration_ms": round(self.total_duration_ms, 1),
            "finalized": self.final_response is not None,
        }


__all__ = [
    "AgentBudget",
    "AgentStep",
    "AgentTranscript",
    "FinalAnswer",
    "TerminationReason",
    "ToolCall",
    "ToolResult",
]
