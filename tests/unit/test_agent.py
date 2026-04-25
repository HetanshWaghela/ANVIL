"""Tests for the M6 agentic tool-calling loop.

Covers:
1. Happy path — scripted backend calls retrieve → calculate → finalize,
   transcript records every step in order, response unchanged from what
   `FinalAnswer` carried.
2. Step-budget exhaustion — script that never finalizes ⇒ termination
   reason `budget_steps_exhausted`, response is a refusal.
3. Tool-error budget — script that triggers max_tool_errors+1 failed
   tool calls ⇒ termination `tool_error_limit`, refusal response.
4. Decider exception — async `decide()` raises ⇒ termination
   `decider_error`, refusal carries the exception text.
5. ToolRegistry adapters — exercise each adapter (retrieve_context,
   graph_lookup, pinned_lookup x3 kinds, calculate happy path,
   calculate engine-refusal path).
6. AgentDecision XOR invariant — both fields set or neither set both
   raise.
7. Joint-type coercion regression — locks the LLM-friendly variants
   ('Type 1', '1', 1) all converge to the int form pinned data
   demands.
"""

from __future__ import annotations

import pytest

from anvil.generation.agent import AnvilAgent
from anvil.generation.agent_backend import (
    AgentDecision,
    ScriptedAgentBackend,
)
from anvil.generation.agent_tools import ToolRegistry, _coerce_joint_type
from anvil.pipeline import build_pipeline
from anvil.schemas.agent import (
    AgentBudget,
    FinalAnswer,
    ToolCall,
)
from anvil.schemas.generation import (
    AnvilResponse,
    Citation,
    ResponseConfidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    """Real registry built off the production pipeline — exercises the
    actual adapters end-to-end."""
    pipe = build_pipeline()
    return ToolRegistry(
        retriever=pipe.retriever,
        graph_store=pipe.graph_store,
        calc_engine=pipe.generator.calc_engine,
    )


def _stub_response(query: str, *, confidence: ResponseConfidence) -> AnvilResponse:
    """Refusal-shaped or high-confidence response with a single canonical citation.

    The agent loop is supposed to be agnostic to citation content — its
    job is to pass through whatever the decider produced. So we use a
    minimal, schema-valid response shape across all tests.
    """
    if confidence == ResponseConfidence.INSUFFICIENT:
        return AnvilResponse(
            query=query,
            answer="I cannot answer with the available context.",
            citations=[],
            calculation_steps=[],
            confidence=confidence,
            refusal_reason="No supporting evidence retrieved.",
            retrieved_context_ids=[],
        )
    return AnvilResponse(
        query=query,
        answer="Cylindrical shell thickness per A-27(c)(1) is t = PR/(SE-0.6P).",
        citations=[
            Citation(
                paragraph_ref="A-27(c)(1)",
                page_number=1,
                quoted_text=(
                    "When P does not exceed 0.385SE, the formula t = PR/(SE-0.6P) applies."
                ),
                source_element_id="A-27(c)(1)",
            ),
        ],
        calculation_steps=[],
        confidence=confidence,
        retrieved_context_ids=[],
    )


# ---------------------------------------------------------------------------
# 1) Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_records_full_transcript(registry: ToolRegistry) -> None:
    """A scripted retrieve → finalize sequence:
    transcript.steps == 1, finalize lands the response unchanged."""
    final = _stub_response("test", confidence=ResponseConfidence.HIGH)
    decider = ScriptedAgentBackend(
        [
            ToolCall(
                name="retrieve_context",
                arguments={"query": "cylindrical shell thickness", "top_k": 3},
            ),
            FinalAnswer(response=final),
        ]
    )
    agent = AnvilAgent(decider=decider, registry=registry, budget=AgentBudget())
    outcome = await agent.run("What is the cylindrical shell formula?")

    # Response identity preserved (Pydantic equality is structural)
    assert outcome.response == final
    # Transcript shape
    assert outcome.transcript.termination.kind == "finalized"
    assert outcome.transcript.n_tool_calls == 1
    assert outcome.transcript.n_tool_errors == 0
    assert len(outcome.transcript.steps) == 1
    step = outcome.transcript.steps[0]
    assert step.step_index == 0
    assert step.call.name == "retrieve_context"
    assert step.result.ok
    assert step.result.output["n"] >= 1
    assert outcome.transcript.final_response == final
    # `to_summary` is what RunLogger persists — verify it's compact and complete.
    summary = outcome.transcript.to_summary()
    assert summary["termination"] == "finalized"
    assert summary["tools_used"] == ["retrieve_context"]
    assert summary["finalized"] is True


# ---------------------------------------------------------------------------
# 2) Step-budget exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_budget_exhaustion_yields_refusal(registry: ToolRegistry) -> None:
    """Decider only ever calls retrieve_context; budget caps at 3 ⇒
    termination=budget_steps_exhausted, response is a refusal."""
    decider = ScriptedAgentBackend(
        [ToolCall(name="retrieve_context", arguments={"query": "anything"})] * 5
    )
    agent = AnvilAgent(
        decider=decider,
        registry=registry,
        budget=AgentBudget(max_steps=3),
    )
    outcome = await agent.run("Loop me forever")

    assert outcome.transcript.termination.kind == "budget_steps_exhausted"
    assert outcome.transcript.n_tool_calls == 3
    assert outcome.response.confidence == ResponseConfidence.INSUFFICIENT
    assert outcome.response.refusal_reason is not None
    assert "max_steps=3" in outcome.response.refusal_reason
    # final_response is None on non-finalized terminations — preserves
    # the distinction between "agent gave up" and "agent answered".
    assert outcome.transcript.final_response is None


# ---------------------------------------------------------------------------
# 3) Tool-error budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_error_budget_yields_refusal(registry: ToolRegistry) -> None:
    """4 unknown-tool calls with max_tool_errors=3 ⇒ tool_error_limit."""
    decider = ScriptedAgentBackend(
        [ToolCall(name="not_a_real_tool", arguments={})] * 5
    )
    agent = AnvilAgent(
        decider=decider,
        registry=registry,
        budget=AgentBudget(max_steps=10, max_tool_errors=3),
    )
    outcome = await agent.run("Trigger the error path")

    assert outcome.transcript.termination.kind == "tool_error_limit"
    # 4 calls — 1, 2, 3 (tolerated) + 4th (exceeds) → break
    assert outcome.transcript.n_tool_calls == 4
    assert outcome.transcript.n_tool_errors == 4
    assert outcome.response.confidence == ResponseConfidence.INSUFFICIENT


# ---------------------------------------------------------------------------
# 4) Decider exception
# ---------------------------------------------------------------------------


class _ExplodingDecider:
    """A decider that always raises — simulates a transient LLM error."""

    async def decide(self, query, history, tool_manifest):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated decider crash")


@pytest.mark.asyncio
async def test_decider_exception_is_caught(registry: ToolRegistry) -> None:
    agent = AnvilAgent(
        decider=_ExplodingDecider(),  # type: ignore[arg-type]
        registry=registry,
    )
    outcome = await agent.run("test")
    assert outcome.transcript.termination.kind == "decider_error"
    assert "simulated decider crash" in outcome.transcript.termination.detail
    assert outcome.response.refusal_reason is not None
    assert outcome.response.confidence == ResponseConfidence.INSUFFICIENT


# ---------------------------------------------------------------------------
# 5) Tool adapters — round-trip each
# ---------------------------------------------------------------------------


def test_tool_retrieve_context(registry: ToolRegistry) -> None:
    out = registry.execute(
        ToolCall(
            name="retrieve_context",
            arguments={"query": "cylindrical shell formula", "top_k": 5},
        )
    )
    assert out.ok and out.output["n"] >= 1
    chunks = out.output["chunks"]
    assert all("paragraph_ref" in c and "content" in c for c in chunks)


def test_tool_retrieve_context_validates_query(registry: ToolRegistry) -> None:
    out = registry.execute(
        ToolCall(name="retrieve_context", arguments={"query": "", "top_k": 5})
    )
    assert not out.ok
    assert "query" in (out.error or "").lower()


def test_tool_graph_lookup(registry: ToolRegistry) -> None:
    """A-27 is the parent paragraph for the cylindrical-shell formulas
    A-27(c)(1) etc. and is guaranteed to be in the graph."""
    out = registry.execute(
        ToolCall(
            name="graph_lookup",
            arguments={"paragraph_ref": "A-27", "max_hops": 1},
        )
    )
    assert out.ok
    # Seed exists, expansion should pull in at least the formula children.
    assert len(out.output["seeds"]) >= 1
    assert len(out.output["expanded"]) >= len(out.output["seeds"])
    assert out.output["max_hops"] == 1


def test_tool_graph_lookup_unknown_ref_returns_empty(registry: ToolRegistry) -> None:
    """Unknown paragraph_ref must NOT raise — the tool surfaces an
    empty result so the agent can pivot to a different ref."""
    out = registry.execute(
        ToolCall(
            name="graph_lookup",
            arguments={"paragraph_ref": "Z-99-not-real"},
        )
    )
    assert out.ok
    assert out.output["seeds"] == []
    assert out.output["expanded"] == []


def test_tool_pinned_lookup_material(registry: ToolRegistry) -> None:
    out = registry.execute(
        ToolCall(
            name="pinned_lookup",
            arguments={"kind": "material", "key": "SM-516 Gr 70"},
        )
    )
    assert out.ok and out.output["found"] is True
    assert out.output["spec_grade"].startswith("SM-516")
    assert out.output["max_temp_c"] > 0
    assert out.output["tabulated_temps_c"]  # non-empty list


def test_tool_pinned_lookup_allowable_stress(registry: ToolRegistry) -> None:
    out = registry.execute(
        ToolCall(
            name="pinned_lookup",
            arguments={
                "kind": "allowable_stress",
                "key": "SM-516 Gr 70",
                "temp_c": 200,
            },
        )
    )
    assert out.ok and out.output["found"] is True
    S = out.output["allowable_stress_mpa"]
    assert isinstance(S, (int, float)) and S > 0


def test_tool_pinned_lookup_joint_efficiency(registry: ToolRegistry) -> None:
    """The agent might pass `'Type 1'` or `1` — both must work, and the
    joint_type comes back coerced to int in the output."""
    for key in ("Type 1", "1"):
        out = registry.execute(
            ToolCall(
                name="pinned_lookup",
                arguments={
                    "kind": "joint_efficiency",
                    "key": key,
                    "rt_level": "Full RT",
                },
            )
        )
        assert out.ok, f"failed for key={key!r}: {out.error}"
        assert out.output["found"] is True
        assert out.output["joint_type"] == 1
        assert out.output["joint_efficiency"] == 1.0


def test_tool_calculate_happy_path(registry: ToolRegistry) -> None:
    out = registry.execute(
        ToolCall(
            name="calculate",
            arguments={
                "component": "cylindrical_shell",
                "P_mpa": 1.5,
                "design_temp_c": 200,
                "inside_diameter_mm": 1000,
                "material": "SM-516 Gr 70",
                "joint_type": "Type 1",
                "rt_level": "Full RT",
                "corrosion_allowance_mm": 0.0,
            },
        )
    )
    assert out.ok and out.output["ok"] is True
    assert out.output["t_min_mm"] > 0 and out.output["mawp_mpa"] > 0
    assert out.output["formula_ref"]


def test_tool_calculate_unknown_material_returns_engine_refusal(
    registry: ToolRegistry,
) -> None:
    """The engine raises CalculationError; the adapter must capture it
    in the output (ok=False, error=...) — not propagate as a tool error,
    because this is a *legitimate* engine refusal the agent should
    interpret and refuse on."""
    out = registry.execute(
        ToolCall(
            name="calculate",
            arguments={
                "component": "cylindrical_shell",
                "P_mpa": 1.5,
                "design_temp_c": 200,
                "inside_diameter_mm": 1000,
                "material": "NOT-A-MATERIAL",
                "joint_type": 1,
                "rt_level": "Full RT",
                "corrosion_allowance_mm": 0.0,
            },
        )
    )
    # The TOOL itself succeeded (it returned a structured result).
    assert out.ok
    # The ENGINE refused — surfaced via output["ok"]=False.
    assert out.output["ok"] is False
    assert "CalculationError" in out.output["error"]


def test_tool_unknown_returns_error(registry: ToolRegistry) -> None:
    out = registry.execute(ToolCall(name="not_a_tool", arguments={}))
    assert not out.ok
    assert "Unknown tool" in (out.error or "")


# ---------------------------------------------------------------------------
# 6) AgentDecision XOR invariant
# ---------------------------------------------------------------------------


def test_agent_decision_requires_exactly_one_branch() -> None:
    final = _stub_response("q", confidence=ResponseConfidence.HIGH)
    # Both set → invalid.
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        AgentDecision(
            tool_call=ToolCall(name="retrieve_context", arguments={"query": "x"}),
            final=FinalAnswer(response=final),
        )
    # Neither set → invalid.
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        AgentDecision()


# ---------------------------------------------------------------------------
# 7) Joint-type coercion regression
# ---------------------------------------------------------------------------


def test_coerce_joint_type_accepts_llm_shapes() -> None:
    assert _coerce_joint_type(1) == 1
    assert _coerce_joint_type("1") == 1
    assert _coerce_joint_type("Type 1") == 1
    assert _coerce_joint_type("type  3") == 3
    assert _coerce_joint_type(2.0) == 2


def test_coerce_joint_type_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        _coerce_joint_type("seven")
    with pytest.raises(ValueError):
        _coerce_joint_type(True)  # bool is rejected even though it's an int subclass
    with pytest.raises(ValueError):
        _coerce_joint_type(2.5)
    with pytest.raises(ValueError):
        _coerce_joint_type([1])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8) ScriptedAgentBackend bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scripted_backend_exhausted_script_raises_loudly(
    registry: ToolRegistry,
) -> None:
    """An exhausted script must raise IndexError — silent looping
    forever would mask test bugs."""
    decider = ScriptedAgentBackend([])  # empty
    agent = AnvilAgent(decider=decider, registry=registry)
    outcome = await agent.run("test")
    # The agent catches decider exceptions → terminates as decider_error.
    assert outcome.transcript.termination.kind == "decider_error"
    assert "exhausted" in outcome.transcript.termination.detail.lower()
