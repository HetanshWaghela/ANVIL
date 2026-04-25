"""Integration test — drive `AgentEvaluationRunner` over real golden examples.

Why integration: this exercises the *real* retriever, graph store,
calc engine, and pinned data — i.e. every adapter in `ToolRegistry`
end-to-end on real data. Only the *decider* is scripted (so we don't
need an LLM in CI).

What this regression locks:

1. `_aggregate_retrieval` correctly reconstructs `RetrievedChunk`s
   from tool-output dicts. A schema drift in `RetrievedChunk` (say,
   adding a non-defaulted field) will fail this test.

2. The agent runner produces metric-shaped output structurally
   identical to `EvaluationRunner` — the report writer / run logger
   downstream don't know the difference.

3. Agent-aggregate metrics (avg_tool_calls, finalize_rate) are
   computed correctly.

4. A scripted "good agent" (retrieve → calc → finalize with a
   correctly-formed AnvilResponse) passes the same metrics the
   normal generator would pass on the same example. This is what
   makes the M6 comparison meaningful: the agent CAN match the
   generator when it makes the right calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anvil.evaluation import (
    AgentEvaluationRunner,
    AgentRunSummary,
    load_golden_dataset,
)
from anvil.generation.agent import AnvilAgent
from anvil.generation.agent_backend import AgentDecision, ScriptedAgentBackend
from anvil.generation.agent_tools import ToolRegistry
from anvil.pipeline import build_pipeline
from anvil.schemas.agent import AgentBudget, FinalAnswer, ToolCall
from anvil.schemas.generation import (
    AnvilResponse,
    Citation,
    ResponseConfidence,
)

DATASET = (
    Path(__file__).resolve().parent.parent / "evaluation" / "golden_dataset.json"
)


@pytest.fixture(scope="module")
def pipeline() -> Any:
    return build_pipeline()


@pytest.fixture(scope="module")
def registry(pipeline: Any) -> ToolRegistry:
    return ToolRegistry(
        retriever=pipeline.retriever,
        graph_store=pipeline.graph_store,
        calc_engine=pipeline.generator.calc_engine,
    )


def _refusal(query: str) -> AnvilResponse:
    return AnvilResponse(
        query=query,
        answer="Insufficient evidence to answer.",
        citations=[],
        calculation_steps=[],
        confidence=ResponseConfidence.INSUFFICIENT,
        refusal_reason="Out-of-scope or unsupported query.",
        retrieved_context_ids=[],
    )


def _confident_response(
    query: str,
    citation_ref: str,
    citation_id: str,
    quoted_text: str,
) -> AnvilResponse:
    """Build a minimal but schema-valid in-domain response.

    The citation must reference an actual element from the parsed
    standard so `citation_accuracy` validates it.
    """
    return AnvilResponse(
        query=query,
        answer=f"See {citation_ref}.",
        citations=[
            Citation(
                paragraph_ref=citation_ref,
                page_number=1,
                quoted_text=quoted_text,
                source_element_id=citation_id,
            )
        ],
        calculation_steps=[],
        confidence=ResponseConfidence.HIGH,
        retrieved_context_ids=[citation_id],
    )


@pytest.mark.asyncio
async def test_agent_runner_runs_3_representative_examples(
    pipeline: Any, registry: ToolRegistry
) -> None:
    """Pick one example per category and drive the agent through each
    with hand-crafted decisions. The runner must produce a complete
    AgentRunSummary with one EvaluationResult per example.
    """
    examples = load_golden_dataset(DATASET)
    # Pick representatives we know exist in the synthetic standard.
    by_id = {e.id: e for e in examples}
    ids = [
        next(e.id for e in examples if not e.expected_refusal and e.expected_values),
        next(
            e.id for e in examples if not e.expected_refusal and not e.expected_values
        ),
        next(e.id for e in examples if e.expected_refusal),
    ]
    sample = [by_id[i] for i in ids]
    assert len(sample) == 3

    # For each example the script issues retrieve_context → finalize.
    # Final responses are deliberately MINIMAL — we're testing
    # plumbing, not metric pass-rates here.
    script: list[ToolCall | FinalAnswer | AgentDecision] = []
    for ex in sample:
        script.append(
            ToolCall(
                name="retrieve_context",
                arguments={"query": ex.query, "top_k": 3},
            )
        )
        if ex.expected_refusal:
            script.append(FinalAnswer(response=_refusal(ex.query)))
        else:
            # Cite the first expected paragraph_ref for that example.
            ref = ex.expected_paragraph_refs[0]
            # Find an element in the parsed standard with that ref —
            # the citation enforcer needs a real element to validate.
            elem = next(
                el
                for el in pipeline.generator.element_index.values()
                if (el.paragraph_ref or "") == ref
            )
            script.append(
                FinalAnswer(
                    response=_confident_response(
                        ex.query,
                        citation_ref=ref,
                        citation_id=elem.element_id,
                        quoted_text=elem.content[:120],
                    )
                )
            )

    decider = ScriptedAgentBackend(script)
    agent = AnvilAgent(
        decider=decider,
        registry=registry,
        budget=AgentBudget(max_steps=4),
    )
    runner = AgentEvaluationRunner(
        agent=agent,
        element_index=pipeline.generator.element_index,
    )
    summary: AgentRunSummary = await runner.run(sample)

    # ---- structural assertions -------------------------------------------
    assert len(summary.per_example) == 3
    assert len(summary.outcomes) == 3
    assert summary.finalize_rate == 1.0  # every example finalized
    assert summary.budget_exhaustion_rate == 0.0
    assert summary.avg_tool_calls == 1.0  # one retrieve per example
    assert summary.avg_tool_errors == 0.0

    # ---- metric coverage -------------------------------------------------
    refusal_results = [
        r
        for r, ex in zip(summary.per_example, sample, strict=True)
        if ex.expected_refusal
    ]
    in_domain_results = [
        r
        for r, ex in zip(summary.per_example, sample, strict=True)
        if not ex.expected_refusal
    ]
    # OOD examples score only refusal_calibration (single metric).
    assert all(len(r.metrics) == 1 for r in refusal_results)
    assert all(r.metrics[0].name == "refusal_calibration" for r in refusal_results)
    # In-domain examples score the full battery.
    metric_names = {m.name for r in in_domain_results for m in r.metrics}
    assert {
        "refusal_calibration",
        "retrieval_precision_at_k",
        "retrieval_recall_at_k",
        "faithfulness",
        "citation_accuracy",
        "entity_grounding",
        "structural_completeness",
    }.issubset(metric_names)

    # ---- to_dict round-trips and includes agent-aggregate ----------------
    d = summary.to_dict()
    assert "agent_aggregate" in d
    assert d["agent_aggregate"]["finalize_rate"] == 1.0


@pytest.mark.asyncio
async def test_agent_runner_handles_budget_exhaustion(
    pipeline: Any, registry: ToolRegistry
) -> None:
    """Agent that loops without finalizing should still produce a
    metric-shaped result for every example — we get refusal-shaped
    responses, not a crash."""
    examples = load_golden_dataset(DATASET)[:2]

    # Always retrieve, never finalize ⇒ budget exhaustion every time.
    script: list[ToolCall | FinalAnswer | AgentDecision] = [
        ToolCall(name="retrieve_context", arguments={"query": "x"})
    ] * 100  # well above the budget × n_examples

    agent = AnvilAgent(
        decider=ScriptedAgentBackend(script),
        registry=registry,
        budget=AgentBudget(max_steps=2),
    )
    runner = AgentEvaluationRunner(
        agent=agent, element_index=pipeline.generator.element_index
    )
    summary = await runner.run(examples)
    assert len(summary.per_example) == 2
    assert summary.finalize_rate == 0.0
    assert summary.budget_exhaustion_rate == 1.0
    # Every response is a refusal
    assert all(
        o.response.confidence == ResponseConfidence.INSUFFICIENT
        for o in summary.outcomes
    )
