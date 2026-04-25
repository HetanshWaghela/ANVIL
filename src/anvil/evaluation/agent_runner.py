"""M6 — Evaluation runner that drives `AnvilAgent` instead of `AnvilGenerator`.

Mirrors `EvaluationRunner` so we get apples-to-apples comparison:
same metrics, same golden examples, same scoring — only the
*producer* of the response changes.

The metric layer expects (response, retrieved_chunks). The agent
doesn't have a single "the retrieval" — it can call `retrieve_context`
multiple times. We aggregate: the union of all chunks returned by
every `retrieve_context` step in the transcript, deduplicated by
`element_id`, ordered by their first appearance. This is the
fairest interpretation:

- An agent that never retrieves gets retrieval_recall=0, which is
  the right signal.
- An agent that retrieves the same chunk twice doesn't double-count.
- Chunks pulled in via `graph_lookup` are NOT counted as retrieval —
  graph lookup is a *separate* tool, and conflating it would
  inflate retrieval scores.

Per-example output:
  AgentEvaluationResult(EvaluationResult, transcript_summary)

The transcript summary is what `RunLogger.write_per_example` will
persist next to the metric scores so reviewers can replay agent
decisions.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from anvil.evaluation.metrics import (
    calculation_correctness,
    citation_accuracy,
    entity_grounding,
    faithfulness,
    refusal_calibration,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    structural_completeness,
)
from anvil.generation.agent import AgentOutcome, AnvilAgent
from anvil.logging_config import get_logger
from anvil.schemas.document import DocumentElement
from anvil.schemas.evaluation import EvaluationResult, GoldenExample, MetricScore
from anvil.schemas.generation import AnvilResponse
from anvil.schemas.retrieval import RetrievedChunk

log = get_logger(__name__)


@dataclass
class AgentRunSummary:
    """Aggregated agent-eval result. Same shape as RunSummary
    but carries `AgentOutcome`s in `outcomes`."""

    per_example: list[EvaluationResult]
    outcomes: list[AgentOutcome] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0

    # Agent-specific aggregates surfaced for the headline table.
    avg_tool_calls: float = 0.0
    avg_tool_errors: float = 0.0
    finalize_rate: float = 0.0  # fraction terminated via FinalAnswer
    budget_exhaustion_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate,
            "pass_rate": self.pass_rate,
            "agent_aggregate": {
                "avg_tool_calls": self.avg_tool_calls,
                "avg_tool_errors": self.avg_tool_errors,
                "finalize_rate": self.finalize_rate,
                "budget_exhaustion_rate": self.budget_exhaustion_rate,
            },
            "per_example": [r.model_dump() for r in self.per_example],
        }


def _aggregate_retrieval(outcome: AgentOutcome) -> list[RetrievedChunk]:
    """Union of chunks from every `retrieve_context` step, dedup by element_id.

    Why dedup-by-element-id and not by content hash? Because metric
    code keys retrieval-relevance off `element_id` (from the underlying
    DocumentElement). Two retrievals returning the same chunk with
    different scores still represent ONE retrieved element.
    """
    seen: OrderedDict[str, RetrievedChunk] = OrderedDict()
    for step in outcome.transcript.steps:
        if step.call.name != "retrieve_context" or not step.result.ok:
            continue
        for raw in step.result.output.get("chunks", []):
            eid = raw.get("element_id")
            if not eid or eid in seen:
                continue
            # Reconstruct the typed chunk from the tool-output dict.
            # This mirrors the round-trip the agent_tools._retrieve
            # adapter performed.
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


class AgentEvaluationRunner:
    """Run the golden dataset through an `AnvilAgent` and score each output."""

    def __init__(
        self,
        agent: AnvilAgent,
        element_index: dict[str, DocumentElement] | None = None,
        inter_example_delay_s: float = 0.0,
    ) -> None:
        self.agent = agent
        self.element_index = element_index
        self.inter_example_delay_s = inter_example_delay_s

    async def run(self, examples: list[GoldenExample]) -> AgentRunSummary:
        per_example: list[EvaluationResult] = []
        outcomes: list[AgentOutcome] = []
        for idx, ex in enumerate(examples):
            if idx > 0 and self.inter_example_delay_s > 0:
                await asyncio.sleep(self.inter_example_delay_s)
            outcome = await self.agent.run(ex.query)
            outcomes.append(outcome)
            retrieved = _aggregate_retrieval(outcome)
            metrics = self._score(
                ex,
                outcome.response,
                retrieved,
                element_index=self.element_index,
            )
            passed = all(m.passed for m in metrics)
            per_example.append(
                EvaluationResult(
                    example_id=ex.id,
                    category=ex.category,
                    metrics=metrics,
                    passed=passed,
                )
            )
            log.info(
                "agent_runner.example.done",
                example_id=ex.id,
                passed=passed,
                **outcome.transcript.to_summary(),
            )

        # Aggregate metric scores
        aggregate: dict[str, float] = {}
        names = {m.name for r in per_example for m in r.metrics}
        for name in names:
            values = [
                m.value for r in per_example for m in r.metrics if m.name == name
            ]
            if values:
                aggregate[name] = sum(values) / len(values)

        n = len(per_example)
        pass_rate = (sum(1 for r in per_example if r.passed) / n) if n else 0.0
        if outcomes:
            avg_calls = sum(o.transcript.n_tool_calls for o in outcomes) / n
            avg_errors = sum(o.transcript.n_tool_errors for o in outcomes) / n
            finalize_rate = sum(
                1 for o in outcomes if o.transcript.termination.kind == "finalized"
            ) / n
            budget_rate = sum(
                1
                for o in outcomes
                if o.transcript.termination.kind == "budget_steps_exhausted"
            ) / n
        else:
            avg_calls = avg_errors = finalize_rate = budget_rate = 0.0

        return AgentRunSummary(
            per_example=per_example,
            outcomes=outcomes,
            aggregate=aggregate,
            pass_rate=pass_rate,
            avg_tool_calls=avg_calls,
            avg_tool_errors=avg_errors,
            finalize_rate=finalize_rate,
            budget_exhaustion_rate=budget_rate,
        )

    # Metric scoring is identical to EvaluationRunner. Inlined rather
    # than imported because the future may diverge — agent runs may
    # gain agent-specific metrics (tool-call efficiency, etc.).
    def _score(
        self,
        example: GoldenExample,
        response: AnvilResponse,
        retrieved: list[RetrievedChunk],
        element_index: dict[str, DocumentElement] | None = None,
    ) -> list[MetricScore]:
        metrics: list[MetricScore] = [refusal_calibration(response, example)]
        if example.expected_refusal:
            return metrics

        metrics.append(
            retrieval_precision_at_k(retrieved, example.expected_paragraph_refs, k=10)
        )
        metrics.append(
            retrieval_recall_at_k(retrieved, example.expected_paragraph_refs, k=10)
        )
        metrics.append(faithfulness(response, retrieved))
        metrics.append(
            citation_accuracy(response, retrieved, element_index=element_index)
        )
        metrics.append(entity_grounding(response, retrieved))
        metrics.append(
            structural_completeness(
                response,
                example.expected_paragraph_refs,
                retrieved=retrieved,
            )
        )
        if example.expected_values:
            metrics.append(
                calculation_correctness(
                    response,
                    example.expected_values,
                    tolerance=example.numeric_tolerance,
                )
            )
        return metrics


def transcripts_to_jsonable(outcomes: list[AgentOutcome]) -> list[dict[str, Any]]:
    """Helper for run_logger persistence."""
    return [o.transcript.model_dump(mode="json") for o in outcomes]


__all__ = [
    "AgentEvaluationRunner",
    "AgentRunSummary",
    "transcripts_to_jsonable",
]
