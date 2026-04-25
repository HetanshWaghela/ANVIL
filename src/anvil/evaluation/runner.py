"""Evaluation runner — run the golden dataset through the full pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
from anvil.generation.generator import AnvilGenerator
from anvil.schemas.document import DocumentElement
from anvil.schemas.evaluation import EvaluationResult, GoldenExample, MetricScore
from anvil.schemas.generation import AnvilResponse
from anvil.schemas.retrieval import RetrievedChunk


@dataclass
class RunSummary:
    """Aggregated result of an evaluation run."""

    per_example: list[EvaluationResult]
    aggregate: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate,
            "pass_rate": self.pass_rate,
            "per_example": [r.model_dump() for r in self.per_example],
        }


class EvaluationRunner:
    """Runs the golden dataset through an AnvilGenerator and scores each output."""

    def __init__(self, generator: AnvilGenerator) -> None:
        self.generator = generator

    async def run(self, examples: list[GoldenExample]) -> RunSummary:
        per_example: list[EvaluationResult] = []
        # Reuse the generator's element_index so citation_accuracy can
        # validate canonical-ref quotes against the parsed standard.
        element_index = getattr(self.generator, "element_index", None)
        for ex in examples:
            outcome = await self.generator.generate(ex.query, top_k=10)
            metrics = self._score(
                ex,
                outcome.response,
                outcome.retrieved_chunks,
                element_index=element_index,
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

        # Aggregate
        aggregate: dict[str, float] = {}
        names = {m.name for r in per_example for m in r.metrics}
        for name in names:
            values = [
                m.value
                for r in per_example
                for m in r.metrics
                if m.name == name
            ]
            if values:
                aggregate[name] = sum(values) / len(values)
        pass_rate = (
            sum(1 for r in per_example if r.passed) / len(per_example)
            if per_example
            else 0.0
        )
        return RunSummary(
            per_example=per_example, aggregate=aggregate, pass_rate=pass_rate
        )

    def _score(
        self,
        example: GoldenExample,
        response: AnvilResponse,
        retrieved: list[RetrievedChunk],
        element_index: dict[str, DocumentElement] | None = None,
    ) -> list[MetricScore]:
        metrics: list[MetricScore] = []
        # Refusal-aware: OOD/refusal examples get refusal calibration only
        metrics.append(refusal_calibration(response, example))
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


def write_summary_json(summary: RunSummary, path: str | Path) -> None:
    import json

    Path(path).write_text(json.dumps(summary.to_dict(), indent=2, default=str))
