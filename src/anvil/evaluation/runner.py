"""Evaluation runner — run the golden dataset through the full pipeline."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from anvil import EvaluationError, RetryableGenerationError
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
from anvil.generation.generator import AnvilGenerator, GenerationOutcome
from anvil.logging_config import get_logger
from anvil.schemas.document import DocumentElement
from anvil.schemas.evaluation import EvaluationResult, GoldenExample, MetricScore
from anvil.schemas.generation import AnvilResponse
from anvil.schemas.retrieval import RetrievedChunk

log = get_logger(__name__)


@dataclass
class RunSummary:
    """Aggregated result of an evaluation run.

    `outcomes` carries one `GenerationOutcome` per example, parallel to
    `per_example`, so callers (notably the run-logger / headline-runs
    script) can record raw responses without invoking the generator a
    second time. Re-invoking would double the cost of every real-NIM
    eval run, an issue caught in M5 testing.
    """

    per_example: list[EvaluationResult]
    outcomes: list[GenerationOutcome] = field(default_factory=list)
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

    def __init__(
        self,
        generator: AnvilGenerator,
        *,
        retryable_attempts: int | None = None,
        retryable_cooldown_s: float | None = None,
        inter_example_delay_s: float | None = None,
        retry_backend_errors: bool | None = None,
    ) -> None:
        self.generator = generator
        self.retryable_attempts = retryable_attempts or int(
            os.environ.get("ANVIL_EVAL_RETRYABLE_ATTEMPTS", "3")
        )
        self.retryable_cooldown_s = (
            retryable_cooldown_s
            if retryable_cooldown_s is not None
            else float(os.environ.get("ANVIL_EVAL_RETRY_COOLDOWN_S", "75"))
        )
        self.inter_example_delay_s = (
            inter_example_delay_s
            if inter_example_delay_s is not None
            else float(os.environ.get("ANVIL_EVAL_INTER_EXAMPLE_DELAY_S", "0"))
        )
        self.retry_backend_errors = (
            retry_backend_errors
            if retry_backend_errors is not None
            else os.environ.get("ANVIL_EVAL_RETRY_BACKEND_ERRORS", "").lower()
            in {"1", "true", "yes"}
        )

    async def run(self, examples: list[GoldenExample]) -> RunSummary:
        per_example_by_index: dict[int, EvaluationResult] = {}
        outcomes_by_index: dict[int, GenerationOutcome] = {}
        # Reuse the generator's element_index so citation_accuracy can
        # validate canonical-ref quotes against the parsed standard.
        element_index = getattr(self.generator, "element_index", None)
        pending: list[tuple[int, GoldenExample, int]] = [
            (idx, ex, 1) for idx, ex in enumerate(examples)
        ]
        while pending:
            deferred: list[tuple[int, GoldenExample, int]] = []
            while pending:
                idx, ex, attempt = pending.pop(0)
                try:
                    outcome = await self.generator.generate(ex.query, top_k=10)
                except RetryableGenerationError as exc:
                    if attempt >= self.retryable_attempts:
                        raise EvaluationError(
                            "Retryable LLM error persisted after "
                            f"{attempt} attempts for example {ex.id}: {exc}"
                        ) from exc
                    log.warning(
                        "evaluation.retryable_deferred",
                        example_id=ex.id,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        remaining_before_retry=len(pending),
                        cooldown_s=self.retryable_cooldown_s,
                        error=repr(exc),
                    )
                    # Defer the failed request and every not-yet-attempted
                    # request. Continuing immediately after a 429 would only
                    # pollute more examples with provider throttling.
                    deferred.append((idx, ex, attempt + 1))
                    deferred.extend(pending)
                    pending = []
                    break
                if self.retry_backend_errors and outcome.backend_error is not None:
                    if attempt >= self.retryable_attempts:
                        raise EvaluationError(
                            "LLM backend error persisted after "
                            f"{attempt} attempts for example {ex.id}: "
                            f"{outcome.backend_error}"
                        )
                    log.warning(
                        "evaluation.backend_error_deferred",
                        example_id=ex.id,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        remaining_before_retry=len(pending),
                        cooldown_s=self.retryable_cooldown_s,
                        error=outcome.backend_error,
                    )
                    deferred.append((idx, ex, attempt + 1))
                    deferred.extend(pending)
                    pending = []
                    break
                outcomes_by_index[idx] = outcome
                if self.inter_example_delay_s > 0:
                    await asyncio.sleep(self.inter_example_delay_s)
                metrics = self._score(
                    ex,
                    outcome.response,
                    outcome.retrieved_chunks,
                    element_index=element_index,
                )
                passed = all(m.passed for m in metrics)
                per_example_by_index[idx] = EvaluationResult(
                    example_id=ex.id,
                    category=ex.category,
                    metrics=metrics,
                    passed=passed,
                )
            if deferred:
                log.info(
                    "evaluation.retryable_cooldown",
                    n_deferred=len(deferred),
                    cooldown_s=self.retryable_cooldown_s,
                )
                if self.retryable_cooldown_s > 0:
                    await asyncio.sleep(self.retryable_cooldown_s)
                pending = deferred

        per_example = [per_example_by_index[idx] for idx in range(len(examples))]
        outcomes = [outcomes_by_index[idx] for idx in range(len(examples))]

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
            per_example=per_example,
            outcomes=outcomes,
            aggregate=aggregate,
            pass_rate=pass_rate,
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
