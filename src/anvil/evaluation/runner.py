"""Evaluation runner — run the golden dataset through the full pipeline."""

from __future__ import annotations

import asyncio
import json
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
from anvil.generation.citation_enforcer import CitationValidationResult
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

    async def run(
        self,
        examples: list[GoldenExample],
        *,
        checkpoint_path: Path | None = None,
        resume: bool = False,
    ) -> RunSummary:
        per_example_by_index: dict[int, EvaluationResult] = {}
        outcomes_by_index: dict[int, GenerationOutcome] = {}
        # Reuse the generator's element_index so citation_accuracy can
        # validate canonical-ref quotes against the parsed standard.
        element_index = getattr(self.generator, "element_index", None)
        if checkpoint_path is not None and resume:
            loaded_results, loaded_outcomes = _load_checkpoint(
                checkpoint_path,
                examples,
            )
            per_example_by_index.update(loaded_results)
            outcomes_by_index.update(loaded_outcomes)
            if loaded_results:
                log.info(
                    "evaluation.resume_loaded",
                    checkpoint=str(checkpoint_path),
                    n_completed=len(loaded_results),
                )
        pending: list[tuple[int, GoldenExample, int]] = [
            (idx, ex, 1)
            for idx, ex in enumerate(examples)
            if idx not in per_example_by_index
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
                if checkpoint_path is not None:
                    _write_checkpoint(
                        checkpoint_path,
                        examples,
                        per_example_by_index,
                        outcomes_by_index,
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


def _load_checkpoint(
    checkpoint_path: Path,
    examples: list[GoldenExample],
) -> tuple[dict[int, EvaluationResult], dict[int, GenerationOutcome]]:
    if not checkpoint_path.exists():
        return {}, {}
    payload = json.loads(checkpoint_path.read_text())
    if not isinstance(payload, dict):
        raise EvaluationError(f"{checkpoint_path} did not contain a JSON object")
    ids = payload.get("example_ids")
    expected_ids = [ex.id for ex in examples]
    if ids != expected_ids:
        raise EvaluationError(
            f"{checkpoint_path} was created for a different example set; "
            "refusing to resume and risk mixing runs."
        )
    completed = payload.get("completed")
    if not isinstance(completed, list):
        raise EvaluationError(f"{checkpoint_path} is missing completed examples")

    per_example_by_index: dict[int, EvaluationResult] = {}
    outcomes_by_index: dict[int, GenerationOutcome] = {}
    for item in completed:
        if not isinstance(item, dict):
            raise EvaluationError(f"{checkpoint_path} contains a malformed item")
        idx = int(item["index"])
        if idx < 0 or idx >= len(examples):
            raise EvaluationError(f"{checkpoint_path} contains out-of-range index {idx}")
        if item.get("example_id") != examples[idx].id:
            raise EvaluationError(
                f"{checkpoint_path} example id mismatch at index {idx}"
            )
        result = EvaluationResult.model_validate(item["evaluation_result"])
        response = AnvilResponse.model_validate(item["response"])
        retrieved = [
            RetrievedChunk.model_validate(chunk)
            for chunk in item.get("retrieved", [])
        ]
        per_example_by_index[idx] = result
        outcomes_by_index[idx] = GenerationOutcome(
            response=response,
            retrieved_chunks=retrieved,
            citation_validation=CitationValidationResult(total=0, valid=0, issues=[]),
            backend_error=item.get("backend_error"),
        )
    return per_example_by_index, outcomes_by_index


def _write_checkpoint(
    checkpoint_path: Path,
    examples: list[GoldenExample],
    per_example_by_index: dict[int, EvaluationResult],
    outcomes_by_index: dict[int, GenerationOutcome],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    for idx in sorted(per_example_by_index):
        outcome = outcomes_by_index[idx]
        completed.append(
            {
                "index": idx,
                "example_id": examples[idx].id,
                "evaluation_result": per_example_by_index[idx].model_dump(),
                "response": outcome.response.model_dump(),
                "retrieved": [c.model_dump() for c in outcome.retrieved_chunks],
                "backend_error": outcome.backend_error,
            }
        )
    payload = {
        "version": 1,
        "example_ids": [ex.id for ex in examples],
        "completed": completed,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    tmp_path.replace(checkpoint_path)
