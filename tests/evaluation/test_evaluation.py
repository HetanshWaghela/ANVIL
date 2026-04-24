"""Evaluation-suite tests: run the golden dataset through the full pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.evaluation.dataset import load_golden_dataset
from anvil.evaluation.metrics import (
    calculation_correctness,
    citation_accuracy,
    faithfulness,
    refusal_calibration,
    retrieval_recall_at_k,
)
from anvil.evaluation.regression import compare_runs
from anvil.evaluation.runner import EvaluationRunner
from anvil.schemas.evaluation import GoldenExample

DATASET = Path(__file__).resolve().parent / "golden_dataset.json"


@pytest.fixture(scope="module")
def examples() -> list[GoldenExample]:
    return load_golden_dataset(DATASET)


@pytest.mark.asyncio
async def test_run_full_eval_pass_rate(pipeline, examples) -> None:
    runner = EvaluationRunner(pipeline.generator)
    summary = await runner.run(examples)
    # Expect most examples to pass
    assert summary.pass_rate >= 0.7, (
        f"pass_rate={summary.pass_rate}, aggregate={summary.aggregate}, "
        f"failures={[r.example_id for r in summary.per_example if not r.passed]}"
    )
    # Expect key aggregate metrics above threshold
    assert summary.aggregate.get("refusal_calibration", 0) >= 0.8
    assert summary.aggregate.get("calculation_correctness", 0) >= 0.7


@pytest.mark.asyncio
async def test_ood_examples_all_refuse(pipeline, examples) -> None:
    ood = [e for e in examples if e.expected_refusal]
    assert len(ood) >= 5
    refused = 0
    for ex in ood:
        outcome = await pipeline.generator.generate(ex.query, top_k=5)
        score = refusal_calibration(outcome.response, ex)
        assert score.value == 1.0, (
            f"{ex.id}: expected refusal but got {outcome.response.confidence}"
        )
        refused += 1
    assert refused == len(ood)


@pytest.mark.asyncio
async def test_calculation_examples_match_ground_truth(pipeline, examples) -> None:
    calc_examples = [
        e
        for e in examples
        if not e.expected_refusal and e.expected_values
    ]
    assert len(calc_examples) >= 6
    for ex in calc_examples:
        outcome = await pipeline.generator.generate(ex.query, top_k=10)
        score = calculation_correctness(
            outcome.response, ex.expected_values, tolerance=ex.numeric_tolerance
        )
        assert score.value >= 0.99, (
            f"{ex.id}: calc_correctness={score.value}, details={score.details}"
        )


@pytest.mark.asyncio
async def test_faithfulness_and_citation_accuracy(pipeline, examples) -> None:
    non_ood = [e for e in examples if not e.expected_refusal]
    faith_total = 0.0
    cit_total = 0.0
    for ex in non_ood:
        outcome = await pipeline.generator.generate(ex.query, top_k=8)
        faith_total += faithfulness(outcome.response, outcome.retrieved_chunks).value
        cit_total += citation_accuracy(outcome.response, outcome.retrieved_chunks).value
    faith_avg = faith_total / len(non_ood)
    cit_avg = cit_total / len(non_ood)
    assert faith_avg >= 0.7, faith_avg
    assert cit_avg >= 0.8, cit_avg


@pytest.mark.asyncio
async def test_retrieval_recall_for_calculation_examples(pipeline, examples) -> None:
    calc = [e for e in examples if not e.expected_refusal and e.expected_paragraph_refs]
    recalls: list[float] = []
    for ex in calc:
        outcome = await pipeline.generator.generate(ex.query, top_k=10)
        r = retrieval_recall_at_k(outcome.retrieved_chunks, ex.expected_paragraph_refs, k=10)
        recalls.append(r.value)
    avg = sum(recalls) / len(recalls)
    assert avg >= 0.8, avg


def test_regression_comparison_detects_drops() -> None:
    """Meta-test: regression module correctly flags metric drops."""
    from anvil.evaluation.runner import RunSummary

    baseline = RunSummary(
        per_example=[], aggregate={"faithfulness": 0.9, "citation_accuracy": 0.95}, pass_rate=1.0
    )
    worse = RunSummary(
        per_example=[], aggregate={"faithfulness": 0.7, "citation_accuracy": 0.95}, pass_rate=0.8
    )
    report = compare_runs(baseline, worse, tolerance=0.05)
    assert report.has_regression
    assert "faithfulness" in report.regressed_metrics
