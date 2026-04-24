"""Evaluation framework."""

from __future__ import annotations

from anvil.evaluation.dataset import load_golden_dataset
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
from anvil.evaluation.regression import RegressionReport, compare_runs
from anvil.evaluation.runner import EvaluationRunner, RunSummary

__all__ = [
    "EvaluationRunner",
    "RegressionReport",
    "RunSummary",
    "calculation_correctness",
    "citation_accuracy",
    "compare_runs",
    "entity_grounding",
    "faithfulness",
    "load_golden_dataset",
    "refusal_calibration",
    "retrieval_precision_at_k",
    "retrieval_recall_at_k",
    "structural_completeness",
]
