"""Evaluation framework."""

from __future__ import annotations

from anvil.evaluation.agent_runner import (
    AgentEvaluationRunner,
    AgentRunSummary,
    transcripts_to_jsonable,
)
from anvil.evaluation.dataset import load_golden_dataset
from anvil.evaluation.manifest import (
    RunManifest,
    build_manifest,
    dataset_hash,
    make_run_id,
    redact_key,
)
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
from anvil.evaluation.report_writer import render_report, write_report
from anvil.evaluation.run_logger import (
    RunLogger,
    RunLoggerConfig,
    make_default_logger,
)
from anvil.evaluation.runner import EvaluationRunner, RunSummary

__all__ = [
    "AgentEvaluationRunner",
    "AgentRunSummary",
    "EvaluationRunner",
    "RegressionReport",
    "RunLogger",
    "RunLoggerConfig",
    "RunManifest",
    "RunSummary",
    "transcripts_to_jsonable",
    "build_manifest",
    "calculation_correctness",
    "citation_accuracy",
    "compare_runs",
    "dataset_hash",
    "entity_grounding",
    "faithfulness",
    "load_golden_dataset",
    "make_default_logger",
    "make_run_id",
    "redact_key",
    "refusal_calibration",
    "render_report",
    "retrieval_precision_at_k",
    "retrieval_recall_at_k",
    "structural_completeness",
    "write_report",
]
