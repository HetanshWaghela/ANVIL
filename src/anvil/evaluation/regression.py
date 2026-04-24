"""Regression comparison between evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

from anvil.evaluation.runner import RunSummary


@dataclass
class RegressionReport:
    baseline_pass_rate: float
    current_pass_rate: float
    metric_deltas: dict[str, float]
    regressed_metrics: list[str]
    improved_metrics: list[str]

    @property
    def has_regression(self) -> bool:
        return len(self.regressed_metrics) > 0


def compare_runs(
    baseline: RunSummary, current: RunSummary, tolerance: float = 0.01
) -> RegressionReport:
    """Compare aggregate metrics between two runs.

    A metric regresses when its current value is lower than the baseline
    by more than `tolerance`.
    """
    deltas: dict[str, float] = {}
    regressed: list[str] = []
    improved: list[str] = []
    for name, base_val in baseline.aggregate.items():
        curr_val = current.aggregate.get(name, 0.0)
        delta = curr_val - base_val
        deltas[name] = delta
        if delta < -tolerance:
            regressed.append(name)
        elif delta > tolerance:
            improved.append(name)
    return RegressionReport(
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        metric_deltas=deltas,
        regressed_metrics=regressed,
        improved_metrics=improved,
    )
