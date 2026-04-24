"""Evaluation-layer schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    """A single entry in the golden dataset."""

    id: str
    query: str
    category: str = Field(
        description="'calculation' | 'lookup' | 'cross_reference' | 'out_of_domain' | 'edge_case'"
    )
    expected_refusal: bool = False
    expected_paragraph_refs: list[str] = Field(
        default_factory=list,
        description="Paragraph refs that must be present in retrieval results",
    )
    expected_values: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric values that must appear in the response (e.g. t_min_mm=11.94)",
    )
    numeric_tolerance: float = Field(default=0.02, ge=0)
    expected_materials: list[str] = Field(default_factory=list)
    notes: str | None = None


class MetricScore(BaseModel):
    """A single metric value with diagnostic metadata."""

    name: str
    value: float
    passed: bool
    threshold: float | None = None
    details: dict[str, float | str | int] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Per-example evaluation result."""

    example_id: str
    category: str
    metrics: list[MetricScore]
    passed: bool

    def get_metric(self, name: str) -> MetricScore | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None
