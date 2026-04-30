"""Generation-layer schemas: the structured response contract."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """A citation binds a claim to a specific source element and quoted text."""

    source_element_id: str = Field(description="ID of the document element cited")
    paragraph_ref: str = Field(description="Paragraph reference, e.g. 'A-27(c)(1)'")
    quoted_text: str = Field(
        min_length=1,
        description="The exact text from the source supporting this claim",
    )
    page_number: int = Field(ge=1)


class InputValue(BaseModel):
    """A single input value in a calculation, with provenance."""

    symbol: str
    value: float
    unit: str
    source: str = Field(
        description="Origin tag: 'user_input' | 'table_lookup' | 'calculated' | 'pinned_data'"
    )
    citation: Citation | None = None


class StepKey(StrEnum):
    """Stable identifier for the semantic role of a calculation step.

    Downstream consumers (metrics, UI, auditors) match on this enum rather
    than parsing the human-readable `description` — so rewording a step's
    description never silently breaks evaluation.
    """

    ALLOWABLE_STRESS = "allowable_stress"
    JOINT_EFFICIENCY = "joint_efficiency"
    APPLICABILITY_CHECK = "applicability_check"
    MIN_THICKNESS = "min_thickness"
    DESIGN_THICKNESS = "design_thickness"
    NOMINAL_PLATE = "nominal_plate"
    MAWP = "mawp"


class CalculationStep(BaseModel):
    """A single, traceable step in a calculation."""

    step_number: int = Field(ge=1)
    result_key: StepKey = Field(
        description=(
            "Stable, machine-readable identifier for the semantic role of "
            "this step. Metrics and auditors key off this, not `description`."
        )
    )
    description: str
    formula: str = Field(description="Formula applied, e.g. 't = (P × R) / (S × E − 0.6 × P)'")
    inputs: dict[str, InputValue]
    result: float
    unit: str
    citation: Citation


class ResponseConfidence(StrEnum):
    """Calibrated confidence tier for a response."""

    HIGH = "high"
    MEDIUM = "medium"
    INSUFFICIENT = "insufficient"


class AnvilResponse(BaseModel):
    """Top-level structured response from the anvil pipeline."""

    query: str
    answer: str
    citations: list[Citation] = Field(
        default_factory=list,
        description="Every factual claim must be cited. Empty only for refusals.",
    )
    calculation_steps: list[CalculationStep] = Field(default_factory=list)
    confidence: ResponseConfidence
    refusal_reason: str | None = Field(
        default=None,
        description="If confidence is 'insufficient', explain why.",
    )
    retrieved_context_ids: list[str] = Field(
        default_factory=list,
        description="IDs of all retrieved elements the response was conditioned on",
    )

    @model_validator(mode="after")
    def validate_refusal_consistency(self) -> AnvilResponse:
        if self.confidence == ResponseConfidence.INSUFFICIENT and not self.refusal_reason:
            raise ValueError("Must provide refusal_reason when confidence is insufficient")
        if self.confidence != ResponseConfidence.INSUFFICIENT and self.refusal_reason:
            raise ValueError("refusal_reason should only be set when confidence is insufficient")
        if self.confidence != ResponseConfidence.INSUFFICIENT and not self.citations:
            raise ValueError("Non-refusal responses must have at least one citation")
        return self


class LLMAnvilResponse(BaseModel):
    """Subset of `AnvilResponse` that the LLM is allowed to produce.

    Calculation steps are deliberately absent. They are computed and injected by
    the deterministic host-side calculation engine, so a model cannot invent a
    step key, formula, input value, or citation that fails schema validation
    before the trusted steps are attached.
    """

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ResponseConfidence
    refusal_reason: str | None = None
    retrieved_context_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_refusal_consistency(self) -> LLMAnvilResponse:
        if self.confidence == ResponseConfidence.INSUFFICIENT and not self.refusal_reason:
            raise ValueError("Must provide refusal_reason when confidence is insufficient")
        if self.confidence != ResponseConfidence.INSUFFICIENT and self.refusal_reason:
            # LLMs frequently include a refusal_reason alongside a valid
            # answer. Auto-correct rather than crash — the answer and
            # citations are still usable.
            self.refusal_reason = None
        if self.confidence != ResponseConfidence.INSUFFICIENT and not self.citations:
            raise ValueError("Non-refusal responses must have at least one citation")
        return self
