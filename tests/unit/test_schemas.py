"""Pydantic schema behavior tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anvil.schemas import (
    AnvilResponse,
    CalculationStep,
    Citation,
    InputValue,
    LLMAnvilResponse,
    ResponseConfidence,
    StepKey,
)


def _cite() -> Citation:
    return Citation(
        source_element_id="sec-a-27",
        paragraph_ref="A-27(c)(1)",
        quoted_text="t = (P × R) / (S × E − 0.6 × P)",
        page_number=1,
    )


def test_citation_requires_non_empty_quote() -> None:
    with pytest.raises(ValidationError):
        Citation(
            source_element_id="x",
            paragraph_ref="A-27",
            quoted_text="",
            page_number=1,
        )


def test_citation_requires_positive_page() -> None:
    with pytest.raises(ValidationError):
        Citation(
            source_element_id="x",
            paragraph_ref="A-27",
            quoted_text="q",
            page_number=0,
        )


def test_refusal_requires_reason() -> None:
    with pytest.raises(ValidationError):
        AnvilResponse(
            query="q",
            answer="a",
            confidence=ResponseConfidence.INSUFFICIENT,
        )


def test_non_refusal_requires_citations() -> None:
    with pytest.raises(ValidationError):
        AnvilResponse(
            query="q",
            answer="a",
            confidence=ResponseConfidence.HIGH,
            citations=[],
        )


def test_refusal_reason_forbidden_when_confident() -> None:
    with pytest.raises(ValidationError):
        AnvilResponse(
            query="q",
            answer="a",
            confidence=ResponseConfidence.HIGH,
            citations=[_cite()],
            refusal_reason="bogus",
        )


def test_valid_high_response() -> None:
    resp = AnvilResponse(
        query="q",
        answer="a",
        citations=[_cite()],
        confidence=ResponseConfidence.HIGH,
    )
    assert resp.confidence == ResponseConfidence.HIGH
    assert len(resp.citations) == 1


def test_valid_refusal_response() -> None:
    resp = AnvilResponse(
        query="q",
        answer="a",
        confidence=ResponseConfidence.INSUFFICIENT,
        refusal_reason="missing context",
    )
    assert resp.refusal_reason == "missing context"


def test_llm_response_excludes_untrusted_calculation_steps() -> None:
    """LLM output may contain garbage calculation_steps; the host ignores them."""
    llm_resp = LLMAnvilResponse.model_validate(
        {
            "query": "q",
            "answer": "a",
            "citations": [_cite().model_dump()],
            "confidence": "high",
            "calculation_steps": [
                {
                    "step_number": 1,
                    "result_key": "minimum_required_thickness",
                    "description": "model-invented step",
                }
            ],
        }
    )

    trusted_step = CalculationStep(
        step_number=1,
        result_key=StepKey.MIN_THICKNESS,
        description="compute thickness",
        formula="t = (P*R)/(S*E - 0.6*P)",
        inputs={
            "P": InputValue(symbol="P", value=1.5, unit="MPa", source="user_input")
        },
        result=11.94,
        unit="mm",
        citation=_cite(),
    )
    resp = AnvilResponse(**llm_resp.model_dump(), calculation_steps=[trusted_step])

    assert resp.calculation_steps == [trusted_step]


def test_calculation_step_structure() -> None:
    step = CalculationStep(
        step_number=1,
        result_key=StepKey.MIN_THICKNESS,
        description="compute thickness",
        formula="t = (P*R)/(S*E - 0.6*P)",
        inputs={
            "P": InputValue(symbol="P", value=1.5, unit="MPa", source="user_input")
        },
        result=11.94,
        unit="mm",
        citation=_cite(),
    )
    assert step.step_number == 1
    assert step.result_key == StepKey.MIN_THICKNESS
    assert step.inputs["P"].value == 1.5


def test_calculation_step_result_key_is_required() -> None:
    """result_key is what downstream consumers key off — must never be missing."""
    with pytest.raises(ValidationError):
        CalculationStep(
            step_number=1,
            description="compute",
            formula="t = (P*R)/(S*E - 0.6*P)",
            inputs={},
            result=10.0,
            unit="mm",
            citation=_cite(),
        )
