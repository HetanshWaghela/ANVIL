"""Generation layer — evidence-enforced structured responses."""

from __future__ import annotations

from anvil.generation.calculation_engine import (
    CalculationEngine,
    CalculationInputs,
    CalculationResult,
)
from anvil.generation.citation_enforcer import (
    CitationValidationResult,
    validate_citations,
)
from anvil.generation.generator import AnvilGenerator, LLMBackend
from anvil.generation.llm_backend import (
    FakeLLMBackend,
    InstructorBackend,
    OpenAICompatibleBackend,
    get_default_backend,
)
from anvil.generation.prompt_builder import build_context_prompt
from anvil.generation.refusal_gate import (
    RELEVANCE_THRESHOLD,
    RefusalDecision,
    should_refuse,
)

__all__ = [
    "AnvilGenerator",
    "CalculationEngine",
    "CalculationInputs",
    "CalculationResult",
    "CitationValidationResult",
    "FakeLLMBackend",
    "InstructorBackend",
    "LLMBackend",
    "OpenAICompatibleBackend",
    "RELEVANCE_THRESHOLD",
    "RefusalDecision",
    "build_context_prompt",
    "get_default_backend",
    "should_refuse",
    "validate_citations",
]
