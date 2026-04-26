"""REGRESSION: a backend that fails Pydantic validation must produce a
soft-refusal AnvilResponse, not crash the eval loop.

Surfaced by the M5 real-NIM run: llama-3.3-70b emitted a calculation
step with `citation: null`, instructor exhausted retries, and the
GenerationError propagated up to kill the entire 30-example eval. The
fix in `anvil.generation.generator.AnvilGenerator.generate` catches
the failure, re-invokes the backend with a `refusal_reason`, and lets
the rest of the eval proceed.
"""

from __future__ import annotations

import pytest

from anvil import GenerationError, RetryableGenerationError
from anvil.generation.generator import AnvilGenerator
from anvil.generation.llm_backend import FakeLLMBackend, LLMBackend
from anvil.pipeline import build_pipeline
from anvil.schemas.generation import AnvilResponse, ResponseConfidence
from anvil.schemas.retrieval import RetrievedChunk


class _FlakyBackend(FakeLLMBackend):
    """Mimics the real-NIM failure mode: raises GenerationError on the
    first (real) call, but happily returns a refusal when re-invoked
    with `refusal_reason`. This is the same contract every backend in
    `llm_backend.py` honors — see the `if refusal_reason is not None`
    short-circuits."""

    def __init__(self) -> None:
        super().__init__()
        self.real_calls = 0
        self.refusal_calls = 0

    async def generate(  # type: ignore[override]
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list,  # type: ignore[type-arg]
        refusal_reason: str | None = None,
    ) -> AnvilResponse:
        if refusal_reason is None:
            self.real_calls += 1
            raise GenerationError(
                "instructor exhausted retries: 1 validation error for "
                "AnvilResponse calculation_steps.2.citation"
            )
        self.refusal_calls += 1
        return await super().generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=query,
            retrieved_chunks=retrieved_chunks,
            calculation_steps=calculation_steps,
            refusal_reason=refusal_reason,
        )


class _RetryableOnceBackend(FakeLLMBackend):
    """Mimics a hosted-provider 429/timeout that succeeds after cooldown."""

    def __init__(self) -> None:
        super().__init__()
        self.real_calls = 0

    async def generate(  # type: ignore[override]
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list,  # type: ignore[type-arg]
        refusal_reason: str | None = None,
    ) -> AnvilResponse:
        if refusal_reason is None:
            self.real_calls += 1
            if self.real_calls == 1:
                raise RetryableGenerationError("HTTP 429: Too Many Requests")
        return await super().generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=query,
            retrieved_chunks=retrieved_chunks,
            calculation_steps=calculation_steps,
            refusal_reason=refusal_reason,
        )


@pytest.mark.asyncio
async def test_backend_validation_failure_becomes_soft_refusal() -> None:
    """When the backend raises GenerationError on a non-refusal path,
    `generate` must NOT propagate it — the response must come back
    shaped as a refusal with the error captured in `refusal_reason`."""
    pipe = build_pipeline()
    flaky: LLMBackend = _FlakyBackend()
    gen = AnvilGenerator(
        retriever=pipe.retriever,
        backend=flaky,
        calc_engine=pipe.generator.calc_engine,
        element_index=pipe.generator.element_index,
    )

    # A non-OOD query so we get past the refusal gate and hit step 5.
    outcome = await gen.generate(
        "What is the joint efficiency for a Type 1 joint with full radiography?",
        top_k=10,
    )

    assert outcome.response.refusal_reason is not None
    assert "exhausted retries" in outcome.response.refusal_reason
    assert outcome.response.confidence == ResponseConfidence.INSUFFICIENT
    # Both the failed real call AND the soft-refusal re-call must have
    # happened — exactly two backend invocations per failed example.
    assert flaky.real_calls == 1, "real backend call must have been attempted"  # type: ignore[attr-defined]
    assert flaky.refusal_calls == 1, (  # type: ignore[attr-defined]
        "soft-refusal recall must have been issued"
    )


@pytest.mark.asyncio
async def test_eval_run_completes_when_one_example_fails() -> None:
    """A single bad example must NOT abort a multi-example run. Pre-fix,
    a GenerationError on example 26/30 killed the whole eval."""
    from anvil.evaluation.dataset import load_golden_dataset
    from anvil.evaluation.runner import EvaluationRunner
    from tests.conftest import GOLDEN_DATASET_JSON

    pipe = build_pipeline()
    flaky = _FlakyBackend()
    gen = AnvilGenerator(
        retriever=pipe.retriever,
        backend=flaky,
        calc_engine=pipe.generator.calc_engine,
        element_index=pipe.generator.element_index,
    )
    runner = EvaluationRunner(gen)
    examples = load_golden_dataset(GOLDEN_DATASET_JSON)[:3]
    summary = await runner.run(examples)
    # All three must come back; none crashed.
    assert len(summary.per_example) == 3
    # Every refusal-gated outcome carries the soft-refusal reason —
    # at least one of the in-domain examples actually exercised the
    # try/except path.
    soft_refusals = [
        r
        for r in summary.outcomes
        if r.response.refusal_reason
        and "exhausted retries" in r.response.refusal_reason
    ]
    assert soft_refusals, "soft-refusal path must have triggered at least once"


@pytest.mark.asyncio
async def test_eval_runner_defers_retryable_generation_errors() -> None:
    """Provider 429s/timeouts must be retried later, not scored as answers."""
    from anvil.evaluation.dataset import load_golden_dataset
    from anvil.evaluation.runner import EvaluationRunner
    from tests.conftest import GOLDEN_DATASET_JSON

    pipe = build_pipeline()
    backend = _RetryableOnceBackend()
    gen = AnvilGenerator(
        retriever=pipe.retriever,
        backend=backend,
        calc_engine=pipe.generator.calc_engine,
        element_index=pipe.generator.element_index,
    )
    runner = EvaluationRunner(
        gen,
        retryable_attempts=2,
        retryable_cooldown_s=0,
    )
    examples = load_golden_dataset(GOLDEN_DATASET_JSON)[:1]
    summary = await runner.run(examples)

    assert len(summary.per_example) == 1
    assert len(summary.outcomes) == 1
    assert summary.outcomes[0].response.confidence != ResponseConfidence.INSUFFICIENT
    assert backend.real_calls == 2
