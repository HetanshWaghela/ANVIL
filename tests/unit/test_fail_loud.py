"""Fail-loud regression tests.

Every test here asserts that a particular failure mode is surfaced to the
caller — NOT silently swallowed, downgraded, or papered over with a fallback.
If any of these start passing (i.e. the behavior reverts to a silent
fallback), a safety-relevant regression has been introduced.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from anvil import CalculationError, GenerationError, RetrievalError
from anvil.generation.calculation_engine import (
    CalculationInputs,
    CitationBuilder,
)
from anvil.generation.citation_enforcer import _paragraph_refs_compatible
from anvil.generation.llm_backend import (
    InstructorBackend,
    OpenAICompatibleBackend,
    get_default_backend,
)

# ---------------------------------------------------------------------------
# Calculation engine must fail loud on invariant violations.
# ---------------------------------------------------------------------------


def test_citation_builder_material_not_found_raises(parsed_elements) -> None:
    """Asking to cite a material that isn't in Table M-1 must raise —
    never silently quote the section intro."""
    builder = CitationBuilder.from_elements(parsed_elements)
    with pytest.raises(CalculationError) as excinfo:
        builder.for_material("SM-999 UnobtainiumTitanium")
    # The error message must identify WHICH material failed so debugging is fast.
    assert "SM-999" in str(excinfo.value)
    assert "Table M-1" in str(excinfo.value)


def test_citation_builder_empty_material_raises(parsed_elements) -> None:
    builder = CitationBuilder.from_elements(parsed_elements)
    with pytest.raises(CalculationError):
        builder.for_material("")
    with pytest.raises(CalculationError):
        builder.for_material("   ")


def test_citation_builder_invalid_joint_type_raises(parsed_elements) -> None:
    """Citing a non-existent joint type must raise, not fall back to intro."""
    builder = CitationBuilder.from_elements(parsed_elements)
    with pytest.raises(CalculationError) as excinfo:
        builder.for_joint_efficiency(joint_type=99)
    assert "99" in str(excinfo.value)
    assert "B-12" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Paragraph-ref boundary matching must NOT admit 'A-2' matching 'A-27'.
# ---------------------------------------------------------------------------


def test_paragraph_ref_boundary_rejects_digit_prefix() -> None:
    """`A-2` is NOT a prefix of `A-27`; the prefix match must respect
    paragraph-number boundaries."""
    assert not _paragraph_refs_compatible("A-2", "A-27")
    assert not _paragraph_refs_compatible("A-27", "A-2")


def test_paragraph_ref_boundary_accepts_subparagraph() -> None:
    """Sub-paragraph markers `(c)(1)` are a legitimate boundary."""
    assert _paragraph_refs_compatible("A-27(c)(1)", "A-27")
    assert _paragraph_refs_compatible("A-27", "A-27(c)(1)")
    # Table prefix stripped on both sides
    assert _paragraph_refs_compatible("Table M-1", "M-1")


def test_paragraph_ref_exact_match() -> None:
    assert _paragraph_refs_compatible("A-27", "A-27")
    assert _paragraph_refs_compatible("B-12", "B-12")


# ---------------------------------------------------------------------------
# Generator must refuse loudly on calculation failure, not swallow.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_calc_failure_produces_refusal(pipeline) -> None:
    """If the engine raises CalculationError during generate(), the response
    must be a refusal carrying the engine's reason — not a free-form LLM
    answer that silently omits the number.
    """
    bad_inputs = CalculationInputs(
        component="cylindrical_shell",
        P_mpa=1.5,
        design_temp_c=700,  # exceeds SM-516 Gr 70 max_temp
        material="SM-516 Gr 70",
        joint_type=1,
        rt_level="Full RT",
        corrosion_allowance_mm=3.0,
        inside_diameter_mm=1800,
    )
    outcome = await pipeline.generator.generate(
        "compute thickness", calculation_inputs=bad_inputs
    )
    assert outcome.response.confidence.value == "insufficient"
    assert outcome.response.refusal_reason is not None
    # The refusal text must explain WHY in domain terms.
    reason = outcome.response.refusal_reason.lower()
    assert "700" in reason or "max" in reason or "temperature" in reason


# ---------------------------------------------------------------------------
# Backend selection must be explicit. Unknown values raise. Fake logs warning.
# ---------------------------------------------------------------------------


def test_unknown_backend_raises() -> None:
    with patch.dict(os.environ, {"ANVIL_LLM_BACKEND": "gpt5"}, clear=False):
        with pytest.raises(GenerationError) as excinfo:
            get_default_backend()
        assert "gpt5" in str(excinfo.value)


def test_instructor_backend_requires_model_env() -> None:
    with patch.dict(
        os.environ,
        {"ANVIL_LLM_BACKEND": "instructor", "ANVIL_LLM_MODEL": ""},
        clear=False,
    ):
        with pytest.raises(GenerationError) as excinfo:
            get_default_backend()
        assert "ANVIL_LLM_MODEL" in str(excinfo.value)


def test_instructor_backend_requires_model_arg() -> None:
    with pytest.raises(GenerationError):
        InstructorBackend(model="")


def test_nvidia_nim_backend_requires_api_key() -> None:
    # Remove NVIDIA_API_KEY from environment for the test
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    env["ANVIL_LLM_BACKEND"] = "nvidia_nim"
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(GenerationError) as excinfo:
            get_default_backend()
        assert "NVIDIA_API_KEY" in str(excinfo.value)


def test_openai_compatible_backend_requires_all_env() -> None:
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "ANVIL_LLM_MODEL")}
    env["ANVIL_LLM_BACKEND"] = "openai_compatible"
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(GenerationError) as excinfo:
            get_default_backend()
        msg = str(excinfo.value)
        assert "OPENAI_COMPAT_BASE_URL" in msg
        assert "OPENAI_COMPAT_API_KEY" in msg
        assert "ANVIL_LLM_MODEL" in msg


def test_openai_compatible_backend_requires_api_key_on_direct_construction() -> None:
    """Constructing OpenAICompatibleBackend with no api_key must raise."""
    with pytest.raises(GenerationError):
        OpenAICompatibleBackend(
            base_url="https://example.com/v1",
            api_key=None,
            model="some-model",
        )


def test_openai_compatible_backend_requires_base_url() -> None:
    with pytest.raises(GenerationError):
        OpenAICompatibleBackend(base_url="", api_key="k", model="m")


def test_openai_compatible_backend_requires_model() -> None:
    with pytest.raises(GenerationError):
        OpenAICompatibleBackend(base_url="u", api_key="k", model="")


def test_nvidia_nim_backend_selection_via_env(monkeypatch) -> None:
    """With a dummy API key, the factory must construct a NIM backend
    pointed at the NIM base URL without making any network calls."""
    monkeypatch.setenv("ANVIL_LLM_BACKEND", "nvidia_nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    monkeypatch.setenv("ANVIL_LLM_MODEL", "deepseek-ai/deepseek-v3.1")
    backend = get_default_backend()
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.base_url == "https://integrate.api.nvidia.com/v1"
    assert backend.model == "deepseek-ai/deepseek-v3.1"


def test_nvidia_nim_backend_reasoning_extra_body(monkeypatch) -> None:
    """ANVIL_NIM_REASONING=1 should wire chat_template_kwargs into extra_body."""
    monkeypatch.setenv("ANVIL_LLM_BACKEND", "nvidia_nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    monkeypatch.setenv("ANVIL_NIM_REASONING", "1")
    monkeypatch.setenv("ANVIL_NIM_REASONING_EFFORT", "high")
    backend = get_default_backend()
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.extra_body is not None
    ctk = backend.extra_body.get("chat_template_kwargs", {})
    assert ctk.get("thinking") is True
    assert ctk.get("reasoning_effort") == "high"


# ---------------------------------------------------------------------------
# Embedder selection: unknown values must raise.
# ---------------------------------------------------------------------------


def test_unknown_embedder_raises(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_EMBEDDER", "mystical_bge")
    from anvil.retrieval.embedder import get_default_embedder

    with pytest.raises(RetrievalError) as excinfo:
        get_default_embedder()
    assert "mystical_bge" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _summarize_calculation must raise if invariants are broken.
# ---------------------------------------------------------------------------


def test_summarize_calculation_raises_on_missing_min_thickness_step() -> None:
    """If a CalculationResult somehow has no MIN_THICKNESS step, the
    summariser must raise rather than emit an empty-formula prompt."""
    from anvil.generation.calculation_engine import CalculationInputs, CalculationResult
    from anvil.generation.generator import _summarize_calculation

    inp = CalculationInputs(
        component="cylindrical_shell",
        P_mpa=1.5,
        design_temp_c=350,
        material="SM-516 Gr 70",
        joint_type=1,
        rt_level="Full RT",
        corrosion_allowance_mm=3.0,
        inside_diameter_mm=1800,
    )
    bogus = CalculationResult(
        inputs=inp,
        S_mpa=114.0,
        E=1.0,
        R_mm=900.0,
        formula_ref="A-27(c)(1)",
        applicability_lhs=1.5,
        applicability_rhs=43.89,
        applicability_ok=True,
        t_min_mm=11.94,
        t_design_mm=14.94,
        t_nominal_mm=16,
        mawp_mpa=1.633,
        steps=[],  # <-- invariant violation
        warnings=[],
    )
    with pytest.raises(CalculationError) as excinfo:
        _summarize_calculation(bogus)
    assert "MIN_THICKNESS" in str(excinfo.value)
