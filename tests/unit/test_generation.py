"""Phase 4 integration-ish tests for the generator (Fake backend)."""

from __future__ import annotations

import pytest

from anvil.generation.calculation_engine import CalculationInputs


@pytest.mark.asyncio
async def test_generate_calculation_query_succeeds(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "Calculate thickness for cylindrical shell ID=1800 mm, P=1.5 MPa, "
        "T=350°C, SM-516 Gr 70, Type 1 with full RT, CA=3.0 mm.",
        top_k=8,
    )
    resp = outcome.response
    assert resp.confidence.value != "insufficient"
    assert len(resp.calculation_steps) >= 5
    # The final step's result should be MAWP ≈ 1.633 MPa
    final = resp.calculation_steps[-1]
    assert final.unit == "MPa"
    assert abs(final.result - 1.633) < 0.01


@pytest.mark.asyncio
async def test_generate_with_explicit_inputs(pipeline) -> None:
    inp = CalculationInputs(
        component="cylindrical_shell",
        P_mpa=2.0,
        design_temp_c=200,
        material="SM-516 Gr 70",
        joint_type=1,
        rt_level="Full RT",
        corrosion_allowance_mm=2.0,
        inside_diameter_mm=1500,
    )
    outcome = await pipeline.generator.generate(
        "compute thickness", top_k=5, calculation_inputs=inp
    )
    assert outcome.calculation is not None
    assert abs(outcome.calculation.t_min_mm - 11.30) < 0.02


@pytest.mark.asyncio
async def test_generate_refuses_ood(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "What is the weather in San Jose today?", top_k=5
    )
    assert outcome.response.confidence.value == "insufficient"
    assert outcome.response.refusal_reason is not None


@pytest.mark.asyncio
async def test_generate_refuses_unknown_material(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "Compute thickness for SM-999 Gr XYZ, ID=1000 mm, P=1 MPa, T=200°C, "
        "Type 1 Full RT, CA=2 mm.",
        top_k=5,
    )
    assert outcome.response.confidence.value == "insufficient"


@pytest.mark.asyncio
async def test_generate_refuses_over_temp(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "Compute thickness for SM-516 Gr 70 at 700°C, ID=1000 mm, P=1 MPa, "
        "Type 1 Full RT, CA=2 mm.",
        top_k=5,
    )
    assert outcome.response.confidence.value == "insufficient"
    assert "700" in (outcome.response.refusal_reason or "")
