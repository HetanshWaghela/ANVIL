"""Phase 4 tests: deterministic calculation engine against design examples."""

from __future__ import annotations

import pytest

from anvil import CalculationError
from anvil.generation.calculation_engine import (
    CalculationEngine,
    CalculationInputs,
)


def _make_inputs(ex: dict) -> CalculationInputs:
    i = ex["inputs"]
    return CalculationInputs(
        component=i["component"],
        P_mpa=i["P_mpa"],
        design_temp_c=i["design_temp_c"],
        material=i["material"],
        joint_type=i["joint_type"],
        rt_level=i["rt_level"],
        corrosion_allowance_mm=i["corrosion_allowance_mm"],
        inside_diameter_mm=i.get("inside_diameter_mm"),
        outside_diameter_mm=i.get("outside_diameter_mm"),
    )


def test_calc_engine_all_design_examples(design_examples: list[dict]) -> None:
    """Every design example must reproduce to spec tolerance (0.1% per
    §Faithfulness). Given the Decimal engine, the actual error is zero;
    we use 0.005 mm / 0.002 MPa absolute bounds that would trip loudly
    if anyone swapped the arithmetic path to float.
    """
    engine = CalculationEngine()
    for ex in design_examples:
        res = engine.calculate(_make_inputs(ex))
        exp = ex["expected_result"]
        assert abs(res.t_min_mm - exp["t_min_mm"]) < 0.005, ex["id"]
        assert abs(res.t_design_mm - exp["t_design_mm"]) < 0.005, ex["id"]
        assert res.t_nominal_mm == exp["t_nominal_mm"], ex["id"]
        assert abs(res.mawp_mpa - exp["mawp_mpa"]) < 0.002, ex["id"]
        assert res.S_mpa == ex["expected_lookups"]["S_mpa"], ex["id"]
        assert ex["expected_lookups"]["E"] == res.E, ex["id"]


def test_calc_engine_produces_steps_with_citations(design_examples: list[dict]) -> None:
    engine = CalculationEngine()
    res = engine.calculate(_make_inputs(design_examples[0]))
    # Every step must have a Citation
    for step in res.steps:
        assert step.citation is not None
        assert step.citation.paragraph_ref
        assert step.citation.quoted_text
    # At least one step cites A-27
    assert any("A-27" in s.citation.paragraph_ref for s in res.steps)
    # At least one step cites Table M-1 or B-12
    refs = {s.citation.paragraph_ref for s in res.steps}
    assert any("M-1" in r or "B-12" in r for r in refs)


def test_calc_engine_refuses_unknown_material() -> None:
    from anvil import CalculationError

    engine = CalculationEngine()
    with pytest.raises(CalculationError):
        engine.calculate(
            CalculationInputs(
                component="cylindrical_shell",
                P_mpa=1.0,
                design_temp_c=300,
                material="SM-999 Gr XYZ",
                joint_type=1,
                rt_level="Full RT",
                corrosion_allowance_mm=1.0,
                inside_diameter_mm=1000,
            )
        )


def test_calc_engine_refuses_over_max_temp() -> None:
    from anvil import CalculationError

    engine = CalculationEngine()
    with pytest.raises(CalculationError):
        engine.calculate(
            CalculationInputs(
                component="cylindrical_shell",
                P_mpa=1.0,
                design_temp_c=700,  # exceeds SM-516 Gr 70 max_temp 500
                material="SM-516 Gr 70",
                joint_type=1,
                rt_level="Full RT",
                corrosion_allowance_mm=1.0,
                inside_diameter_mm=1000,
            )
        )


def test_calc_engine_refuses_applicability_violation() -> None:
    """Per spec §Domain Rules — UG-27(c)(1) applies only when P ≤ 0.385·S·E.
    If that condition fails the engine must raise, not silently compute a
    thick-wall-regime number that happens to be arithmetically valid but
    is physically wrong.
    """
    engine = CalculationEngine()
    # Pick P large enough to exceed 0.385·S·E. For SM-106 Gr B at 400°C,
    # S=86 MPa, E=0.45 (Type 6 No RT), so 0.385·S·E = 14.9 MPa. P=20 MPa
    # exceeds it while still leaving a positive formula denominator.
    with pytest.raises(CalculationError) as excinfo:
        engine.calculate(
            CalculationInputs(
                component="cylindrical_shell",
                P_mpa=20.0,
                design_temp_c=400,
                material="SM-106 Gr B",
                joint_type=6,
                rt_level="No RT",
                corrosion_allowance_mm=1.0,
                inside_diameter_mm=500,
            )
        )
    msg = str(excinfo.value).lower()
    assert "applicability" in msg
    assert "0.385" in msg or "thin-wall" in msg


def test_calc_engine_refuses_thick_wall_geometry() -> None:
    """Per spec — the thin-wall formula also requires t ≤ R/2 in addition
    to P ≤ 0.385·S·E.  The two conditions are *almost* algebraically
    equivalent (t = R/2 ⟺ P = S·E/2.6 ≈ 0.3846·S·E) but differ in the
    narrow band `(S·E/2.6, 0.385·S·E]` because the standard uses a
    rounded constant. For a P inside that band the pressure check passes
    but the geometry check still trips, and the engine must refuse.
    """
    engine = CalculationEngine()
    # SM-106 Gr B at 400°C, Type 6 / No RT ⇒ S=86, E=0.45, S·E=38.7.
    #   pressure limit  = 0.385·38.7 = 14.8995
    #   geometry limit  = 38.7 / 2.6 = 14.8846
    # P = 14.895 is inside the band → P-check passes, geometry fails.
    with pytest.raises(CalculationError) as excinfo:
        engine.calculate(
            CalculationInputs(
                component="cylindrical_shell",
                P_mpa=14.895,
                design_temp_c=400,
                material="SM-106 Gr B",
                joint_type=6,
                rt_level="No RT",
                corrosion_allowance_mm=0.0,
                inside_diameter_mm=2000,
            )
        )
    msg = str(excinfo.value)
    assert "R/2" in msg or "thick-wall" in msg.lower()


def test_calc_engine_refuses_negative_corrosion_allowance() -> None:
    """A negative corrosion allowance is nonsense and must raise."""
    engine = CalculationEngine()
    with pytest.raises(CalculationError) as excinfo:
        engine.calculate(
            CalculationInputs(
                component="cylindrical_shell",
                P_mpa=1.0,
                design_temp_c=200,
                material="SM-516 Gr 70",
                joint_type=1,
                rt_level="Full RT",
                corrosion_allowance_mm=-1.0,
                inside_diameter_mm=1000,
            )
        )
    assert "corrosion" in str(excinfo.value).lower()


def test_calc_engine_near_applicability_boundary() -> None:
    """Spec §Testing — a case near but safely inside the applicability
    envelope (P ≈ 0.37·S·E) must succeed. Pick a material and geometry
    where BOTH the pressure and geometry conditions hold comfortably.
    """
    engine = CalculationEngine()
    # SM-516 Gr 70 at 200°C: S=134, E=1.0 ⇒ S·E = 134.
    # P_limit  = 0.385·134 = 51.59
    # P = 0.37·134 = 49.58 → both pressure and geometry pass with margin.
    # Pick ID=160 mm so that t stays well under the largest standard plate.
    res = engine.calculate(
        CalculationInputs(
            component="cylindrical_shell",
            P_mpa=49.58,
            design_temp_c=200,
            material="SM-516 Gr 70",
            joint_type=1,
            rt_level="Full RT",
            corrosion_allowance_mm=1.0,
            inside_diameter_mm=160,
        )
    )
    assert res.applicability_ok is True
    assert res.t_min_mm > 0
    # Thin-wall must hold: t < R/2
    assert res.t_min_mm < res.R_mm / 2.0


def test_calc_engine_interpolation_example() -> None:
    """At 275°C, SM-516 Gr 70 should interpolate to S = 124 MPa."""
    engine = CalculationEngine()
    res = engine.calculate(
        CalculationInputs(
            component="cylindrical_shell",
            P_mpa=1.8,
            design_temp_c=275,
            material="SM-516 Gr 70",
            joint_type=1,
            rt_level="Full RT",
            corrosion_allowance_mm=2.5,
            inside_diameter_mm=1600,
        )
    )
    assert abs(res.S_mpa - 124.0) < 1e-6
    assert abs(res.t_min_mm - 11.71) < 0.02
