"""Phase 0 tests: pinned data ground-truth integrity and formula correctness."""

from __future__ import annotations

import pytest

from anvil.pinned import (
    JOINT_EFFICIENCIES,
    MATERIALS,
    RT_LEVELS,
    cylindrical_shell_thickness_inside_radius,
    cylindrical_shell_thickness_outside_radius,
    get_allowable_stress,
    get_joint_efficiency,
    interpolate_stress,
    mawp_cylindrical_inside_radius,
    mawp_cylindrical_outside_radius,
    mawp_spherical,
    next_standard_plate,
    spherical_shell_thickness,
)

# ---- Materials -------------------------------------------------------------


def test_materials_loaded_non_empty() -> None:
    assert len(MATERIALS) >= 4
    assert "SM-516 Gr 70" in MATERIALS


def test_stress_monotonically_non_increasing_with_temperature() -> None:
    """Allowable stress must never increase as temperature rises."""
    for key, mat in MATERIALS.items():
        temps = sorted(mat.stress_by_temp_c.keys())
        for i in range(len(temps) - 1):
            s_lo = mat.stress_by_temp_c[temps[i]]
            s_hi = mat.stress_by_temp_c[temps[i + 1]]
            assert s_lo >= s_hi, (
                f"{key}: stress at {temps[i]}°C ({s_lo}) < stress at "
                f"{temps[i+1]}°C ({s_hi})"
            )


def test_stress_values_positive() -> None:
    for key, mat in MATERIALS.items():
        for t, s in mat.stress_by_temp_c.items():
            assert s > 0, f"{key} at {t}°C: non-positive stress {s}"


def test_max_temp_c_consistent_with_tabulated() -> None:
    for mat in MATERIALS.values():
        tabulated_max = max(mat.stress_by_temp_c.keys())
        assert mat.max_temp_c >= tabulated_max


def test_get_allowable_stress_out_of_range() -> None:
    # SM-516 Gr 70 has max_temp_c = 500
    assert get_allowable_stress("SM-516 Gr 70", 700) is None
    # Below tabulated range
    assert get_allowable_stress("SM-516 Gr 70", 0) is None
    # Unknown material
    assert get_allowable_stress("SM-999 Gr XYZ", 100) is None


def test_get_allowable_stress_exact_tabulated() -> None:
    assert get_allowable_stress("SM-516 Gr 70", 350) == 114.0
    assert get_allowable_stress("SM-516 Gr 70", 200) == 134.0


def test_interpolation_known_midpoint() -> None:
    # Midpoint between 250 (127) and 300 (121) should be 124
    s = interpolate_stress({250: 127, 300: 121}, 275)
    assert s is not None
    assert abs(s - 124.0) < 1e-6


def test_interpolation_rejects_extrapolation() -> None:
    assert interpolate_stress({100: 10, 200: 5}, 50) is None
    assert interpolate_stress({100: 10, 200: 5}, 250) is None


# ---- Joint efficiencies ----------------------------------------------------


def test_joint_efficiencies_in_range() -> None:
    for jtype, efficiencies in JOINT_EFFICIENCIES.items():
        for level, value in efficiencies.items():
            assert level in RT_LEVELS
            assert 0.0 < value <= 1.0, (
                f"Type {jtype} / {level}: E={value} out of (0, 1]"
            )


def test_joint_efficiencies_decrease_with_weaker_exam() -> None:
    """For each joint type, Full RT >= Spot RT >= No RT."""
    for jtype, eff in JOINT_EFFICIENCIES.items():
        assert eff["Full RT"] >= eff["Spot RT"] >= eff["No RT"], (
            f"Type {jtype}: monotonicity broken {eff}"
        )


def test_joint_efficiency_lookup_normalizes_rt_text() -> None:
    assert get_joint_efficiency(1, "full rt") == 1.0
    assert get_joint_efficiency(1, "Full-RT") == 1.0
    assert get_joint_efficiency(1, "FULL RADIOGRAPHY") is not None


def test_joint_efficiency_invalid_inputs() -> None:
    assert get_joint_efficiency(99, "Full RT") is None
    assert get_joint_efficiency(1, "invalid level") is None


# ---- Formulas --------------------------------------------------------------


def test_ug27_c1_known_example() -> None:
    t = cylindrical_shell_thickness_inside_radius(P=1.5, R=900, S=114, E=1.0)
    assert abs(t - 11.94) < 0.005


def test_ug27_c2_known_example() -> None:
    t = cylindrical_shell_thickness_outside_radius(P=1.0, Ro=500, S=134, E=0.9)
    assert abs(t - 4.13) < 0.005


def test_ug27_d_known_example() -> None:
    t = spherical_shell_thickness(P=2.5, R=1000, S=124, E=1.0)
    assert abs(t - 10.10) < 0.005


def test_mawp_cylindrical_inside_radius() -> None:
    # From design example 1: t_corr=13, S=114, E=1, R=900 → 1.633
    P = mawp_cylindrical_inside_radius(S=114, E=1.0, t_corroded=13, R=900)
    assert abs(P - 1.633) < 0.002


def test_mawp_spherical() -> None:
    # Example 4: t_corr=11, S=124, E=1, R=1000 → 2.722
    P = mawp_spherical(S=124, E=1.0, t_corroded=11, R=1000)
    assert abs(P - 2.722) < 0.002


def test_mawp_cylindrical_outside_radius() -> None:
    # Example 8: t_corr=4.5, S=134, E=0.9, Ro=500 → 1.089
    P = mawp_cylindrical_outside_radius(S=134, E=0.9, t_corroded=4.5, Ro=500)
    assert abs(P - 1.089) < 0.002


@pytest.mark.parametrize(
    "t_design, expected",
    [(0.1, 6), (5.9, 6), (6.0, 6), (6.01, 8), (11.94, 12), (14.94, 16), (50.0, 50)],
)
def test_next_standard_plate(t_design: float, expected: int) -> None:
    assert next_standard_plate(t_design) == expected


def test_next_standard_plate_too_large() -> None:
    from anvil import CalculationError

    with pytest.raises(CalculationError):
        next_standard_plate(60.0)


def test_formula_degenerate_case_raises() -> None:
    from anvil import CalculationError

    # S*E - 0.6P would be negative → degenerate
    with pytest.raises(CalculationError):
        cylindrical_shell_thickness_inside_radius(P=300, R=100, S=1, E=1)


def test_all_design_examples_reproduce_ground_truth(design_examples: list[dict]) -> None:
    """Every design example's expected_result must be reproducible from pinned formulas."""
    for ex in design_examples:
        inp = ex["inputs"]
        exp = ex["expected_result"]
        lookups = ex["expected_lookups"]
        S = lookups["S_mpa"]
        E = lookups["E"]
        ca = inp["corrosion_allowance_mm"]

        if inp["component"] == "spherical_shell":
            R = lookups["R_mm"]
            t = spherical_shell_thickness(P=inp["P_mpa"], R=R, S=S, E=E)
        elif inp["component"] == "cylindrical_shell_outside_radius":
            Ro = lookups["Ro_mm"]
            t = cylindrical_shell_thickness_outside_radius(P=inp["P_mpa"], Ro=Ro, S=S, E=E)
        else:
            R = lookups["R_mm"]
            t = cylindrical_shell_thickness_inside_radius(P=inp["P_mpa"], R=R, S=S, E=E)

        # Spec §Faithfulness says 0.1% tolerance. The engine uses Decimal
        # arithmetic with ROUND_HALF_UP and reproduces each example
        # exactly, so a tight 0.005 mm absolute bound is easily met and
        # would fail loudly if anyone broke the Decimal math path.
        assert abs(t - exp["t_min_mm"]) < 0.005, (
            f"{ex['id']}: t_min {t} != {exp['t_min_mm']}"
        )
        t_design = round(t + ca, 2)
        assert abs(t_design - exp["t_design_mm"]) < 0.005

        plate = next_standard_plate(t_design)
        assert plate == exp["t_nominal_mm"], f"{ex['id']}: plate mismatch"

        t_corroded = plate - ca
        if inp["component"] == "spherical_shell":
            mawp = mawp_spherical(S=S, E=E, t_corroded=t_corroded, R=R)
        elif inp["component"] == "cylindrical_shell_outside_radius":
            mawp = mawp_cylindrical_outside_radius(S=S, E=E, t_corroded=t_corroded, Ro=Ro)
        else:
            mawp = mawp_cylindrical_inside_radius(S=S, E=E, t_corroded=t_corroded, R=R)
        assert abs(mawp - exp["mawp_mpa"]) < 0.002, (
            f"{ex['id']}: MAWP {mawp} != {exp['mawp_mpa']}"
        )
