"""Deterministic SPES-1 formulas implemented with decimal arithmetic.

All calculation results in anvil pass through these functions. Formulas are
mirrored from the ASME BPVC (Section VIII Div. 1, UG-27) but referenced by
the synthetic paragraph numbers used in SPES-1 (A-27(c)(1), A-27(c)(2), A-27(d)).

The LLM never performs arithmetic. The LLM orchestrates — selects the formula,
identifies the inputs — and these functions compute the result.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

from anvil import CalculationError

# 28 digits is more than enough for engineering precision — we only report to
# 3 decimal places. Using Decimal eliminates floating-point drift.
getcontext().prec = 28


STANDARD_PLATES_MM: tuple[int, ...] = (
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32, 36, 40, 45, 50,
)


def _D(x: float | int | str | Decimal) -> Decimal:
    """Coerce any numeric to Decimal via string to avoid binary-float errors."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def check_cylindrical_applicability(P: float, S: float, E: float) -> tuple[float, float, bool]:
    """Return (P, 0.385·S·E, P ≤ 0.385·S·E) for A-27(c)(1) / A-27(c)(2)."""
    lhs = _D(P)
    rhs = _D("0.385") * _D(S) * _D(E)
    return float(lhs), float(rhs), lhs <= rhs


def check_spherical_applicability(P: float, S: float, E: float) -> tuple[float, float, bool]:
    """Return (P, 0.665·S·E, P ≤ 0.665·S·E) for A-27(d)."""
    lhs = _D(P)
    rhs = _D("0.665") * _D(S) * _D(E)
    return float(lhs), float(rhs), lhs <= rhs


def cylindrical_shell_thickness_inside_radius(
    P: float, R: float, S: float, E: float
) -> float:
    """A-27(c)(1): t = (P × R) / (S × E − 0.6 × P).

    Args:
        P: design pressure, MPa
        R: inside radius, mm
        S: allowable stress, MPa
        E: joint efficiency, dimensionless

    Returns:
        Minimum required thickness t (mm), rounded to 2 decimal places.

    Raises:
        CalculationError: if the denominator is ≤ 0 (degenerate case).
    """
    Pd, Rd, Sd, Ed = _D(P), _D(R), _D(S), _D(E)
    denom = Sd * Ed - _D("0.6") * Pd
    if denom <= 0:
        raise CalculationError(
            f"A-27(c)(1) degenerate: S·E − 0.6·P = {denom} ≤ 0 "
            f"(P={P}, S={S}, E={E})"
        )
    t = (Pd * Rd) / denom
    return float(_round_half_up(t, 2))


def cylindrical_shell_thickness_outside_radius(
    P: float, Ro: float, S: float, E: float
) -> float:
    """A-27(c)(2): t = (P × Ro) / (S × E + 0.4 × P)."""
    Pd, Rd, Sd, Ed = _D(P), _D(Ro), _D(S), _D(E)
    denom = Sd * Ed + _D("0.4") * Pd
    if denom <= 0:
        raise CalculationError(
            f"A-27(c)(2) degenerate: S·E + 0.4·P = {denom} ≤ 0"
        )
    t = (Pd * Rd) / denom
    return float(_round_half_up(t, 2))


def spherical_shell_thickness(P: float, R: float, S: float, E: float) -> float:
    """A-27(d): t = (P × R) / (2 × S × E − 0.2 × P)."""
    Pd, Rd, Sd, Ed = _D(P), _D(R), _D(S), _D(E)
    denom = _D(2) * Sd * Ed - _D("0.2") * Pd
    if denom <= 0:
        raise CalculationError(
            f"A-27(d) degenerate: 2·S·E − 0.2·P = {denom} ≤ 0"
        )
    t = (Pd * Rd) / denom
    return float(_round_half_up(t, 2))


def mawp_cylindrical_inside_radius(
    S: float, E: float, t_corroded: float, R: float
) -> float:
    """Back-calculate MAWP for a cylindrical shell (inside radius).

    P = (S × E × t_corr) / (R + 0.6 × t_corr)
    """
    Sd, Ed, td, Rd = _D(S), _D(E), _D(t_corroded), _D(R)
    denom = Rd + _D("0.6") * td
    if denom <= 0:
        raise CalculationError("MAWP denominator non-positive")
    P = (Sd * Ed * td) / denom
    return float(_round_half_up(P, 3))


def mawp_cylindrical_outside_radius(
    S: float, E: float, t_corroded: float, Ro: float
) -> float:
    """MAWP for A-27(c)(2): P = (S × E × t_corr) / (Ro − 0.4 × t_corr)."""
    Sd, Ed, td, Rd = _D(S), _D(E), _D(t_corroded), _D(Ro)
    denom = Rd - _D("0.4") * td
    if denom <= 0:
        raise CalculationError("MAWP denominator non-positive")
    P = (Sd * Ed * td) / denom
    return float(_round_half_up(P, 3))


def mawp_spherical(S: float, E: float, t_corroded: float, R: float) -> float:
    """MAWP for A-27(d): P = (2 × S × E × t_corr) / (R + 0.2 × t_corr)."""
    Sd, Ed, td, Rd = _D(S), _D(E), _D(t_corroded), _D(R)
    denom = Rd + _D("0.2") * td
    if denom <= 0:
        raise CalculationError("MAWP denominator non-positive")
    P = (_D(2) * Sd * Ed * td) / denom
    return float(_round_half_up(P, 3))


def next_standard_plate(t_design_mm: float) -> int:
    """Select the smallest standard plate thickness ≥ t_design.

    Raises:
        CalculationError: if t_design exceeds the largest standard plate.
    """
    if t_design_mm < 0:
        raise CalculationError("Design thickness cannot be negative")
    for plate in STANDARD_PLATES_MM:
        if plate >= t_design_mm:
            return plate
    raise CalculationError(
        f"Required thickness {t_design_mm} mm exceeds the largest standard plate "
        f"{STANDARD_PLATES_MM[-1]} mm"
    )


def _round_half_up(d: Decimal, places: int) -> Decimal:
    """Round half-up to the given number of decimal places (engineering convention)."""
    from decimal import ROUND_HALF_UP
    quantizer = Decimal(1).scaleb(-places)
    return d.quantize(quantizer, rounding=ROUND_HALF_UP)
