"""Pinned material property records for SPES-1.

The canonical source is `data/synthetic/table_1a_materials.json`. This module
loads the same values into typed Python structures so calculations can access
them without parsing JSON on every call.

Hallucinated material properties are the highest-risk failure mode, so the
generation pipeline consults this module BEFORE RAG retrieval for material
lookups.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

_MATERIALS_JSON = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "synthetic"
    / "table_1a_materials.json"
)


class MaterialRecord(BaseModel):
    """A single row from Table M-1 (Material Properties)."""

    line_no: int
    spec_no: str
    grade: str
    nominal_composition: str
    product_form: str
    p_no: int
    group_no: int
    min_tensile_mpa: float
    min_yield_mpa: float
    stress_by_temp_c: dict[int, float] = Field(
        description="Allowable stress (MPa) keyed by tabulated temperature (°C)"
    )
    max_temp_c: int
    notes: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Canonical lookup key: 'SM-516 Gr 70', 'SM-105', etc."""
        return f"{self.spec_no} {self.grade}".strip()


def _load_materials() -> dict[str, MaterialRecord]:
    raw = json.loads(_MATERIALS_JSON.read_text())
    out: dict[str, MaterialRecord] = {}
    for entry in raw:
        rec = MaterialRecord(
            line_no=entry["line_no"],
            spec_no=entry["spec_no"],
            grade=entry.get("grade", ""),
            nominal_composition=entry["nominal_composition"],
            product_form=entry["product_form"],
            p_no=entry["p_no"],
            group_no=entry["group_no"],
            min_tensile_mpa=entry["min_tensile_mpa"],
            min_yield_mpa=entry["min_yield_mpa"],
            stress_by_temp_c={int(k): float(v) for k, v in entry["stress_values_mpa"].items()},
            max_temp_c=entry["max_temp_c"],
            notes=entry.get("notes", []),
        )
        out[rec.key] = rec
    return out


MATERIALS: dict[str, MaterialRecord] = _load_materials()


def list_materials() -> list[str]:
    """List all material lookup keys present in pinned data."""
    return sorted(MATERIALS.keys())


def get_material(spec_grade: str) -> MaterialRecord | None:
    """Look up a material record by its canonical key (e.g. 'SM-516 Gr 70')."""
    return MATERIALS.get(spec_grade.strip())


def interpolate_stress(
    stress_by_temp: dict[int, float], target_temp: float
) -> float | None:
    """Linear interpolation between bracketing tabulated temperatures.

    Returns None if target_temp is outside the tabulated range (no extrapolation).
    Uses Decimal arithmetic internally for reproducible results.
    """
    if not stress_by_temp:
        return None
    temps = sorted(stress_by_temp.keys())
    if target_temp < temps[0] or target_temp > temps[-1]:
        return None

    # Exact tabulated value
    if int(target_temp) == target_temp and int(target_temp) in stress_by_temp:
        return float(stress_by_temp[int(target_temp)])

    # Bracketing temperatures
    lower = max(t for t in temps if t <= target_temp)
    upper = min(t for t in temps if t > target_temp)
    if lower == upper:
        return float(stress_by_temp[lower])

    s_lower = Decimal(str(stress_by_temp[lower]))
    s_upper = Decimal(str(stress_by_temp[upper]))
    fraction = Decimal(str(target_temp - lower)) / Decimal(str(upper - lower))
    return float(s_lower + fraction * (s_upper - s_lower))


def get_allowable_stress(spec_grade: str, temp_c: float) -> float | None:
    """Look up allowable stress for the given material at the given temperature.

    Returns None if:
    - material is not in pinned data, or
    - temperature is outside the tabulated range (no extrapolation), or
    - temperature exceeds the material's max_temp_c (per M-23).
    """
    mat = get_material(spec_grade)
    if mat is None:
        return None
    if temp_c > mat.max_temp_c:
        return None
    return interpolate_stress(mat.stress_by_temp_c, temp_c)
