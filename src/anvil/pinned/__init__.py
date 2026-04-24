"""Pinned ground-truth data: verified lookups that the generation layer
prefers over RAG retrieval.

Hallucinating a material stress value is catastrophic. These modules hold
verified values extracted from the synthetic standard. The RAG pipeline may
retrieve the same data for citation, but numerical values used in
calculations are ALWAYS sourced from here (or refused).
"""

from __future__ import annotations

from anvil.pinned.formulas import (
    STANDARD_PLATES_MM,
    cylindrical_shell_thickness_inside_radius,
    cylindrical_shell_thickness_outside_radius,
    mawp_cylindrical_inside_radius,
    mawp_cylindrical_outside_radius,
    mawp_spherical,
    next_standard_plate,
    spherical_shell_thickness,
)
from anvil.pinned.joint_efficiencies import (
    JOINT_EFFICIENCIES,
    RT_LEVELS,
    get_joint_efficiency,
)
from anvil.pinned.materials import (
    MATERIALS,
    MaterialRecord,
    get_allowable_stress,
    get_material,
    interpolate_stress,
    list_materials,
)

__all__ = [
    "JOINT_EFFICIENCIES",
    "MATERIALS",
    "MaterialRecord",
    "RT_LEVELS",
    "STANDARD_PLATES_MM",
    "cylindrical_shell_thickness_inside_radius",
    "cylindrical_shell_thickness_outside_radius",
    "get_allowable_stress",
    "get_joint_efficiency",
    "get_material",
    "interpolate_stress",
    "list_materials",
    "mawp_cylindrical_inside_radius",
    "mawp_cylindrical_outside_radius",
    "mawp_spherical",
    "next_standard_plate",
    "spherical_shell_thickness",
]
