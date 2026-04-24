"""Joint efficiency lookup table (B-12) as code."""

from __future__ import annotations

import json
from pathlib import Path

_JOINT_JSON = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "synthetic"
    / "joint_efficiency_table.json"
)

RT_LEVELS: tuple[str, ...] = ("Full RT", "Spot RT", "No RT")


def _load_joint_efficiencies() -> dict[int, dict[str, float]]:
    raw = json.loads(_JOINT_JSON.read_text())
    out: dict[int, dict[str, float]] = {}
    for joint in raw["joints"]:
        out[int(joint["type"])] = {
            level: float(joint["efficiencies"][level]) for level in RT_LEVELS
        }
    return out


JOINT_EFFICIENCIES: dict[int, dict[str, float]] = _load_joint_efficiencies()


def _normalize_rt(rt_level: str) -> str | None:
    s = rt_level.strip().lower().replace("-", " ").replace("_", " ")
    if "full" in s:
        return "Full RT"
    if "spot" in s:
        return "Spot RT"
    if "no" in s or "none" in s:
        return "No RT"
    return None


def get_joint_efficiency(joint_type: int, rt_level: str) -> float | None:
    """Look up joint efficiency E for (joint_type, rt_level). None if invalid."""
    if joint_type not in JOINT_EFFICIENCIES:
        return None
    normalized = _normalize_rt(rt_level)
    if normalized is None:
        return None
    return JOINT_EFFICIENCIES[joint_type].get(normalized)
