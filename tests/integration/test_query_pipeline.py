"""End-to-end query pipeline integration."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_end_to_end_calculation_high_confidence(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "Calculate thickness for cylindrical shell ID=1500 mm, P=2.0 MPa, "
        "T=200°C, SM-516 Gr 70, Type 1 full RT, CA=2.0 mm.",
        top_k=10,
    )
    assert outcome.response.confidence.value == "high"
    assert outcome.calculation is not None
    assert abs(outcome.calculation.t_min_mm - 11.30) < 0.02
    assert outcome.citation_validation.accuracy >= 0.8


@pytest.mark.asyncio
async def test_end_to_end_lookup(pipeline) -> None:
    outcome = await pipeline.generator.generate(
        "What is the allowable stress for SM-516 Gr 70 at 300°C?",
        top_k=5,
    )
    # Lookup queries are not refused — they should produce a cited answer.
    assert outcome.response.confidence.value != "insufficient"
    refs = {c.paragraph_ref for c in outcome.response.citations}
    assert any("M-1" in r or "M" in r for r in refs)
