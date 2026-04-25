"""M4 — trust-boundary calibration lock + smoke test.

`docs/trust_calibration.md` chooses `RELEVANCE_THRESHOLD = 0.05` after a
defended sweep. These tests guard that operating point against silent
edits and exercise the sweep script's sensitivity (precision/recall
must move in the documented direction as the threshold rises).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.evaluation.dataset import load_golden_dataset
from anvil.generation.refusal_gate import RELEVANCE_THRESHOLD, should_refuse
from anvil.pipeline import build_pipeline
from anvil.schemas.retrieval import RetrievalQuery

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "tests" / "evaluation" / "golden_dataset.json"


def test_relevance_threshold_locked_at_documented_value() -> None:
    """REGRESSION (plan M4 acceptance): the operating point chosen in
    `docs/trust_calibration.md` is 0.05. Changing this constant requires
    a coordinated edit of the doc, the ADR, and this lock — see the
    "Operating-point in code" section of trust_calibration.md.
    """
    assert RELEVANCE_THRESHOLD == 0.05, (
        "RELEVANCE_THRESHOLD changed without updating docs/trust_calibration.md "
        "and ADR-005. See docs/trust_calibration.md §'Operating-point in code'."
    )


@pytest.mark.asyncio
async def test_calibration_recall_holds_at_chosen_threshold() -> None:
    """At the chosen operating point the gate must catch every
    `expected_refusal=True` example in the golden dataset."""
    pipe = build_pipeline()
    examples = load_golden_dataset(DATASET_PATH)
    ood = [e for e in examples if e.expected_refusal]
    assert ood, "no OOD examples in dataset — calibration meaningless"
    misses: list[str] = []
    for ex in ood:
        chunks = pipe.retriever.retrieve(
            RetrievalQuery(text=ex.query, top_k=10, enable_graph_expansion=True)
        )
        decision = should_refuse(
            ex.query, chunks, relevance_threshold=RELEVANCE_THRESHOLD
        )
        if not decision.should_refuse:
            misses.append(ex.id)
    assert not misses, (
        f"Refusal recall < 1.0 at the documented operating point. Missed: {misses}"
    )


@pytest.mark.asyncio
async def test_calibration_precision_holds_at_chosen_threshold() -> None:
    """At the chosen operating point the gate must NOT refuse any
    in-domain example. (At higher thresholds in-domain queries get
    falsely refused — that's why we don't pick those.)"""
    pipe = build_pipeline()
    examples = load_golden_dataset(DATASET_PATH)
    in_domain = [e for e in examples if not e.expected_refusal]
    false_refusals: list[str] = []
    for ex in in_domain:
        chunks = pipe.retriever.retrieve(
            RetrievalQuery(text=ex.query, top_k=10, enable_graph_expansion=True)
        )
        decision = should_refuse(
            ex.query, chunks, relevance_threshold=RELEVANCE_THRESHOLD
        )
        if decision.should_refuse:
            false_refusals.append(ex.id)
    assert not false_refusals, (
        f"Refusal precision < 1.0 at the documented operating point. "
        f"False-refused: {false_refusals}"
    )


@pytest.mark.asyncio
async def test_calibration_higher_threshold_increases_false_refusals() -> None:
    """Sanity check on the sweep direction: at a much higher threshold
    we MUST see at least one FP. If this stops being true the sweep is
    no longer informative and the trust-calibration doc needs to be
    revisited."""
    pipe = build_pipeline()
    examples = load_golden_dataset(DATASET_PATH)
    in_domain = [e for e in examples if not e.expected_refusal]
    fps_high = 0
    for ex in in_domain:
        chunks = pipe.retriever.retrieve(
            RetrievalQuery(text=ex.query, top_k=10, enable_graph_expansion=True)
        )
        if should_refuse(ex.query, chunks, relevance_threshold=0.30).should_refuse:
            fps_high += 1
    assert fps_high > 0, (
        "At threshold=0.30 we expect at least one in-domain FP; got zero. "
        "Either retrieval got dramatically better or the dataset shifted."
    )
