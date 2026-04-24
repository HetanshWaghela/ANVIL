"""Spec §Phase 3 — parametric retrieval test over the design examples.

For each of the 10 hand-verified design examples in
`data/synthetic/design_examples.json`, query the retrieval pipeline with
the example's description and verify that every required piece of the
evidence chain surfaces in the top-10 results:

  * the formula paragraph (`A-27`),
  * the allowable-stress table (`M-1`),
  * the joint-efficiency table (`B-12`).

The spec treats this as the critical retrieval-completeness test — if
any required element is missing we cannot cite the calculation honestly.
"""

from __future__ import annotations

import pytest

from anvil.schemas.retrieval import RetrievalQuery


def _covers(chunks, ref_candidates: tuple[str, ...]) -> bool:
    """Same boundary rule as `refusal_gate._chunk_covers` — A-27(c)(1)
    covers A-27 but A-2 does not."""
    for c in chunks:
        ref = (c.paragraph_ref or "").upper().replace(" ", "")
        if ref.startswith("TABLE"):
            ref = ref[5:]
        for cand in ref_candidates:
            cand_n = cand.upper().replace(" ", "")
            if cand_n.startswith("TABLE"):
                cand_n = cand_n[5:]
            if ref == cand_n:
                return True
            longer, shorter = (ref, cand_n) if len(ref) > len(cand_n) else (cand_n, ref)
            if longer.startswith(shorter) and longer[len(shorter) : len(shorter) + 1] in ("", "("):
                return True
    return False


@pytest.mark.parametrize(
    "required_label,refs",
    [
        ("formula A-27", ("A-27",)),
        ("allowable stress M-1", ("M-1", "Table M-1")),
        ("joint efficiency B-12", ("B-12", "Table B-12")),
    ],
)
def test_retrieval_surfaces_required_elements_for_every_design_example(
    pipeline, design_examples: list[dict], required_label: str, refs: tuple[str, ...]
) -> None:
    """For each design example, the top-10 must contain each required
    paragraph. Parametrized over (example × required-element) so failures
    point at exactly which example missed which evidence.
    """
    missing: list[str] = []
    for ex in design_examples:
        q = RetrievalQuery(
            text=ex["description"] if "description" in ex else ex["query"],
            top_k=10,
            enable_graph_expansion=True,
        )
        chunks = pipeline.retriever.retrieve(q)
        if not _covers(chunks, refs):
            missing.append(ex["id"])
    assert not missing, (
        f"Retrieval failed to surface {required_label} for examples: "
        f"{missing}. This is a retrieval-completeness regression — the "
        f"graph-expansion step should pull the required table into the "
        f"top-10 whenever A-27 is retrieved."
    )
