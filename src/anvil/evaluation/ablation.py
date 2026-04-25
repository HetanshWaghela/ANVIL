"""Pipeline ablation configuration.

Each ablation is a single Pydantic-validated config object that
`build_pipeline(ablation=...)` honors. Keeping this in one place means:

  * the run-logger's manifest captures `ablation_config` exactly as
    constructed — no scattered flags to track down,
  * the 7 named configs from the plan §4.1 are constants, so adding an
    8th is a deliberate edit (not a flag tweak),
  * tests can compare the `to_label()` strings to lock the ablation
    catalog the same way `NIM_MODELS` is locked.

The seven ablations are NOT all equally useful — some are obvious
straw-men (e.g. disabling the refusal gate to measure false-confident
rate) — but each one exists to defend a specific ADR with a number
rather than a claim. See `docs/ablations.md` for the full study.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal[
    "hybrid",            # baseline: BM25 + vector + graph + RRF
    "bm25_only",         # lexical only — degrades on synonym queries
    "vector_only",       # semantic only — degrades on exact-ref lookups
    "hybrid_no_graph",   # fusion without graph expansion
]


class PipelineAblation(BaseModel):
    """One named ablation config.

    Defaults are the production-baseline pipeline; flipping any flag
    moves to a non-baseline config that the plan defends with data.
    """

    name: str = Field(
        description=(
            "Slug used as the directory name for the ablation's run "
            "(`abl-<name>` in the run-id). Must be filesystem-safe."
        ),
    )
    retrieval_mode: RetrievalMode = "hybrid"
    use_pinned_data: bool = Field(
        default=True,
        description=(
            "When False, the calculation engine is bypassed entirely "
            "and the LLM is left to extract numerical values from "
            "retrieved chunks. This is the highest-risk failure mode "
            "we exist to prevent — measuring it is the point."
        ),
    )
    use_refusal_gate: bool = Field(
        default=True,
        description=(
            "When False, the pre-LLM refusal gate is skipped. OOD "
            "queries reach the LLM. Used to measure the false-confident "
            "rate the gate prevents."
        ),
    )
    use_citation_enforcer: bool = Field(
        default=True,
        description=(
            "When False, post-generation citation validation is "
            "skipped. Used to measure the fraction of generated quotes "
            "that fail validation in the wild."
        ),
    )

    def to_summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "retrieval_mode": self.retrieval_mode,
            "use_pinned_data": self.use_pinned_data,
            "use_refusal_gate": self.use_refusal_gate,
            "use_citation_enforcer": self.use_citation_enforcer,
        }


# Locked ablation catalog. Each name is a stable slug recorded in the
# run manifest so `data/runs/<*_abl-<name>>/` directories are
# self-describing. Adding a new ablation is a deliberate decision (an
# ADR + a test inventory entry).
ABLATIONS: dict[str, PipelineAblation] = {
    "baseline": PipelineAblation(name="baseline"),
    "bm25-only": PipelineAblation(name="bm25-only", retrieval_mode="bm25_only"),
    "vector-only": PipelineAblation(name="vector-only", retrieval_mode="vector_only"),
    "no-graph": PipelineAblation(name="no-graph", retrieval_mode="hybrid_no_graph"),
    "no-pinned": PipelineAblation(name="no-pinned", use_pinned_data=False),
    "no-refusal": PipelineAblation(name="no-refusal", use_refusal_gate=False),
    "no-citation-enforcer": PipelineAblation(
        name="no-citation-enforcer", use_citation_enforcer=False
    ),
}


def get_ablation(name: str) -> PipelineAblation:
    """Resolve an ablation slug → config. Raises if unknown."""
    if name not in ABLATIONS:
        known = sorted(ABLATIONS)
        raise ValueError(
            f"Unknown ablation {name!r}. Known: {known}. "
            f"Add the new ablation to ABLATIONS in src/anvil/evaluation/ablation.py."
        )
    return ABLATIONS[name]
