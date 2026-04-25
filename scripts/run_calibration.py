"""M4 — trust-boundary calibration sweep.

Sweeps `RELEVANCE_THRESHOLD` ∈ {0.05, 0.075, 0.10, 0.125, 0.15, 0.175,
0.20, 0.25, 0.30} (plan §5.1). For each threshold, runs the refusal
gate over every golden example and records:

  * `refusal_precision` — of refused examples, fraction whose
    `expected_refusal=True`.
  * `refusal_recall` — of `expected_refusal=True` examples, fraction
    that were refused.
  * `non_refusal_coverage` — of `expected_refusal=False` examples,
    fraction that were NOT refused (i.e. preserved for downstream
    generation).

The output lands in `data/calibration/calibration.json` and
`data/calibration/calibration.md` (committed). The fully-fleshed
narrative + chosen operating point lives in `docs/trust_calibration.md`.

Note: this script does NOT call the LLM. The refusal gate runs
deterministically against retrieved-chunk scores, so the sweep is
backend-agnostic and free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.generation.refusal_gate import should_refuse  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402
from anvil.schemas.retrieval import RetrievalQuery  # noqa: E402

DEFAULT_THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_THRESHOLDS,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Where to drop calibration.json + calibration.md.",
    )
    return p.parse_args()


async def _run() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = ROOT / "tests" / "evaluation" / "golden_dataset.json"
    examples = load_golden_dataset(dataset_path)
    pipeline = build_pipeline()  # baseline retrieval

    # Cache per-example retrieved chunks once — retrieval is deterministic
    # at a given git sha and the gate is the only thing that varies.
    per_example_chunks: dict[str, list] = {}
    per_example_max_score: dict[str, float] = {}
    for ex in examples:
        chunks = pipeline.retriever.retrieve(
            RetrievalQuery(text=ex.query, top_k=10, enable_graph_expansion=True)
        )
        per_example_chunks[ex.id] = chunks
        per_example_max_score[ex.id] = max((c.score for c in chunks), default=0.0)

    results: list[dict] = []
    for t in args.thresholds:
        tp = fp = tn = fn = 0
        per_ex: list[dict] = []
        for ex in examples:
            chunks = per_example_chunks[ex.id]
            decision = should_refuse(ex.query, chunks, relevance_threshold=t)
            refused = decision.should_refuse
            expected = ex.expected_refusal
            if expected and refused:
                tp += 1
            elif expected and not refused:
                fn += 1
            elif (not expected) and refused:
                fp += 1
            else:
                tn += 1
            per_ex.append(
                {
                    "example_id": ex.id,
                    "category": ex.category,
                    "expected_refusal": expected,
                    "refused": refused,
                    "max_relevance": per_example_max_score[ex.id],
                    "reason": decision.reason,
                }
            )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        non_refusal_coverage = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        results.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "refusal_precision": precision,
                "refusal_recall": recall,
                "non_refusal_coverage": non_refusal_coverage,
                "per_example": per_ex,
            }
        )

    json_path = args.out_dir / "calibration.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    # Markdown table
    md_lines = [
        "# Refusal-gate calibration sweep",
        "",
        "| threshold | refusal_precision | refusal_recall | non_refusal_coverage | TP | FP | TN | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        md_lines.append(
            f"| {r['threshold']:.3f} "
            f"| {r['refusal_precision']:.3f} "
            f"| {r['refusal_recall']:.3f} "
            f"| {r['non_refusal_coverage']:.3f} "
            f"| {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )
    md_path = args.out_dir / "calibration.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print("\n=== Calibration sweep summary ===")
    print(json.dumps(
        [
            {
                "threshold": r["threshold"],
                "precision": r["refusal_precision"],
                "recall": r["refusal_recall"],
                "coverage": r["non_refusal_coverage"],
            }
            for r in results
        ],
        indent=2,
    ))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
