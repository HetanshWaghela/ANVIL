"""Run the full evaluation suite against the golden dataset.

Produces a stamped run directory under `data/runs/<run_id>/` per the
plan §3.1 layout (manifest.json, summary.json, report.md,
per_example.json — all committed). Default backend is `fake` (no
network); pass `--backend nvidia_nim --model meta/llama-3.3-70b-instruct`
for a real NIM run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.evaluation.manifest import make_run_id  # noqa: E402
from anvil.evaluation.run_logger import (  # noqa: E402
    RunLogger,
    RunLoggerConfig,
)
from anvil.evaluation.runner import EvaluationRunner  # noqa: E402
from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

log = get_logger("anvil.evaluate")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ANVIL golden-dataset evaluation."
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("ANVIL_LLM_BACKEND", "fake"),
        choices=["fake", "nvidia_nim", "openai_compatible", "instructor"],
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override ANVIL_LLM_MODEL.",
    )
    from anvil.evaluation.ablation import ABLATIONS

    p.add_argument(
        "--ablation",
        default="baseline",
        choices=sorted(ABLATIONS),
        help="Ablation slug (see src/anvil/evaluation/ablation.py).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/runs"),
        help="Where to drop the run directory (default: data/runs).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests" / "evaluation" / "golden_dataset.json",
        help="Golden dataset JSON.",
    )
    p.add_argument(
        "--standard",
        type=Path,
        default=ROOT / "data" / "synthetic" / "standard.md",
        help="Markdown standard to parse for retrieval and citations.",
    )
    p.add_argument(
        "--dataset-version",
        default="goldenv1",
        help="Run-id dataset slug, e.g. goldenv1 or asme-private-v1.",
    )
    p.add_argument(
        "--legacy-summary",
        type=Path,
        default=Path("data/indexes/evaluation.json"),
        help=(
            "Also write a flat summary at this path for backward "
            "compatibility with CI artifact uploaders."
        ),
    )
    return p.parse_args()


async def _run() -> int:
    args = _parse_args()

    # Wire env so build_pipeline picks up the right backend.
    os.environ["ANVIL_LLM_BACKEND"] = args.backend
    model = args.model or (
        os.environ.get("ANVIL_LLM_MODEL") if args.backend != "fake" else None
    )
    if model:
        os.environ["ANVIL_LLM_MODEL"] = model
    else:
        os.environ.pop("ANVIL_LLM_MODEL", None)

    pipeline = build_pipeline(standard_path=args.standard, ablation=args.ablation)
    effective_model = getattr(pipeline.generator.backend, "model", args.model)
    runner = EvaluationRunner(pipeline.generator)
    dataset_path = args.dataset
    examples = load_golden_dataset(dataset_path)

    run_id = make_run_id(
        backend=args.backend,
        model=effective_model,
        ablation=args.ablation,
        dataset_version=args.dataset_version,
    )
    from anvil.evaluation.ablation import ABLATIONS as _ABL

    cfg = RunLoggerConfig(
        run_id=run_id,
        backend=args.backend,
        model=effective_model,
        ablation=args.ablation,
        ablation_config=_ABL[args.ablation].to_summary(),
        dataset_path=dataset_path,
        output_root=args.output_root,
    )
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        summary = await runner.run(examples)
        rl.write_summary(summary)
        # Per-example raw responses (gitignored). Use the runner's cached
        # outcomes; re-invoking generation would double real-LLM calls.
        for ex, outcome in zip(examples, summary.outcomes, strict=True):
            rl.record_example(
                ex.id,
                outcome.response,
                retrieved=outcome.retrieved_chunks,
            )

    # Legacy flat-file summary so existing CI artifact uploaders keep
    # working without churn.
    legacy = args.legacy_summary
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        json.dumps(
            {"aggregate": summary.aggregate, "pass_rate": summary.pass_rate},
            indent=2,
        )
    )

    print("\n=== Evaluation summary ===")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "backend": args.backend,
                "model": effective_model,
                "ablation": args.ablation,
                "dataset_version": args.dataset_version,
                "pass_rate": summary.pass_rate,
                "aggregate": summary.aggregate,
                "run_dir": str(args.output_root / run_id),
            },
            indent=2,
        )
    )
    print(f"\nRun artifacts: {args.output_root / run_id}")
    return 0 if summary.pass_rate >= 0.7 else 1


def main() -> None:
    rc = asyncio.run(_run())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
