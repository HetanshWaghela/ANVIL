"""Run the M3 ablation matrix.

Iterates the locked `ABLATIONS` catalog × the chosen backend, producing
one stamped run directory per (backend, ablation) pair under
`data/runs/`. After all runs land, calls `scripts/compare_runs.py`
internally to render `docs/ablations.md`.

Default backend is `fake` (no NIM key, no network) — this is the
configuration CI runs and the one that proves every ablation flag is
threaded correctly. To run against a real NIM model:

    NVIDIA_API_KEY=… uv run python scripts/run_ablations.py \\
        --backend nvidia_nim --model meta/llama-3.3-70b-instruct

Cost note (plan §13): the matrix is 7 ablations × N examples ≈ 7×30 =
210 LLM calls per backend. NIM free-tier credits accommodate one full
matrix per model per day; rerun by `workflow_dispatch` only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.evaluation.ablation import ABLATIONS  # noqa: E402
from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.evaluation.manifest import make_run_id  # noqa: E402
from anvil.evaluation.run_logger import RunLogger, RunLoggerConfig  # noqa: E402
from anvil.evaluation.runner import EvaluationRunner  # noqa: E402
from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

log = get_logger("anvil.run_ablations")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        default=os.environ.get("ANVIL_LLM_BACKEND", "fake"),
        choices=["fake", "nvidia_nim", "openai_compatible", "instructor"],
    )
    p.add_argument("--model", default=os.environ.get("ANVIL_LLM_MODEL"))
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Run only the named ablations (default: all 7). Useful for "
            "quick local verification before kicking off a full matrix."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/runs"),
    )
    p.add_argument(
        "--write-table",
        type=Path,
        default=Path("docs/ablations_table.md"),
        help="Where to write the per-ablation comparison table.",
    )
    return p.parse_args()


async def _run() -> int:
    args = _parse_args()
    if args.backend != "fake":
        os.environ["ANVIL_LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["ANVIL_LLM_MODEL"] = args.model

    requested = args.only or list(ABLATIONS)
    unknown = sorted(set(requested) - set(ABLATIONS))
    if unknown:
        sys.stderr.write(
            f"unknown ablations: {unknown}. Known: {sorted(ABLATIONS)}\n"
        )
        return 2

    dataset_path = ROOT / "tests" / "evaluation" / "golden_dataset.json"
    examples = load_golden_dataset(dataset_path)

    when = datetime.now(UTC)
    run_dirs: list[Path] = []
    for ablation_name in requested:
        ablation_cfg = ABLATIONS[ablation_name]
        log.info(
            "ablation.start",
            backend=args.backend,
            model=args.model,
            ablation=ablation_name,
        )
        pipeline = build_pipeline(ablation=ablation_cfg)
        runner = EvaluationRunner(pipeline.generator)
        run_id = make_run_id(
            backend=args.backend,
            model=args.model,
            ablation=ablation_name,
            when=when,
        )
        cfg = RunLoggerConfig(
            run_id=run_id,
            backend=args.backend,
            model=args.model,
            ablation=ablation_name,
            ablation_config=ablation_cfg.to_summary(),
            dataset_path=dataset_path,
            output_root=args.output_root,
        )
        run_dir = args.output_root / run_id
        async with RunLogger(cfg) as rl:
            rl.attach_examples(examples)
            summary = await runner.run(examples)
            rl.write_summary(summary)
            for ex in examples:
                outcome = await pipeline.generator.generate(ex.query, top_k=10)
                rl.record_example(
                    ex.id,
                    outcome.response,
                    retrieved=outcome.retrieved_chunks,
                )
        run_dirs.append(run_dir)
        log.info(
            "ablation.done",
            ablation=ablation_name,
            pass_rate=summary.pass_rate,
            run_dir=str(run_dir),
        )

    # Render the ablation comparison table by reusing compare_runs.py.
    # The module sits next to this one in `scripts/`; we add the
    # scripts directory to sys.path rather than turning it into a
    # package — avoids inventing an import name and keeps the layout
    # flat.
    if args.write_table and run_dirs:
        sys.path.insert(0, str(Path(__file__).parent))
        from compare_runs import render_table  # type: ignore[import-not-found]

        md = render_table(run_dirs)
        args.write_table.parent.mkdir(parents=True, exist_ok=True)
        args.write_table.write_text(md)
        print(f"\nWrote {args.write_table}")

    print("\n=== Ablation matrix summary ===")
    print(
        json.dumps(
            {"runs": [str(d) for d in run_dirs], "n": len(run_dirs)},
            indent=2,
        )
    )
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
