"""M5 — produce the README headline table by evaluating each NIM model.

Iterates the locked 3-model `NIM_MODELS` catalog and runs the baseline
ablation on each. Produces:

  * one stamped run directory per model under `data/runs/`,
  * a regenerated `docs/headline_results.md` table comparing the 3
    models against the FakeLLMBackend baseline.

Cost (plan §6): NIM free-tier credits accommodate ~30 examples × 3
models = 90 requests. Sequential execution + the 0-temperature probes
keep latency predictable. Aborts immediately if `NVIDIA_API_KEY` is
unset so the script never silently runs against `FakeLLMBackend` in
production CI.
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

from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.evaluation.manifest import make_run_id  # noqa: E402
from anvil.evaluation.run_logger import RunLogger, RunLoggerConfig  # noqa: E402
from anvil.evaluation.runner import EvaluationRunner  # noqa: E402
from anvil.generation.nim_health import get_nim_model_catalog  # noqa: E402
from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

log = get_logger("anvil.run_nim_headlines")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--include-fake",
        action="store_true",
        help="Also run the FakeLLMBackend baseline so the table includes a control row.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/runs"),
    )
    p.add_argument(
        "--write-table",
        type=Path,
        default=Path("docs/headline_results.md"),
    )
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated NIM model ids. Defaults to ANVIL_NIM_MODELS "
            "when set, otherwise the built-in production catalog."
        ),
    )
    return p.parse_args()


async def _evaluate_one(
    *,
    backend: str,
    model: str | None,
    output_root: Path,
    when: datetime,
) -> Path:
    """Run one (backend, model) pair end-to-end → returns the run_dir."""
    if backend != "fake":
        os.environ["ANVIL_LLM_BACKEND"] = backend
    else:
        os.environ.pop("ANVIL_LLM_BACKEND", None)
    if model:
        os.environ["ANVIL_LLM_MODEL"] = model
    else:
        os.environ.pop("ANVIL_LLM_MODEL", None)

    pipeline = build_pipeline(ablation="baseline")
    effective_model = getattr(pipeline.generator.backend, "model", model)
    runner = EvaluationRunner(
        pipeline.generator,
        retry_backend_errors=backend != "fake",
    )
    dataset_path = ROOT / "tests" / "evaluation" / "golden_dataset.json"
    examples = load_golden_dataset(dataset_path)

    run_id = make_run_id(
        backend=backend, model=effective_model, ablation="baseline", when=when
    )
    cfg = RunLoggerConfig(
        run_id=run_id,
        backend=backend,
        model=effective_model,
        ablation="baseline",
        dataset_path=dataset_path,
        output_root=output_root,
    )
    log.info("nim_headline.start", backend=backend, model=model, run_id=run_id)
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        summary = await runner.run(examples)
        rl.write_summary(summary)
        # Use cached outcomes from the runner — re-invoking the generator
        # here would double the NIM call count for every example.
        for ex, outcome in zip(examples, summary.outcomes, strict=True):
            rl.record_example(
                ex.id, outcome.response, retrieved=outcome.retrieved_chunks
            )
    log.info(
        "nim_headline.done",
        backend=backend,
        model=model,
        pass_rate=summary.pass_rate,
    )
    return output_root / run_id


async def _run() -> int:
    args = _parse_args()
    if args.models:
        os.environ["ANVIL_NIM_MODELS"] = args.models

    has_key = bool(os.environ.get("NVIDIA_API_KEY"))
    if not has_key:
        sys.stderr.write(
            "NVIDIA_API_KEY is not set — `run_nim_headlines.py` can't run "
            "the live NIM rows. To produce the FakeLLMBackend baseline only, "
            "use `scripts/evaluate.py` directly. Aborting.\n"
        )
        return 1

    when = datetime.now(UTC)
    run_dirs: list[Path] = []
    if args.include_fake:
        run_dirs.append(
            await _evaluate_one(
                backend="fake",
                model=None,
                output_root=args.output_root,
                when=when,
            )
        )
    for model_id in get_nim_model_catalog():
        run_dirs.append(
            await _evaluate_one(
                backend="nvidia_nim",
                model=model_id,
                output_root=args.output_root,
                when=when,
            )
        )

    sys.path.insert(0, str(Path(__file__).parent))
    from compare_runs import render_table  # type: ignore[import-not-found]

    md = render_table(run_dirs)
    args.write_table.parent.mkdir(parents=True, exist_ok=True)
    args.write_table.write_text(md)
    print(f"\nWrote {args.write_table}")
    print(json.dumps([str(d) for d in run_dirs], indent=2))
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
