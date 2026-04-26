"""Run a local licensed-ASME evaluation without emitting source text.

This script is intentionally fail-closed:
- standard and dataset inputs must live under data/private/
- output-root must live under data/private_runs/
- raw responses, prompts, retrieved chunks, and agent transcripts are never written

Only aggregate metrics and per-example metric vectors are persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.evaluation.ablation import ABLATIONS  # noqa: E402
from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.evaluation.manifest import make_run_id  # noqa: E402
from anvil.evaluation.runner import EvaluationRunner  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

PRIVATE_INPUT_ROOT = ROOT / "data" / "private"
PRIVATE_RUN_ROOT = ROOT / "data" / "private_runs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run private licensed-ASME eval and write sanitized metrics only."
    )
    p.add_argument("--backend", default=os.environ.get("ANVIL_LLM_BACKEND", "fake"))
    p.add_argument("--model", default=None, help="Override ANVIL_LLM_MODEL.")
    p.add_argument("--ablation", default="baseline", choices=sorted(ABLATIONS))
    p.add_argument("--standard", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=PRIVATE_RUN_ROOT)
    p.add_argument("--dataset-version", default="asme-private-v1")
    p.add_argument("--edition-label", default="ASME private licensed edition")
    p.add_argument("--min-pass-rate", type=float, default=0.0)
    p.add_argument(
        "--sanitized-output",
        type=Path,
        default=None,
        help="Optional JSON path for shareable aggregate metrics only.",
    )
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _require_under(path: Path, root: Path, label: str) -> Path:
    resolved = _resolve(path)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"{label} must live under {root.relative_to(ROOT)}; got {path}"
        ) from exc
    return resolved


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


async def _run() -> int:
    args = _parse_args()
    standard_path = _require_under(args.standard, PRIVATE_INPUT_ROOT, "--standard")
    dataset_path = _require_under(args.dataset, PRIVATE_INPUT_ROOT, "--dataset")
    output_root = _require_under(args.output_root, PRIVATE_RUN_ROOT, "--output-root")

    os.environ["ANVIL_LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["ANVIL_LLM_MODEL"] = args.model
    elif args.backend == "fake":
        os.environ.pop("ANVIL_LLM_MODEL", None)

    pipeline = build_pipeline(standard_path=standard_path, ablation=args.ablation)
    effective_model = getattr(pipeline.generator.backend, "model", args.model)
    examples = load_golden_dataset(dataset_path)
    runner = EvaluationRunner(
        pipeline.generator,
        retry_backend_errors=args.backend != "fake",
    )

    run_id = make_run_id(
        backend=args.backend,
        model=effective_model,
        ablation=args.ablation,
        dataset_version=args.dataset_version,
    )
    run_dir = output_root / run_id
    summary = await runner.run(examples)

    sanitized = {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": args.backend,
        "model": effective_model,
        "ablation": args.ablation,
        "dataset_version": args.dataset_version,
        "edition_label": args.edition_label,
        "standard_path": _rel(standard_path),
        "dataset_path": _rel(dataset_path),
        "n_examples": len(summary.per_example),
        "n_passed": sum(1 for r in summary.per_example if r.passed),
        "pass_rate": summary.pass_rate,
        "aggregate": summary.aggregate,
        "unsafe_artifacts_written": False,
        "omitted_artifacts": [
            "raw_responses.jsonl",
            "prompts.jsonl",
            "request_log.jsonl",
            "agent_transcripts.json",
            "retrieved_chunks",
            "quoted_source_text",
        ],
    }
    per_example = [result.model_dump() for result in summary.per_example]

    _write_json(run_dir / "summary.sanitized.json", sanitized)
    _write_json(run_dir / "per_example.metrics.json", per_example)
    if args.sanitized_output is not None:
        _write_json(_resolve(args.sanitized_output), sanitized)

    print(json.dumps(sanitized, indent=2, default=str))
    return 0 if summary.pass_rate >= args.min_pass_rate else 1


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
