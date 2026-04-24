"""Run the full evaluation suite against the golden dataset."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.evaluation.dataset import load_golden_dataset  # noqa: E402
from anvil.evaluation.runner import EvaluationRunner, write_summary_json  # noqa: E402
from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

log = get_logger("anvil.evaluate")


async def _run() -> None:
    pipeline = build_pipeline()
    runner = EvaluationRunner(pipeline.generator)
    examples = load_golden_dataset(ROOT / "tests" / "evaluation" / "golden_dataset.json")
    summary = await runner.run(examples)
    out = ROOT / "data" / "indexes" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_summary_json(summary, out)
    print("\n=== Evaluation summary ===")
    print(json.dumps({"aggregate": summary.aggregate, "pass_rate": summary.pass_rate}, indent=2))
    print(f"Saved detailed results to {out}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
