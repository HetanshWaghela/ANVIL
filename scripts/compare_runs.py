"""Compare N run directories side-by-side.

Reads each run's `summary.json` (the committed file) and emits a pivot
table: rows are runs (labeled by `<backend>/<model>/<ablation>`) and
columns are metrics. Used to produce:

  * the README headline table (per-NIM-model comparison)
  * `docs/ablations.md` (per-ablation comparison)
  * `docs/pipeline_vs_agent.md` (M6)

Usage:

    uv run python scripts/compare_runs.py data/runs/run-id-1 data/runs/run-id-2
    uv run python scripts/compare_runs.py data/runs/* --out docs/results_table.md

The output is GitHub-flavored Markdown so it pastes directly into any
`.md` document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Display order — reuse the same ranking the report writer uses.
_METRIC_ORDER = [
    "pass_rate",
    "calculation_correctness",
    "citation_accuracy",
    "faithfulness",
    "entity_grounding",
    "structural_completeness",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "refusal_calibration",
]


def _load_summary(run_dir: Path) -> dict[str, Any]:
    sj = run_dir / "summary.json"
    if not sj.exists():
        raise FileNotFoundError(f"{sj} does not exist — is this a run directory?")
    return json.loads(sj.read_text())


def _label(summary: dict[str, Any], run_dir: Path) -> str:
    """Compact human-readable row label."""
    m = summary.get("manifest", {})
    backend = m.get("backend") or "?"
    model = m.get("model")
    ablation = m.get("ablation") or "baseline"
    parts = [backend]
    if model:
        parts.append(model.split("/")[-1])
    parts.append(f"abl-{ablation}")
    return " / ".join(parts) or run_dir.name


def _fmt(value: float | None, ndigits: int = 3) -> str:
    return "—" if value is None else f"{value:.{ndigits}f}"


def render_table(run_dirs: list[Path]) -> str:
    rows: list[tuple[str, dict[str, float | None], str]] = []
    for d in run_dirs:
        try:
            s = _load_summary(d)
        except FileNotFoundError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            continue
        agg = dict(s.get("aggregate") or {})
        agg["pass_rate"] = s.get("pass_rate")
        rows.append((_label(s, d), agg, d.name))

    if not rows:
        return "_No runs found._\n"

    metric_cols = [m for m in _METRIC_ORDER if any(m in r[1] for r in rows)]
    extras = sorted(
        {m for _, agg, _ in rows for m in agg} - set(metric_cols)
    )
    metric_cols += extras

    header = ["run"] + metric_cols + ["run_id"]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append(
        "| " + " | ".join([":---"] + ["---:"] * len(metric_cols) + [":---"]) + " |"
    )
    for label, agg, run_id in rows:
        cells = [label]
        for m in metric_cols:
            v = agg.get(m)
            cells.append(_fmt(v if isinstance(v, (int, float)) else None))
        cells.append(f"`{run_id}`")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Write Markdown here.")
    args = p.parse_args(argv)
    md = render_table(args.run_dirs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
