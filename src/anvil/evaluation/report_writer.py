"""Render per-run Markdown reports from a `RunSummary` + `RunManifest`.

Plain-string formatting on purpose — adding Jinja2 for one ~100 line
template is overkill and adds a CI dependency we don't otherwise need.
The output is the file a reviewer reads first, so its structure is
ordered for skimming:

  1. headline (config + headline numbers)
  2. metric-by-metric table (means)
  3. top failing examples (most informative)
  4. footnote with reproducibility info

Every claim that lands in `docs/report.md` is footnoted with the
run-id of the run that produced it (see plan §11), so this report's
file path itself becomes the citation.
"""

from __future__ import annotations

from pathlib import Path

from anvil.evaluation.manifest import RunManifest
from anvil.evaluation.runner import RunSummary
from anvil.schemas.evaluation import EvaluationResult, MetricScore

_METRIC_ORDER = [
    "refusal_calibration",
    "retrieval_precision_at_k",
    "retrieval_recall_at_k",
    "faithfulness",
    "citation_accuracy",
    "entity_grounding",
    "structural_completeness",
    "calculation_correctness",
]


def _fmt(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{ndigits}f}"


def _ranked_metrics(summary: RunSummary) -> list[str]:
    """Stable order: known metrics first, anything new alphabetically."""
    known = [m for m in _METRIC_ORDER if m in summary.aggregate]
    extras = sorted(set(summary.aggregate) - set(_METRIC_ORDER))
    return known + extras


def _failing_examples(
    summary: RunSummary, limit: int = 5
) -> list[EvaluationResult]:
    """Return up to `limit` failed examples, lowest-score first.

    A "score" here is the mean of all metric values for that example,
    so the ones at the top of the list are the most broken across
    metrics — that's what a reviewer wants to triage first.
    """
    failed = [r for r in summary.per_example if not r.passed]

    def score(r: EvaluationResult) -> float:
        if not r.metrics:
            return 1.0
        return sum(m.value for m in r.metrics) / len(r.metrics)

    return sorted(failed, key=score)[:limit]


def _metric_row(name: str, score: MetricScore) -> str:
    pass_mark = "✓" if score.passed else "✗"
    threshold = _fmt(score.threshold)
    return f"| `{name}` | {_fmt(score.value)} | {threshold} | {pass_mark} |"


def render_report(
    summary: RunSummary,
    manifest: RunManifest,
    *,
    artifact_links: dict[str, str] | None = None,
) -> str:
    """Build the `report.md` string."""
    artifact_links = artifact_links or {
        "summary.json": "summary.json",
        "per_example.json": "per_example.json",
        "manifest.json": "manifest.json",
    }

    n = manifest.n_examples or len(summary.per_example)
    pass_rate = summary.pass_rate

    out: list[str] = []
    out.append(f"# Run report — `{manifest.run_id}`")
    out.append("")
    out.append(
        f"**Pass rate: `{pass_rate:.1%}`** "
        f"({sum(1 for r in summary.per_example if r.passed)} / {len(summary.per_example)} "
        f"examples passed every metric threshold)."
    )
    out.append("")

    out.append("## Run configuration")
    out.append("")
    out.append("| field | value |")
    out.append("| :--- | :--- |")
    out.append(f"| backend | `{manifest.backend}` |")
    if manifest.model:
        out.append(f"| model | `{manifest.model}` |")
    if manifest.base_url:
        out.append(f"| base_url | `{manifest.base_url}` |")
    out.append(f"| ablation | `{manifest.ablation}` |")
    if manifest.ablation_config:
        for k, v in sorted(manifest.ablation_config.items()):
            out.append(f"| ablation_config.{k} | `{v}` |")
    out.append(f"| dataset | `{manifest.dataset_path or '—'}` |")
    if manifest.dataset_hash:
        out.append(f"| dataset_hash | `{manifest.dataset_hash[:12]}…` |")
    out.append(f"| n_examples | {n} |")
    out.append(f"| git_sha | `{manifest.git_sha or '—'}` |")
    if manifest.git_dirty:
        out.append("| git_dirty | **yes — manifest is from a dirty worktree** |")
    out.append(f"| git_branch | `{manifest.git_branch or '—'}` |")
    out.append(f"| python | `{manifest.python_version}` |")
    out.append(f"| started_at_utc | `{manifest.started_at_utc}` |")
    if manifest.finished_at_utc:
        out.append(f"| finished_at_utc | `{manifest.finished_at_utc}` |")
    out.append("")

    out.append("## Aggregate metrics")
    out.append("")
    out.append("| metric | mean | threshold (per-example) | per-example pass |")
    out.append("| :--- | ---: | ---: | :---: |")
    for name in _ranked_metrics(summary):
        mean = summary.aggregate.get(name)
        # Pull a per-example threshold + pass-rate (across the run) for
        # display. Threshold is constant per metric in the current
        # implementation; if that ever changes the mean displayed here
        # is still correct.
        per_example_passes = sum(
            1
            for r in summary.per_example
            for m in r.metrics
            if m.name == name and m.passed
        )
        per_example_total = sum(
            1 for r in summary.per_example for m in r.metrics if m.name == name
        )
        any_score = next(
            (
                m
                for r in summary.per_example
                for m in r.metrics
                if m.name == name
            ),
            None,
        )
        threshold = _fmt(any_score.threshold) if any_score else "—"
        pass_str = (
            f"{per_example_passes}/{per_example_total}"
            if per_example_total
            else "—"
        )
        out.append(f"| `{name}` | {_fmt(mean)} | {threshold} | {pass_str} |")
    out.append("")

    failing = _failing_examples(summary)
    if failing:
        out.append("## Failing examples (worst-first)")
        out.append("")
        for r in failing:
            out.append(f"### `{r.example_id}` — _{r.category}_")
            out.append("")
            out.append("| metric | value | threshold | passed |")
            out.append("| :--- | ---: | ---: | :---: |")
            for m in r.metrics:
                out.append(_metric_row(m.name, m))
            out.append("")
    else:
        out.append("## Failing examples")
        out.append("")
        out.append("_None — every example passed every metric threshold._")
        out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append(
        "To reproduce, check out the recorded git sha and re-run with the "
        "same env vars (the manifest captures the allowlisted set with "
        "secret values redacted)."
    )
    out.append("")
    out.append("## Artifacts")
    out.append("")
    for label, link in sorted(artifact_links.items()):
        out.append(f"- [`{label}`]({link})")
    out.append("")
    out.append(
        "Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, "
        "`request_log.jsonl`) are produced locally but gitignored — "
        "see `.gitignore` and the plan §3.1 storage policy."
    )
    return "\n".join(out) + "\n"


def write_report(
    summary: RunSummary,
    manifest: RunManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, manifest))
