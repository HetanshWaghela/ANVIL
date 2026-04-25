"""Anvil command-line interface.

Single entry point installed as `anvil` (see `[project.scripts]` in
`pyproject.toml`). The CLI mirrors the public reviewer workflow:
`nim-check`, `ingest`, `query`, `calculate`, `eval`, and `compare`.

Design choices:
* Pure stdlib `argparse` — no Click dependency. CI must stay lean and
  the command surface is small enough that argparse is the right tool.
* Every subcommand is async-aware via `asyncio.run`. The pipeline is
  async-by-default; the CLI just forwards.
* Output is human-readable by default and JSON via `--json` for any
  command that produces structured data. CI consumes the JSON form.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from anvil.generation.nim_health import (
    DEFAULT_NIM_BASE_URL,
    check_all_nim_models,
    get_nim_model_catalog,
    list_nim_catalog,
)

# ---------------------------------------------------------------------------
# nim-check
# ---------------------------------------------------------------------------


def _nim_check_summary_text(results: list[Any]) -> str:
    """Plain-text fallback when `rich` is unavailable."""
    lines = []
    lines.append("ANVIL · NVIDIA NIM connection check")
    lines.append("=" * 60)
    for r in results:
        status = "OK " if r.reachable else "FAIL"
        latency = f"{r.latency_ms:6.1f} ms" if r.latency_ms is not None else "   —    "
        lines.append(f"[{status}] {r.label:<14} {latency}  {r.model}")
        lines.append(f"        purpose : {r.purpose}")
        if r.reachable:
            lines.append(f"        sample  : {r.sample_response!r}")
            if r.prompt_tokens is not None:
                lines.append(
                    f"        tokens  : prompt={r.prompt_tokens} completion={r.completion_tokens}"
                )
        else:
            lines.append(f"        error   : {r.error}")
        if r.request_id:
            lines.append(f"        req_id  : {r.request_id}")
    lines.append("=" * 60)
    ok = sum(1 for r in results if r.reachable)
    lines.append(f"summary: {ok}/{len(results)} reachable")
    return "\n".join(lines)


def _nim_check_summary_rich(results: list[Any]) -> None:
    """Pretty Rich table when the optional rich extra is installed."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="ANVIL · NVIDIA NIM connection check", show_lines=False)
    table.add_column("status", justify="center")
    table.add_column("label", style="cyan")
    table.add_column("model", style="dim")
    table.add_column("latency", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("note")
    for r in results:
        status = "[green]OK[/green]" if r.reachable else "[red]FAIL[/red]"
        latency = f"{r.latency_ms:.1f} ms" if r.latency_ms is not None else "—"
        tokens = f"{r.prompt_tokens or 0}/{r.completion_tokens or 0}" if r.reachable else "—"
        note = (r.sample_response or "").strip() if r.reachable else (r.error or "")[:80]
        table.add_row(status, r.label, r.model, latency, tokens, note)
    console.print(table)
    ok = sum(1 for r in results if r.reachable)
    console.print(
        f"[bold]{ok}/{len(results)}[/bold] models reachable. "
        f"Base URL: {results[0].base_url if results else DEFAULT_NIM_BASE_URL}"
    )


async def _cmd_nim_check_async(args: argparse.Namespace) -> int:
    """Probe every selected NIM model and print a summary.

    With `--list`, additionally fetch the live `/v1/models` catalog
    and report any drift between the selected model catalog and
    what NIM is actually serving today.
    """
    if args.models:
        import os

        os.environ["ANVIL_NIM_MODELS"] = args.models
    results = await check_all_nim_models(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    drift_payload: dict[str, Any] | None = None
    if args.list:
        live = await list_nim_catalog(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )
        locked = set(get_nim_model_catalog())
        live_set = set(live)
        drift_payload = {
            "locked": sorted(locked),
            "live_count": len(live),
            "missing_from_live": sorted(locked - live_set),
            "new_in_live": sorted(live_set - locked)[:25],  # cap to avoid noise
        }

    if args.json:
        out = {
            "results": [r.model_dump() for r in results],
            **({"catalog_drift": drift_payload} if drift_payload else {}),
        }
        print(json.dumps(out, indent=2, default=str))
        any_reachable = any(r.reachable for r in results)
        return 0 if any_reachable else 1

    try:
        _nim_check_summary_rich(results)
    except ImportError:
        print(_nim_check_summary_text(results))

    if not any(r.reachable for r in results):
        # Give the user a clear next step. Stderr so the green-path text
        # on stdout stays clean.
        sys.stderr.write(
            "\nNo NIM models are reachable. Common causes:\n"
            "  - NVIDIA_API_KEY unset → export it (https://build.nvidia.com)\n"
            "  - Free-tier quota exhausted → try again in a few hours\n"
            "  - Model rotated out of the catalog → see NIM_MODELS in "
            "src/anvil/generation/nim_health.py\n"
        )
        return 1
    return 0


def _add_nim_check_parser(sub: argparse._SubParsersAction[Any]) -> None:
    p = sub.add_parser(
        "nim-check",
        help="Probe each model in the selected NIM catalog and report status.",
        description=(
            "Run a small probe against each NVIDIA NIM model in the selected "
            "catalog. Prints a status table; exits non-zero "
            "if no model is reachable. Reads NVIDIA_API_KEY from env unless "
            "--api-key is given."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (defaults to NVIDIA_API_KEY env).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help=f"NIM base URL (default {DEFAULT_NIM_BASE_URL}).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-probe timeout in seconds (default 10).",
    )
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model ids to probe. Defaults to ANVIL_NIM_MODELS "
            "when set, otherwise the built-in production catalog."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help=(
            "Also fetch the live /v1/models catalog and report drift "
            "between the locked NIM_MODELS catalog and what's actually "
            "available today (handy when a model has rotated out)."
        ),
    )
    p.set_defaults(_handler=lambda args: asyncio.run(_cmd_nim_check_async(args)))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _apply_runtime_env(args: argparse.Namespace) -> None:
    """Apply common backend/model/embedder overrides before building a pipeline."""
    backend = getattr(args, "backend", None)
    if backend:
        os.environ["ANVIL_LLM_BACKEND"] = backend
    model = getattr(args, "model", None)
    if model:
        os.environ["ANVIL_LLM_MODEL"] = model
    embedder = getattr(args, "embedder", None)
    if embedder:
        os.environ["ANVIL_EMBEDDER"] = embedder


def _json_default(obj: object) -> str:
    """Fallback JSON serializer for Path / Decimal-ish values."""
    return str(obj)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Parse the selected standard and persist KG + element artifacts."""
    from anvil.pipeline import build_pipeline

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(standard_path=args.standard)
    pipeline.graph_store.save(out_dir / "kg.json")
    (out_dir / "elements.json").write_text(
        json.dumps(
            [e.model_dump(mode="json") for e in pipeline.elements],
            indent=2,
            default=_json_default,
        )
    )

    payload = {
        "elements": len(pipeline.elements),
        "nodes": len(pipeline.graph_store.graph.nodes),
        "edges": len(pipeline.graph_store.graph.edges),
        "output_dir": str(out_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Parsed {payload['elements']} elements.")
        print(f"Graph: {payload['nodes']} nodes, {payload['edges']} edges.")
        print(f"Saved to {out_dir}")
    return 0


def _add_ingest_parser(sub: argparse._SubParsersAction[Any]) -> None:
    p = sub.add_parser(
        "ingest",
        help="Parse SPES-1 and write KG / element artifacts.",
        description="Parse a standard into the local ANVIL graph artifacts.",
    )
    p.add_argument(
        "--standard",
        type=Path,
        default=None,
        help="Markdown standard path (default: bundled SPES-1).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes"),
        help="Output directory for kg.json and elements.json.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON summary.")
    p.set_defaults(_handler=_cmd_ingest)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


async def _cmd_query_async(args: argparse.Namespace) -> int:
    """Run a natural-language query through the full pipeline."""
    from anvil.pipeline import build_pipeline

    _apply_runtime_env(args)
    pipeline = build_pipeline(standard_path=args.standard)
    outcome = await pipeline.generator.generate(args.query, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                {
                    "response": outcome.response.model_dump(mode="json"),
                    "retrieved_element_ids": [c.element_id for c in outcome.retrieved_chunks],
                    "citation_validation": {
                        "total": outcome.citation_validation.total,
                        "valid": outcome.citation_validation.valid,
                        "accuracy": outcome.citation_validation.accuracy,
                        "issues": [
                            {
                                "citation_index": i.citation_index,
                                "citation": i.citation.model_dump(mode="json"),
                                "issue": i.issue,
                            }
                            for i in outcome.citation_validation.issues
                        ],
                    },
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    r = outcome.response
    print(r.answer)
    if r.refusal_reason:
        print(f"\nRefusal reason: {r.refusal_reason}")
    if r.calculation_steps:
        print("\nCalculation steps:")
        for step in r.calculation_steps:
            print(
                f"  {step.step_number}. {step.result_key}: "
                f"{step.formula} -> {step.result} {step.unit} "
                f"({step.citation.paragraph_ref})"
            )
    if r.citations:
        print("\nCitations:")
        for c in r.citations:
            quote = c.quoted_text.replace("\n", " ")[:120]
            print(f"  - {c.paragraph_ref} [{c.source_element_id}]: {quote}")
    print(f"\nConfidence: {r.confidence}")
    return 0


def _add_query_parser(sub: argparse._SubParsersAction[Any]) -> None:
    p = sub.add_parser(
        "query",
        help="Ask a natural-language question against the ANVIL pipeline.",
    )
    p.add_argument("query", help="Natural-language query.")
    p.add_argument("--top-k", type=int, default=10, help="Retriever top-k.")
    p.add_argument(
        "--standard",
        type=Path,
        default=None,
        help="Markdown standard path (default: bundled SPES-1).",
    )
    p.add_argument(
        "--backend",
        choices=["fake", "nvidia_nim", "openai_compatible", "instructor"],
        default=None,
        help="Override ANVIL_LLM_BACKEND.",
    )
    p.add_argument("--model", default=None, help="Override ANVIL_LLM_MODEL.")
    p.add_argument(
        "--embedder",
        choices=["hash", "sentence_transformer"],
        default=None,
        help="Override ANVIL_EMBEDDER.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON response.")
    p.set_defaults(_handler=lambda args: asyncio.run(_cmd_query_async(args)))


# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


def _cmd_calculate(args: argparse.Namespace) -> int:
    """Run the deterministic calculation engine with explicit inputs."""
    from anvil.generation.calculation_engine import CalculationInputs
    from anvil.pipeline import build_pipeline

    _apply_runtime_env(args)
    pipeline = build_pipeline(standard_path=args.standard)
    result = pipeline.generator.calc_engine.calculate(
        CalculationInputs(
            component=args.component,
            P_mpa=args.P_mpa,
            design_temp_c=args.design_temp_c,
            material=args.material,
            joint_type=args.joint_type,
            rt_level=args.rt_level,
            corrosion_allowance_mm=args.corrosion_allowance_mm,
            inside_diameter_mm=args.inside_diameter_mm,
            outside_diameter_mm=args.outside_diameter_mm,
        )
    )

    payload = {
        "formula_ref": result.formula_ref,
        "S_mpa": result.S_mpa,
        "E": result.E,
        "R_mm": result.R_mm,
        "applicability_lhs": result.applicability_lhs,
        "applicability_rhs": result.applicability_rhs,
        "applicability_ok": result.applicability_ok,
        "t_min_mm": result.t_min_mm,
        "t_design_mm": result.t_design_mm,
        "t_nominal_mm": result.t_nominal_mm,
        "mawp_mpa": result.mawp_mpa,
        "warnings": result.warnings,
        "steps": [s.model_dump(mode="json") for s in result.steps],
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        print(f"Formula: {result.formula_ref}")
        print(f"S = {result.S_mpa} MPa")
        print(f"E = {result.E}")
        print(f"R = {result.R_mm} mm")
        print(f"Applicability: {result.applicability_lhs} <= {result.applicability_rhs}")
        print(f"t_min = {result.t_min_mm:.2f} mm")
        print(f"t_design = {result.t_design_mm:.2f} mm")
        print(f"t_nominal = {result.t_nominal_mm} mm")
        print(f"MAWP = {result.mawp_mpa:.3f} MPa")
    return 0


def _add_calculate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    p = sub.add_parser(
        "calculate",
        help="Run a deterministic SPES-1 shell thickness calculation.",
    )
    p.add_argument(
        "--component",
        required=True,
        choices=[
            "cylindrical_shell",
            "cylindrical_shell_outside_radius",
            "spherical_shell",
        ],
    )
    p.add_argument("--P", dest="P_mpa", type=float, required=True, help="Pressure MPa.")
    p.add_argument(
        "--temp",
        dest="design_temp_c",
        type=float,
        required=True,
        help="Design temperature °C.",
    )
    p.add_argument("--material", required=True, help="Material spec/grade.")
    p.add_argument("--joint-type", type=int, required=True, help="Weld joint type.")
    p.add_argument(
        "--rt-level",
        required=True,
        choices=["Full RT", "Spot RT", "No RT"],
        help="Radiographic examination extent.",
    )
    p.add_argument(
        "--ca",
        dest="corrosion_allowance_mm",
        type=float,
        required=True,
        help="Corrosion allowance in mm.",
    )
    p.add_argument("--inside-diameter-mm", type=float, default=None)
    p.add_argument("--outside-diameter-mm", type=float, default=None)
    p.add_argument(
        "--standard",
        type=Path,
        default=None,
        help="Markdown standard path (default: bundled SPES-1).",
    )
    p.add_argument(
        "--embedder",
        choices=["hash", "sentence_transformer"],
        default=None,
        help="Override ANVIL_EMBEDDER.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON response.")
    p.set_defaults(_handler=_cmd_calculate)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


async def _cmd_eval_async(args: argparse.Namespace) -> int:
    """Run the golden-dataset evaluation and write a stamped run directory."""
    from anvil.evaluation.ablation import ABLATIONS
    from anvil.evaluation.dataset import load_golden_dataset
    from anvil.evaluation.manifest import make_run_id
    from anvil.evaluation.run_logger import RunLogger, RunLoggerConfig
    from anvil.evaluation.runner import EvaluationRunner
    from anvil.pipeline import build_pipeline

    _apply_runtime_env(args)
    if args.backend == "fake" and not args.model:
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
    cfg = RunLoggerConfig(
        run_id=run_id,
        backend=args.backend,
        model=effective_model,
        ablation=args.ablation,
        ablation_config=ABLATIONS[args.ablation].to_summary(),
        dataset_path=dataset_path,
        output_root=args.output_root,
    )
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        summary = await runner.run(examples)
        rl.write_summary(summary)
        for ex, outcome in zip(examples, summary.outcomes, strict=True):
            rl.record_example(
                ex.id,
                outcome.response,
                retrieved=outcome.retrieved_chunks,
            )

    payload = {
        "run_id": run_id,
        "backend": args.backend,
        "model": effective_model,
        "ablation": args.ablation,
        "dataset_version": args.dataset_version,
        "pass_rate": summary.pass_rate,
        "aggregate": summary.aggregate,
        "run_dir": str(args.output_root / run_id),
    }
    print(json.dumps(payload, indent=2, default=_json_default))
    return 0 if summary.pass_rate >= args.min_pass_rate else 1


def _add_eval_parser(sub: argparse._SubParsersAction[Any]) -> None:
    from anvil.evaluation.ablation import ABLATIONS

    p = sub.add_parser(
        "eval",
        help="Run the golden-dataset evaluation and write data/runs artifacts.",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("ANVIL_LLM_BACKEND", "fake"),
        choices=["fake", "nvidia_nim", "openai_compatible", "instructor"],
    )
    p.add_argument("--model", default=None, help="Override ANVIL_LLM_MODEL.")
    p.add_argument(
        "--embedder",
        choices=["hash", "sentence_transformer"],
        default=None,
        help="Override ANVIL_EMBEDDER.",
    )
    p.add_argument(
        "--ablation",
        default="baseline",
        choices=sorted(ABLATIONS),
        help="Ablation slug.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/evaluation/golden_dataset.json"),
        help="Golden dataset JSON.",
    )
    p.add_argument(
        "--standard",
        type=Path,
        default=Path("data/synthetic/standard.md"),
        help="Markdown standard to parse for retrieval and citations.",
    )
    p.add_argument(
        "--dataset-version",
        default="goldenv1",
        help="Run-id dataset slug, e.g. goldenv1 or asme-private-v1.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/runs"),
        help="Run artifact root.",
    )
    p.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.7,
        help="Exit non-zero if pass_rate is below this threshold.",
    )
    p.set_defaults(_handler=lambda args: asyncio.run(_cmd_eval_async(args)))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist")
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path} did not contain a JSON object")
    return payload


def _run_label(summary: dict[str, Any], run_dir: Path) -> str:
    manifest = summary.get("manifest") or {}
    if not isinstance(manifest, dict):
        return run_dir.name
    backend = manifest.get("backend") or "?"
    model = manifest.get("model")
    ablation = manifest.get("ablation") or "baseline"
    parts = [str(backend)]
    if model:
        parts.append(str(model).split("/")[-1])
    parts.append(f"abl-{ablation}")
    return " / ".join(parts)


def _format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.3f}"
    return "—"


def _render_compare_table(run_dirs: list[Path]) -> str:
    metric_order = [
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
    rows: list[tuple[str, dict[str, object], str]] = []
    for run_dir in run_dirs:
        summary = _load_run_summary(run_dir)
        aggregate = summary.get("aggregate") or {}
        if not isinstance(aggregate, dict):
            aggregate = {}
        metrics: dict[str, object] = dict(aggregate)
        metrics["pass_rate"] = summary.get("pass_rate")
        rows.append((_run_label(summary, run_dir), metrics, run_dir.name))

    if not rows:
        return "_No run summaries found._\n"

    metric_cols = [m for m in metric_order if any(m in metrics for _, metrics, _ in rows)]
    extra_cols = sorted({m for _, metrics, _ in rows for m in metrics} - set(metric_cols))
    metric_cols.extend(extra_cols)

    lines = ["| run | " + " | ".join(metric_cols) + " | run_id |"]
    lines.append("| :--- | " + " | ".join("---:" for _ in metric_cols) + " | :--- |")
    for label, metrics, run_id in rows:
        cells = [label]
        cells.extend(_format_metric(metrics.get(m)) for m in metric_cols)
        cells.append(f"`{run_id}`")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cmd_compare(args: argparse.Namespace) -> int:
    """Render a Markdown comparison table for stamped run directories."""
    table = _render_compare_table(args.run_dirs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
        print(f"wrote {args.out}")
    else:
        print(table)
    return 0


def _add_compare_parser(sub: argparse._SubParsersAction[Any]) -> None:
    p = sub.add_parser(
        "compare",
        help="Compare one or more data/runs directories as a Markdown table.",
    )
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Write Markdown here.")
    p.set_defaults(_handler=_cmd_compare)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil",
        description=(
            "Anvil — compliance-grade RAG over engineering standards. "
            "See `docs/` for architecture, evaluation methodology, and ADRs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_nim_check_parser(sub)
    _add_ingest_parser(sub)
    _add_query_parser(sub)
    _add_calculate_parser(sub)
    _add_eval_parser(sub)
    _add_compare_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error(f"no handler registered for command {args.command!r}")
    rc = handler(args)
    if not isinstance(rc, int):
        rc = 0
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
