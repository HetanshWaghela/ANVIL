"""Anvil command-line interface.

Single entry point installed as `anvil` (see `[project.scripts]` in
`pyproject.toml`). Subcommands are added incrementally per the
implementation plan; M1 ships only `nim-check`. Subsequent milestones
add `ingest`, `query`, `calculate`, `eval`, and `compare`.

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
import sys
from typing import Any

from anvil.generation.nim_health import (
    DEFAULT_NIM_BASE_URL,
    NIM_MODELS,
    check_all_nim_models,
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
                    f"        tokens  : prompt={r.prompt_tokens} "
                    f"completion={r.completion_tokens}"
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
        tokens = (
            f"{r.prompt_tokens or 0}/{r.completion_tokens or 0}"
            if r.reachable
            else "—"
        )
        note = (
            (r.sample_response or "").strip()
            if r.reachable
            else (r.error or "")[:80]
        )
        table.add_row(status, r.label, r.model, latency, tokens, note)
    console.print(table)
    ok = sum(1 for r in results if r.reachable)
    console.print(
        f"[bold]{ok}/{len(results)}[/bold] models reachable. "
        f"Base URL: {results[0].base_url if results else DEFAULT_NIM_BASE_URL}"
    )


async def _cmd_nim_check_async(args: argparse.Namespace) -> int:
    """Probe every model in NIM_MODELS and print a summary.

    With `--list`, additionally fetch the live `/v1/models` catalog
    and report any drift between the locked NIM_MODELS catalog and
    what NIM is actually serving today.
    """
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
        locked = set(NIM_MODELS)
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
        help="Probe each model in the locked NIM catalog and report status.",
        description=(
            "Run a small probe against each NVIDIA NIM model in the locked "
            "catalog (see NIM_MODELS). Prints a status table; exits non-zero "
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
