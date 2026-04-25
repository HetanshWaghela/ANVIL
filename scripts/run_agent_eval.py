"""M6 — agent vs. fixed-pipeline head-to-head on the golden dataset.

Run the same golden examples through:
  1. The fixed `EvaluationRunner` (retrieve → calc → generate),
  2. `AgentEvaluationRunner` driven by `LLMAgentBackend` (the real
     LLM picks tools turn-by-turn).

Aborts unless `NVIDIA_API_KEY` is set — the agent loop is only
interesting on a real model. For CI / local sanity checks, see
`tests/integration/test_agent_eval.py` which exercises the runner
plumbing with a scripted decider.

Outputs:
  data/runs/<run_id>__agent/   — full RunLogger artifact tree
  data/runs/<run_id>__fixed/   — peer fixed-pipeline run
  docs/agent_results.md        — Markdown comparison table
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

from anvil.evaluation import (  # noqa: E402
    AgentEvaluationRunner,
    AgentRunSummary,
    EvaluationRunner,
    RunLogger,
    RunLoggerConfig,
    load_golden_dataset,
    make_run_id,
)
from anvil.generation.agent import AnvilAgent  # noqa: E402
from anvil.generation.agent_backend import LLMAgentBackend  # noqa: E402
from anvil.generation.agent_tools import ToolRegistry  # noqa: E402
from anvil.generation.llm_backend import get_default_backend  # noqa: E402
from anvil.generation.nim_health import NIM_MODELS  # noqa: E402
from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402
from anvil.schemas.agent import AgentBudget  # noqa: E402

log = get_logger("anvil.run_agent_eval")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=next(iter(NIM_MODELS)),
        help="NIM model id (default: head of the locked catalog).",
    )
    p.add_argument(
        "--max-steps", type=int, default=8, help="Agent step budget (default 8)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N golden examples (default: all).",
    )
    p.add_argument(
        "--output-root", type=Path, default=Path("data/runs"),
    )
    p.add_argument(
        "--write-table", type=Path, default=Path("docs/agent_results.md"),
    )
    p.add_argument(
        "--skip-fixed",
        action="store_true",
        help="Don't re-run the fixed pipeline as a comparison baseline.",
    )
    p.add_argument(
        "--sleep-between-examples",
        type=float,
        default=0.0,
        help="Cooldown between agent examples, useful for hosted provider rate limits.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run helpers — kept simple, no nested logging contexts.
# ---------------------------------------------------------------------------


async def _run_agent(
    *,
    model: str,
    max_steps: int,
    sleep_between_examples: float,
    examples: list[Any],
    output_root: Path,
    when: datetime,
    dataset_path: Path,
) -> tuple[Path, AgentRunSummary]:
    os.environ["ANVIL_LLM_BACKEND"] = "nvidia_nim"
    os.environ["ANVIL_LLM_MODEL"] = model

    pipeline = build_pipeline(ablation="baseline")
    backend = get_default_backend()
    decider = LLMAgentBackend(backend=backend, model=model)
    registry = ToolRegistry(
        retriever=pipeline.retriever,
        graph_store=pipeline.graph_store,
        calc_engine=pipeline.generator.calc_engine,
    )
    agent = AnvilAgent(
        decider=decider,
        registry=registry,
        budget=AgentBudget(max_steps=max_steps),
    )
    runner = AgentEvaluationRunner(
        agent=agent,
        element_index=pipeline.generator.element_index,
        inter_example_delay_s=sleep_between_examples,
    )

    run_id = (
        make_run_id(
            backend="nvidia_nim", model=model, ablation="baseline", when=when
        )
        + "__agent"
    )
    cfg = RunLoggerConfig(
        run_id=run_id,
        backend="nvidia_nim",
        model=model,
        ablation="baseline-agent",
        dataset_path=dataset_path,
        output_root=output_root,
    )
    log.info("agent_eval.start", run_id=run_id)
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        summary = await runner.run(examples)
        rl.write_summary(summary)
        for ex, outcome in zip(examples, summary.outcomes, strict=True):
            # Use the union-of-retrieve as the chunks-of-record so the
            # per_example artifact lines up with what the metrics saw.
            from anvil.evaluation.agent_runner import _aggregate_retrieval

            chunks = _aggregate_retrieval(outcome)
            rl.record_example(ex.id, outcome.response, retrieved=chunks)
        # Persist the transcripts next to the metric artifacts —
        # this is the M6-defining provenance.
        transcripts_path = Path(rl.cfg.output_root) / run_id / "agent_transcripts.json"
        transcripts_path.write_text(
            json.dumps(
                [o.transcript.model_dump(mode="json") for o in summary.outcomes],
                indent=2,
            )
        )
    return output_root / run_id, summary


async def _run_fixed(
    *,
    model: str,
    examples: list[Any],
    output_root: Path,
    when: datetime,
    dataset_path: Path,
) -> tuple[Path, Any]:
    os.environ["ANVIL_LLM_BACKEND"] = "nvidia_nim"
    os.environ["ANVIL_LLM_MODEL"] = model
    pipeline = build_pipeline(ablation="baseline")
    runner = EvaluationRunner(pipeline.generator)

    run_id = (
        make_run_id(
            backend="nvidia_nim", model=model, ablation="baseline", when=when
        )
        + "__fixed"
    )
    cfg = RunLoggerConfig(
        run_id=run_id,
        backend="nvidia_nim",
        model=model,
        ablation="baseline-fixed",
        dataset_path=dataset_path,
        output_root=output_root,
    )
    log.info("fixed_eval.start", run_id=run_id)
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        summary = await runner.run(examples)
        rl.write_summary(summary)
        for ex, outcome in zip(examples, summary.outcomes, strict=True):
            rl.record_example(
                ex.id, outcome.response, retrieved=outcome.retrieved_chunks
            )
    return output_root / run_id, summary


def _render_table(
    fixed_summary: Any | None,
    agent_summary: AgentRunSummary,
    model: str,
) -> str:
    headers = [
        "configuration",
        "pass_rate",
        "calc_correctness",
        "citation_accuracy",
        "faithfulness",
        "retrieval_recall",
        "avg_tool_calls",
        "finalize_rate",
    ]
    rows: list[list[str]] = []

    def _row(label: str, summary: Any, agg_calls: float | None, agg_final: float | None) -> list[str]:
        a = summary.aggregate
        return [
            label,
            f"{summary.pass_rate:.3f}",
            f"{a.get('calculation_correctness', 0):.3f}",
            f"{a.get('citation_accuracy', 0):.3f}",
            f"{a.get('faithfulness', 0):.3f}",
            f"{a.get('retrieval_recall_at_k', 0):.3f}",
            f"{agg_calls:.2f}" if agg_calls is not None else "—",
            f"{agg_final:.3f}" if agg_final is not None else "—",
        ]

    if fixed_summary is not None:
        rows.append(_row(f"fixed / {model}", fixed_summary, None, None))
    rows.append(
        _row(
            f"agent / {model}",
            agent_summary,
            agent_summary.avg_tool_calls,
            agent_summary.finalize_rate,
        )
    )

    sep = "| " + " | ".join(":---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n"


async def _run() -> int:
    args = _parse_args()

    if not os.environ.get("NVIDIA_API_KEY"):
        sys.stderr.write(
            "NVIDIA_API_KEY is not set — agent eval requires a real LLM. "
            "Aborting.\n"
        )
        return 1

    dataset_path = ROOT / "tests" / "evaluation" / "golden_dataset.json"
    examples = load_golden_dataset(dataset_path)
    if args.limit:
        examples = examples[: args.limit]
    log.info(
        "agent_eval.config",
        model=args.model,
        n_examples=len(examples),
        max_steps=args.max_steps,
        sleep_between_examples=args.sleep_between_examples,
    )
    when = datetime.now(UTC)

    fixed_summary: Any | None = None
    if not args.skip_fixed:
        _, fixed_summary = await _run_fixed(
            model=args.model,
            examples=examples,
            output_root=args.output_root,
            when=when,
            dataset_path=dataset_path,
        )

    _, agent_summary = await _run_agent(
        model=args.model,
        max_steps=args.max_steps,
        sleep_between_examples=args.sleep_between_examples,
        examples=examples,
        output_root=args.output_root,
        when=when,
        dataset_path=dataset_path,
    )

    md = _render_table(fixed_summary, agent_summary, args.model)
    args.write_table.parent.mkdir(parents=True, exist_ok=True)
    args.write_table.write_text(
        f"# Agent vs. Fixed Pipeline — Headline Comparison\n\n"
        f"Model: `{args.model}`. Dataset: `{dataset_path.relative_to(ROOT)}` "
        f"({len(examples)} examples). Budget: max_steps={args.max_steps}.\n\n"
        f"{md}\n"
        f"_See `data/runs/` for full per-example artifacts and "
        f"`agent_transcripts.json` for the agent's tool-call traces._\n"
    )
    print(f"\nWrote {args.write_table}")
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
