"""Run-logger — context manager that owns one run's artifact directory.

Layout per the plan §3.1:

    data/runs/<run_id>/
      manifest.json        # COMMITTED — run config + git sha + dataset hash
      summary.json         # COMMITTED — aggregate metrics + pass_rate
      report.md            # COMMITTED — human-readable summary
      per_example.json     # COMMITTED — per-example metric vector
      raw_responses.jsonl  # gitignored — full AnvilResponse per example
      prompts.jsonl        # gitignored — system + user prompts sent
      request_log.jsonl    # gitignored — raw NIM/LLM request/response pairs

Usage::

    cfg = RunLoggerConfig(
        run_id=make_run_id(backend="fake", model=None, ablation="baseline"),
        backend="fake",
        ablation="baseline",
        dataset_path=DATASET_PATH,
    )
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        for ex in examples:
            outcome = await generator.generate(ex.query)
            rl.record_request(...)
            rl.record_example(ex.id, outcome.response, metrics, retrieved=...)
        rl.write_summary(summary)

The `__aexit__` finalizes by writing `manifest.json`, `summary.json`,
`per_example.json`, and `report.md`. If `write_summary` was never
called the run is still committed with `summary.json` containing
`{"status": "incomplete"}` so a crashed run can't silently disappear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from structlog.contextvars import bind_contextvars, unbind_contextvars

from anvil.evaluation.manifest import RunManifest, build_manifest, make_run_id
from anvil.evaluation.report_writer import write_report
from anvil.evaluation.runner import RunSummary
from anvil.logging_config import get_logger
from anvil.schemas.evaluation import GoldenExample
from anvil.schemas.generation import AnvilResponse
from anvil.schemas.retrieval import RetrievedChunk

log = get_logger(__name__)


@dataclass
class RunLoggerConfig:
    """All inputs the run-logger needs to create + finalize a run dir."""

    run_id: str
    backend: str
    ablation: str = "baseline"
    model: str | None = None
    base_url: str | None = None
    ablation_config: dict[str, Any] = field(default_factory=dict)
    dataset_path: Path | None = None
    notes: str | None = None
    output_root: Path = field(default_factory=lambda: Path("data/runs"))


class RunLogger:
    """Async context manager owning one run's artifact directory."""

    # Filenames (centralized so tests can assert on them).
    MANIFEST = "manifest.json"
    SUMMARY = "summary.json"
    REPORT = "report.md"
    PER_EXAMPLE = "per_example.json"
    RAW_RESPONSES = "raw_responses.jsonl"
    PROMPTS = "prompts.jsonl"
    REQUEST_LOG = "request_log.jsonl"

    def __init__(self, cfg: RunLoggerConfig) -> None:
        self.cfg = cfg
        self.run_dir = cfg.output_root / cfg.run_id
        self._examples: list[GoldenExample] = []
        self._summary: RunSummary | None = None
        self._manifest: RunManifest | None = None
        # File handles opened lazily on first record_*() call so tests
        # that never write raw data don't leave empty files behind.
        self._raw_fh: Any = None
        self._prompts_fh: Any = None
        self._request_fh: Any = None
        self._bound = False

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    async def __aenter__(self) -> RunLogger:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Bind the run_id into structlog's contextvars so every log line
        # produced inside this block carries `run_id=<id>`. This is what
        # makes a single run's trace reconstructable from the structured
        # log stream.
        bind_contextvars(run_id=self.cfg.run_id)
        self._bound = True
        log.info("run.start", run_dir=str(self.run_dir), backend=self.cfg.backend)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self._finalize()
        finally:
            for fh in (self._raw_fh, self._prompts_fh, self._request_fh):
                if fh is not None:
                    fh.close()
            self._raw_fh = self._prompts_fh = self._request_fh = None
            if self._bound:
                unbind_contextvars("run_id")
                self._bound = False
            if exc is not None:
                log.warning(
                    "run.aborted",
                    exc_type=getattr(exc_type, "__name__", "?"),
                    msg=str(exc),
                )
            else:
                log.info("run.done", pass_rate=self._summary.pass_rate if self._summary else None)

    # ------------------------------------------------------------------
    # Public API used inside the block
    # ------------------------------------------------------------------
    def attach_examples(self, examples: list[GoldenExample]) -> None:
        """Record the dataset; updates the manifest's `dataset_hash`."""
        self._examples = list(examples)
        log.info("run.dataset_attached", n_examples=len(self._examples))

    def write_summary(self, summary: RunSummary) -> None:
        """Hand the summary to the logger; finalize will pick it up."""
        self._summary = summary

    def record_example(
        self,
        example_id: str,
        response: AnvilResponse,
        retrieved: list[RetrievedChunk] | None = None,
    ) -> None:
        """Append one full AnvilResponse to `raw_responses.jsonl`.

        The run-logger never modifies a response — it only persists what
        the runner already produced. `retrieved` is recorded too so the
        raw-response file is sufficient to reconstruct what the LLM saw
        without re-running retrieval.
        """
        if self._raw_fh is None:
            self._raw_fh = (self.run_dir / self.RAW_RESPONSES).open("w")
        rec = {
            "example_id": example_id,
            "response": response.model_dump(),
            "retrieved": [c.model_dump() for c in (retrieved or [])],
            "ts_utc": datetime.now(UTC).isoformat(),
        }
        self._raw_fh.write(json.dumps(rec, default=str) + "\n")
        self._raw_fh.flush()

    def record_prompt(
        self,
        example_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if self._prompts_fh is None:
            self._prompts_fh = (self.run_dir / self.PROMPTS).open("w")
        rec = {
            "example_id": example_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "ts_utc": datetime.now(UTC).isoformat(),
        }
        self._prompts_fh.write(json.dumps(rec) + "\n")
        self._prompts_fh.flush()

    def record_request(
        self,
        *,
        kind: str,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one raw LLM/HTTP request/response pair.

        `kind` namespaces the line (`llm.request`, `nim.health`, etc.)
        so a downstream `jq 'select(.kind=="llm.request")'` works.
        """
        if self._request_fh is None:
            self._request_fh = (self.run_dir / self.REQUEST_LOG).open("w")
        rec: dict[str, Any] = {
            "kind": kind,
            "ts_utc": datetime.now(UTC).isoformat(),
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "error": error,
        }
        if extra:
            rec.update(extra)
        self._request_fh.write(json.dumps(rec, default=str) + "\n")
        self._request_fh.flush()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        """Write the four committed artifacts. Idempotent."""
        self._manifest = build_manifest(
            run_id=self.cfg.run_id,
            backend=self.cfg.backend,
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            ablation=self.cfg.ablation,
            ablation_config=self.cfg.ablation_config,
            dataset_path=self.cfg.dataset_path,
            examples=self._examples or None,
            notes=self.cfg.notes,
        )
        self._manifest.finished_at_utc = datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._manifest.write_to(self.run_dir / self.MANIFEST)

        if self._summary is None:
            (self.run_dir / self.SUMMARY).write_text(
                json.dumps({"status": "incomplete"}, indent=2)
            )
            (self.run_dir / self.PER_EXAMPLE).write_text(
                json.dumps([], indent=2)
            )
            return

        # summary.json — run-level rollup. Includes the manifest at the
        # top level so a downstream tool can read one file and have the
        # full reproducibility envelope.
        summary_payload = {
            "run_id": self.cfg.run_id,
            "manifest": self._manifest.model_dump(),
            "aggregate": self._summary.aggregate,
            "pass_rate": self._summary.pass_rate,
            "n_examples": len(self._summary.per_example),
            "n_passed": sum(1 for r in self._summary.per_example if r.passed),
        }
        (self.run_dir / self.SUMMARY).write_text(
            json.dumps(summary_payload, indent=2, default=str)
        )

        # per_example.json — small enough to commit, large enough to drive
        # ablation comparisons.
        (self.run_dir / self.PER_EXAMPLE).write_text(
            json.dumps(
                [r.model_dump() for r in self._summary.per_example],
                indent=2,
                default=str,
            )
        )

        # report.md — what a human reads first.
        write_report(
            self._summary,
            self._manifest,
            self.run_dir / self.REPORT,
        )


def make_default_logger(
    *,
    backend: str,
    ablation: str = "baseline",
    model: str | None = None,
    dataset_path: Path | None = None,
    output_root: Path | None = None,
    when: datetime | None = None,
) -> RunLogger:
    """Convenience builder used by `scripts/evaluate.py` and friends."""
    run_id = make_run_id(
        backend=backend, model=model, ablation=ablation, when=when
    )
    cfg = RunLoggerConfig(
        run_id=run_id,
        backend=backend,
        model=model,
        ablation=ablation,
        dataset_path=dataset_path,
        output_root=output_root or Path("data/runs"),
    )
    return RunLogger(cfg)
