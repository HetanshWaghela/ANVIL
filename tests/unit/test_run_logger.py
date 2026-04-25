"""M2 — Run-logger + manifest + report-writer tests.

Locks in the artifact-directory contract from plan §3.1: the four
committed files (manifest.json, summary.json, report.md, per_example.json)
are populated, and secrets never leak into a manifest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anvil.evaluation.dataset import load_golden_dataset
from anvil.evaluation.manifest import (
    build_manifest,
    dataset_hash,
    make_run_id,
    redact_key,
)
from anvil.evaluation.report_writer import render_report
from anvil.evaluation.run_logger import RunLogger, RunLoggerConfig
from anvil.evaluation.runner import RunSummary
from anvil.schemas.evaluation import (
    EvaluationResult,
    GoldenExample,
    MetricScore,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "tests" / "evaluation" / "golden_dataset.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_summary() -> RunSummary:
    """Two examples — one passing, one failing — covering both branches
    of the `_failing_examples` rendering path."""
    pass_metrics = [
        MetricScore(name="faithfulness", value=0.95, passed=True, threshold=0.8),
        MetricScore(name="citation_accuracy", value=1.00, passed=True, threshold=0.8),
        MetricScore(
            name="calculation_correctness",
            value=1.00,
            passed=True,
            threshold=1.0,
        ),
    ]
    fail_metrics = [
        MetricScore(name="faithfulness", value=0.40, passed=False, threshold=0.8),
        MetricScore(name="citation_accuracy", value=0.60, passed=False, threshold=0.8),
    ]
    per = [
        EvaluationResult(
            example_id="ex-1",
            category="calculation",
            metrics=pass_metrics,
            passed=True,
        ),
        EvaluationResult(
            example_id="ex-2",
            category="lookup",
            metrics=fail_metrics,
            passed=False,
        ),
    ]
    aggregate = {
        "faithfulness": (0.95 + 0.40) / 2,
        "citation_accuracy": (1.00 + 0.60) / 2,
        "calculation_correctness": 1.00,
    }
    return RunSummary(per_example=per, aggregate=aggregate, pass_rate=0.5)


def _toy_examples() -> list[GoldenExample]:
    """A pair of trivial GoldenExamples — used only for dataset_hash
    coverage; their content does not need to be realistic here."""
    return [
        GoldenExample(
            id="ex-1",
            category="lookup",
            query="what is X?",
            expected_paragraph_refs=["A-23"],
        ),
        GoldenExample(
            id="ex-2",
            category="out_of_domain",
            query="weather?",
            expected_refusal=True,
        ),
    ]


# ---------------------------------------------------------------------------
# manifest.py
# ---------------------------------------------------------------------------


def test_redact_key_is_idempotent_and_short() -> None:
    """REGRESSION: redact_key MUST be deterministic — two runs that used
    the same key must share the same redacted token, but the token must
    NEVER allow the key to be recovered."""
    a = redact_key("nvapi-secret-1234")
    b = redact_key("nvapi-secret-1234")
    c = redact_key("nvapi-secret-9999")
    assert a == b
    assert a != c
    assert "1234" not in a
    assert a.startswith("<redacted:")
    assert len(a) <= 22  # `<redacted:` + 8 hex chars + `>` = 20


def test_redact_key_handles_empty() -> None:
    assert redact_key("") == "<unset>"


def test_make_run_id_is_filesystem_safe_and_deterministic() -> None:
    """The slug becomes a directory name — must contain no `/` or
    spaces, and must be reproducible from the same inputs."""
    when = datetime(2026, 4, 25, 12, 34, 56, tzinfo=UTC)
    rid = make_run_id(
        backend="nvidia_nim",
        model="meta/llama-3.3-70b-instruct",
        ablation="baseline",
        when=when,
    )
    assert "/" not in rid
    assert " " not in rid
    assert rid.startswith("2026-04-25T12-34-56Z")
    assert "llama-3.3-70b-instruct" in rid
    assert "abl-baseline" in rid

    rid2 = make_run_id(
        backend="nvidia_nim",
        model="meta/llama-3.3-70b-instruct",
        ablation="baseline",
        when=when,
    )
    assert rid == rid2


def test_dataset_hash_changes_when_dataset_changes() -> None:
    """A silent edit to the golden dataset MUST flip the hash so
    downstream regression tools refuse stale comparisons."""
    a = dataset_hash(_toy_examples())
    examples = _toy_examples()
    examples[0].query = "different question"
    b = dataset_hash(examples)
    assert a != b


def test_build_manifest_redacts_env_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An NVIDIA_API_KEY in the env must NEVER be written verbatim into
    the manifest."""
    secret = "nvapi-must-not-be-committed-1234567890"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    monkeypatch.setenv("ANVIL_LLM_MODEL", "meta/llama-3.3-70b-instruct")
    m = build_manifest(
        run_id="t-run",
        backend="nvidia_nim",
        model="meta/llama-3.3-70b-instruct",
    )
    payload = m.model_dump_json()
    assert secret not in payload, "API key leaked into manifest"
    assert "<redacted:" in payload
    # Non-secret allowlisted env is captured verbatim.
    assert m.env.get("ANVIL_LLM_MODEL") == "meta/llama-3.3-70b-instruct"


def test_build_manifest_captures_embedder_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANVIL_EMBEDDER", "sentence_transformer")
    monkeypatch.setenv("ANVIL_ST_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("ANVIL_ST_CACHE_DIR", "/tmp/anvil-st")

    m = build_manifest(run_id="t-embedder-env", backend="fake")

    assert m.env["ANVIL_EMBEDDER"] == "sentence_transformer"
    assert m.env["ANVIL_ST_MODEL"] == "BAAI/bge-small-en-v1.5"
    assert m.env["ANVIL_ST_CACHE_DIR"] == "/tmp/anvil-st"


def test_build_manifest_attaches_dataset_hash() -> None:
    examples = _toy_examples()
    m = build_manifest(
        run_id="t-run",
        backend="fake",
        examples=examples,
    )
    assert m.dataset_hash == dataset_hash(examples)
    assert m.n_examples == len(examples)


# ---------------------------------------------------------------------------
# report_writer.py
# ---------------------------------------------------------------------------


def test_render_report_includes_pass_rate_and_metric_table() -> None:
    summary = _toy_summary()
    manifest = build_manifest(run_id="t-render", backend="fake")
    md = render_report(summary, manifest)
    assert "Pass rate: `50.0%`" in md or "Pass rate: `50.00%`" in md
    assert "faithfulness" in md
    assert "citation_accuracy" in md
    assert "ex-2" in md, "failing example must be surfaced"
    assert "ex-1" not in md or md.find("ex-2") < md.find("Reproducibility")


def test_render_report_handles_all_passing_summary() -> None:
    summary = RunSummary(
        per_example=[
            EvaluationResult(
                example_id="ex-ok",
                category="calculation",
                metrics=[
                    MetricScore(name="faithfulness", value=1.0, passed=True, threshold=0.8)
                ],
                passed=True,
            ),
        ],
        aggregate={"faithfulness": 1.0},
        pass_rate=1.0,
    )
    md = render_report(summary, build_manifest(run_id="t-render", backend="fake"))
    assert "every example passed" in md.lower()


# ---------------------------------------------------------------------------
# run_logger.py — full directory contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_logger_writes_all_committed_artifacts(tmp_path: Path) -> None:
    """The four committed files must exist after a successful run."""
    cfg = RunLoggerConfig(
        run_id="2026-04-25T00-00-00Z_fake_goldenv1_abl-test",
        backend="fake",
        ablation="test",
        output_root=tmp_path,
    )
    summary = _toy_summary()
    examples = _toy_examples()

    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        rl.write_summary(summary)

    rdir = tmp_path / cfg.run_id
    assert (rdir / "manifest.json").exists()
    assert (rdir / "summary.json").exists()
    assert (rdir / "report.md").exists()
    assert (rdir / "per_example.json").exists()

    summary_payload = json.loads((rdir / "summary.json").read_text())
    assert summary_payload["run_id"] == cfg.run_id
    assert summary_payload["pass_rate"] == 0.5
    assert summary_payload["aggregate"]["faithfulness"] > 0
    assert summary_payload["manifest"]["backend"] == "fake"

    per_example = json.loads((rdir / "per_example.json").read_text())
    assert isinstance(per_example, list) and len(per_example) == 2
    assert per_example[0]["example_id"] == "ex-1"


@pytest.mark.asyncio
async def test_run_logger_records_raw_responses_when_called(tmp_path: Path) -> None:
    """`raw_responses.jsonl` must be created exactly when
    `record_example` is called, with one JSONL line per call."""
    cfg = RunLoggerConfig(
        run_id="t-raw",
        backend="fake",
        output_root=tmp_path,
    )
    examples = _toy_examples()
    summary = _toy_summary()
    from anvil.schemas.generation import AnvilResponse, ResponseConfidence

    fake_response = AnvilResponse(
        query="q",
        answer="refused",
        confidence=ResponseConfidence.INSUFFICIENT,
        refusal_reason="test fixture",
    )
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        rl.record_example("ex-1", fake_response)
        rl.record_example("ex-2", fake_response)
        rl.write_summary(summary)
    raw_path = tmp_path / cfg.run_id / "raw_responses.jsonl"
    assert raw_path.exists()
    lines = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(lines) == 2
    assert {line["example_id"] for line in lines} == {"ex-1", "ex-2"}


@pytest.mark.asyncio
async def test_run_logger_does_not_create_jsonl_files_if_unused(
    tmp_path: Path,
) -> None:
    """Lazy file-handle creation: a run that never calls record_*()
    must not leave empty .jsonl files behind to confuse downstream
    diffing tools."""
    cfg = RunLoggerConfig(run_id="t-nojsonl", backend="fake", output_root=tmp_path)
    summary = _toy_summary()
    async with RunLogger(cfg) as rl:
        rl.write_summary(summary)
    rdir = tmp_path / cfg.run_id
    assert not (rdir / "raw_responses.jsonl").exists()
    assert not (rdir / "prompts.jsonl").exists()
    assert not (rdir / "request_log.jsonl").exists()


@pytest.mark.asyncio
async def test_run_logger_marks_incomplete_runs(tmp_path: Path) -> None:
    """If `write_summary` is never called, summary.json carries
    `status=incomplete` so a crashed run can't silently disappear."""
    cfg = RunLoggerConfig(run_id="t-crashed", backend="fake", output_root=tmp_path)
    async with RunLogger(cfg) as rl:
        rl.attach_examples(_toy_examples())
        # No write_summary — simulate a crash before completion.
    payload = json.loads((tmp_path / cfg.run_id / "summary.json").read_text())
    assert payload == {"status": "incomplete"}


@pytest.mark.asyncio
async def test_run_logger_redacts_secrets_in_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: run-logger writing a manifest while NVIDIA_API_KEY is
    set must NEVER persist the raw key on disk."""
    secret = "nvapi-disk-must-not-leak-1234567890"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    cfg = RunLoggerConfig(run_id="t-redact", backend="fake", output_root=tmp_path)
    async with RunLogger(cfg) as rl:
        rl.write_summary(_toy_summary())
    manifest_text = (tmp_path / cfg.run_id / "manifest.json").read_text()
    summary_text = (tmp_path / cfg.run_id / "summary.json").read_text()
    assert secret not in manifest_text, "API key leaked into manifest.json"
    assert secret not in summary_text, "API key leaked into summary.json"


@pytest.mark.asyncio
async def test_run_logger_carries_dataset_hash_when_examples_attached(
    tmp_path: Path,
) -> None:
    cfg = RunLoggerConfig(run_id="t-dshash", backend="fake", output_root=tmp_path)
    examples = load_golden_dataset(DATASET_PATH)
    async with RunLogger(cfg) as rl:
        rl.attach_examples(examples)
        rl.write_summary(_toy_summary())
    manifest = json.loads((tmp_path / cfg.run_id / "manifest.json").read_text())
    assert manifest["dataset_hash"]
    assert manifest["n_examples"] == len(examples)
