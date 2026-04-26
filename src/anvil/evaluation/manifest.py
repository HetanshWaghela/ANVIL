"""Run manifest — captures everything needed to reproduce a single eval run.

The manifest is the single source of truth that every committed artifact
(`summary.json`, `report.md`, `per_example.json`) is built against. A
reviewer who pulls the repo at the same git sha and re-runs with the
same env vars must produce a byte-identical `summary.json` (modulo
network latency for real backends).

Design choices:
* Pydantic so the manifest is JSON-serializable and self-validating.
* Secrets are never committed; the redactor replaces every API key with
  `<redacted:<sha256[:8]>>` so two runs that used the same key share
  the same redacted token (lets reviewers say "these two runs used the
  same credentials" without seeing the credential).
* Git sha + dirty flag are captured; a manifest produced on a dirty
  worktree carries `git_dirty=True` so a reviewer never mistakes a
  half-staged result for a clean reproduction.
* Dataset hash is captured from the loaded `GoldenExample` list so a
  silent edit to `golden_dataset.json` invalidates a regression
  comparison immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from anvil.schemas.evaluation import GoldenExample

# Env vars whose VALUES contain credentials and must be redacted.
_SECRET_VARS = {
    "NVIDIA_API_KEY",
    "NVIDIA_API_KEY_FALLBACKS",
    "NVIDIA_API_KEYS",
    "OPENAI_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "REDUCTO_API_KEY",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
}

# Env vars to capture verbatim in the manifest. Anything outside this
# allowlist is omitted to keep the manifest focused and to avoid leaking
# unrelated secrets the user may have in their shell.
_ALLOWLIST_ENV = {
    "ANVIL_LLM_BACKEND",
    "ANVIL_LLM_MODEL",
    "ANVIL_NIM_MODELS",
    "ANVIL_NIM_REASONING",
    "ANVIL_NIM_REASONING_EFFORT",
    "ANVIL_LLM_TIMEOUT_S",
    "ANVIL_LLM_MAX_TOKENS",
    "NVIDIA_NIM_BASE_URL",
    "OPENAI_COMPAT_BASE_URL",
    "ANVIL_EMBEDDER",
    "ANVIL_ST_MODEL",
    "ANVIL_ST_CACHE_DIR",
    "ANVIL_EMBEDDER_MODEL",
    "ANVIL_ENV",
} | _SECRET_VARS


def redact_key(value: str) -> str:
    """Hash-redact a secret value. Idempotent: same key → same token."""
    if not value:
        return "<unset>"
    return f"<redacted:{hashlib.sha256(value.encode()).hexdigest()[:8]}>"


def _git(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a git command, returning stripped stdout or None on failure."""
    try:
        out = subprocess.run(  # noqa: S603 — args are static
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _git_sha(cwd: Path | None = None) -> str | None:
    return _git(["rev-parse", "HEAD"], cwd=cwd)


def _git_dirty(cwd: Path | None = None) -> bool:
    """True if the worktree has any tracked uncommitted changes."""
    out = _git(["status", "--porcelain"], cwd=cwd)
    return bool(out)


def _git_branch(cwd: Path | None = None) -> str | None:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def _capture_env() -> dict[str, str]:
    """Snapshot the allowlisted env vars, redacting secret values."""
    snap: dict[str, str] = {}
    for name in sorted(_ALLOWLIST_ENV):
        val = os.environ.get(name)
        if val is None:
            continue
        snap[name] = redact_key(val) if name in _SECRET_VARS else val
    return snap


def dataset_hash(examples: list[GoldenExample]) -> str:
    """Stable hash over the canonical JSON form of the golden dataset.

    Two manifests with the same dataset_hash are guaranteed to have run
    on the same examples; a silent edit to `golden_dataset.json` flips
    the hash and downstream regression tools refuse to compare.
    """
    payload = json.dumps(
        [e.model_dump() for e in examples],
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class RunManifest(BaseModel):
    """Full manifest committed alongside every run's artifacts."""

    run_id: str
    started_at_utc: str
    finished_at_utc: str | None = None
    backend: str
    model: str | None = None
    base_url: str | None = None
    ablation: str = "baseline"
    ablation_config: dict[str, Any] = Field(default_factory=dict)
    dataset_path: str | None = None
    dataset_hash: str | None = None
    n_examples: int | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    env: dict[str, str] = Field(default_factory=_capture_env)
    notes: str | None = None

    def write_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def make_run_id(
    *,
    backend: str,
    model: str | None,
    ablation: str,
    dataset_version: str = "goldenv2-public100",
    when: datetime | None = None,
) -> str:
    """Deterministic run-id slug.

    Format: ``<UTC-iso>_<backend-slug>_<dataset-version>_<ablation-slug>``.
    The slug is the only thing used as a directory name, so it must be
    filesystem-safe across macOS / Linux / Windows.
    """
    when = when or datetime.now(UTC)
    iso = when.strftime("%Y-%m-%dT%H-%M-%SZ")
    backend_slug = (
        f"{backend}-{model.split('/')[-1]}" if model else backend
    ).replace("/", "-").replace(" ", "-")
    parts = [iso, backend_slug, dataset_version, f"abl-{ablation}"]
    return "_".join(p for p in parts if p)


def _relativize(path: Path, root: Path) -> str:
    """Return the path relative to `root` if possible, else absolute.

    Run manifests are committed to git and read by reviewers on
    different machines — leaking `/Users/<name>/…` into a committed
    file is both a tiny privacy leak and a reproducibility hazard.
    Rewrite to a repo-relative form whenever possible.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_manifest(
    *,
    run_id: str,
    backend: str,
    model: str | None = None,
    base_url: str | None = None,
    ablation: str = "baseline",
    ablation_config: dict[str, Any] | None = None,
    dataset_path: Path | None = None,
    examples: list[GoldenExample] | None = None,
    notes: str | None = None,
    repo_root: Path | None = None,
) -> RunManifest:
    """Assemble a `RunManifest` from the current repo + run config."""
    cwd = repo_root or Path.cwd()
    return RunManifest(
        run_id=run_id,
        started_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        backend=backend,
        model=model,
        base_url=base_url,
        ablation=ablation,
        ablation_config=ablation_config or {},
        dataset_path=_relativize(dataset_path, cwd) if dataset_path else None,
        dataset_hash=dataset_hash(examples) if examples else None,
        n_examples=len(examples) if examples else None,
        git_sha=_git_sha(cwd),
        git_branch=_git_branch(cwd),
        git_dirty=_git_dirty(cwd),
        notes=notes,
    )
