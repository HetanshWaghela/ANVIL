"""Safety checks for private standards artifacts.

The public repo may contain synthetic SPES-1 data and public NASA sources. It
must not contain licensed ASME inputs or derived raw artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PRIVATE_PREFIXES = (
    "data/private/",
    "data/private_runs/",
    "data/asme_private_outputs/",
    "data/parser_benchmark/private/",
    "data/parser_benchmark/asme_private/",
)
UNSAFE_RAW_FILENAMES = {
    "agent_transcripts.json",
    "raw_responses.jsonl",
    "prompts.jsonl",
    "request_log.jsonl",
}


@dataclass(frozen=True)
class ArtifactFinding:
    path: str
    reason: str


def normalize_repo_path(path: str | Path) -> str:
    """Return a stable POSIX-style repo path for matching."""
    return Path(path).as_posix().lstrip("./")


def audit_paths(paths: list[str | Path]) -> list[ArtifactFinding]:
    """Return unsafe artifact findings for repo-relative paths."""
    findings: list[ArtifactFinding] = []
    for path in paths:
        normalized = normalize_repo_path(path)
        lowered = normalized.lower()
        name = Path(normalized).name.lower()

        if any(lowered.startswith(prefix) for prefix in PRIVATE_PREFIXES):
            findings.append(
                ArtifactFinding(
                    path=normalized,
                    reason="private licensed-data path must remain gitignored",
                )
            )
            continue

        if lowered.startswith("data/runs/") and "asme-private" in lowered:
            findings.append(
                ArtifactFinding(
                    path=normalized,
                    reason="private ASME run artifacts must stay under data/private_runs/",
                )
            )
            continue

        if name in UNSAFE_RAW_FILENAMES and (
            "asme" in lowered or "private" in lowered
        ):
            findings.append(
                ArtifactFinding(
                    path=normalized,
                    reason="raw private responses/prompts/transcripts can contain source text",
                )
            )
            continue

        if (
            lowered.startswith("data/parser_benchmark/")
            and "asme" in lowered
            and name == "output.json"
        ):
            findings.append(
                ArtifactFinding(
                    path=normalized,
                    reason="private parser output can contain extracted source text",
                )
            )

    return findings
