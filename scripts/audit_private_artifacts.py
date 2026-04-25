"""Fail if unsafe private standards artifacts are visible to git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.artifact_audit import audit_paths  # noqa: E402


def _git_visible_paths() -> list[str]:
    """Return tracked, staged, and untracked paths not excluded by .gitignore."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> None:
    findings = audit_paths(_git_visible_paths())
    if not findings:
        print("No unsafe private ASME artifacts visible to git.")
        return

    print("Unsafe private standards artifacts are visible to git:", file=sys.stderr)
    for finding in findings:
        print(f"- {finding.path}: {finding.reason}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
