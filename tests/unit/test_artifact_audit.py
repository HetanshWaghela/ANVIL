from __future__ import annotations

from anvil.artifact_audit import audit_paths


def test_private_paths_are_rejected() -> None:
    findings = audit_paths(
        [
            "data/private/asme/asme_viii_private.md",
            "data/private_runs/2026-04-25/summary.sanitized.json",
            "data/asme_private_outputs/chunks.json",
        ]
    )

    assert {finding.path for finding in findings} == {
        "data/private/asme/asme_viii_private.md",
        "data/private_runs/2026-04-25/summary.sanitized.json",
        "data/asme_private_outputs/chunks.json",
    }


def test_raw_asme_artifacts_are_rejected() -> None:
    findings = audit_paths(
        [
            "data/runs/2026_fake_asme-private-v1_abl-baseline/raw_responses.jsonl",
            "data/parser_benchmark/asme_private/viii/output.json",
            "docs/private_asme.md",
            "data/parser_benchmark/public_pressure_sources.json",
        ]
    )

    assert [finding.path for finding in findings] == [
        "data/runs/2026_fake_asme-private-v1_abl-baseline/raw_responses.jsonl",
        "data/parser_benchmark/asme_private/viii/output.json",
    ]
