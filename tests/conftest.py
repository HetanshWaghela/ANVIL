"""Shared fixtures for the anvil test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.pipeline import Pipeline, build_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO_ROOT / "data" / "synthetic"
STANDARD_MD = SYNTH_DIR / "standard.md"
DESIGN_EXAMPLES_JSON = SYNTH_DIR / "design_examples.json"
GOLDEN_DATASET_JSON = REPO_ROOT / "tests" / "evaluation" / "golden_dataset.json"


@pytest.fixture(scope="session")
def standard_md_path() -> Path:
    return STANDARD_MD


@pytest.fixture(scope="session")
def design_examples() -> list[dict]:
    return json.loads(DESIGN_EXAMPLES_JSON.read_text())


@pytest.fixture(scope="session")
def parsed_elements():
    return parse_markdown_standard(STANDARD_MD)


@pytest.fixture(scope="session")
def pipeline() -> Pipeline:
    return build_pipeline(standard_path=STANDARD_MD)
