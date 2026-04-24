"""Golden dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.schemas.evaluation import GoldenExample


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    raw = json.loads(Path(path).read_text())
    return [GoldenExample.model_validate(entry) for entry in raw]
