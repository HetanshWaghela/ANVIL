"""Ingest the SPES-1 synthetic standard into KG + vector store.

Outputs:
  data/indexes/kg.json      — JSON-serialized knowledge graph
  data/indexes/elements.json — parsed DocumentElements
  data/indexes/vector.db    — SQLite vector store (if sqlite-vec available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Let this file be run directly without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.logging_config import get_logger  # noqa: E402
from anvil.pipeline import build_pipeline  # noqa: E402

log = get_logger("anvil.ingest")


def main() -> None:
    out_dir = ROOT / "data" / "indexes"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(standard_path=ROOT / "data" / "synthetic" / "standard.md")

    pipeline.graph_store.save(out_dir / "kg.json")
    (out_dir / "elements.json").write_text(
        json.dumps([e.model_dump() for e in pipeline.elements], indent=2, default=str)
    )

    log.info(
        "ingest.complete",
        num_elements=len(pipeline.elements),
        num_nodes=len(pipeline.graph_store.graph.nodes),
        num_edges=len(pipeline.graph_store.graph.edges),
        out_dir=str(out_dir),
    )
    print(f"Parsed {len(pipeline.elements)} elements.")
    print(
        f"Graph: {len(pipeline.graph_store.graph.nodes)} nodes, "
        f"{len(pipeline.graph_store.graph.edges)} edges."
    )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
