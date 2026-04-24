"""Interactive demo of the anvil pipeline.

Shows the retrieval → calculation → generation → citation flow with Rich.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvil.pipeline import build_pipeline  # noqa: E402

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


DEFAULT_QUERIES = [
    "Calculate the minimum required wall thickness for a cylindrical shell with "
    "inside diameter 1800 mm, P=1.5 MPa, T=350°C, SM-516 Gr 70, Type 1 with full "
    "RT, CA=3.0 mm.",
    "What is the allowable stress of SM-516 Gr 70 at 300°C?",
    "What is the weather in San Jose today?",
]


async def _run_demo(queries: list[str]) -> None:
    pipeline = build_pipeline()
    console = Console() if _RICH else None

    for q in queries:
        if _RICH:
            console.rule(f"[bold cyan]Query")
            console.print(q)
        else:
            print(f"\n=== Query ===\n{q}")

        outcome = await pipeline.generator.generate(q, top_k=8)
        resp = outcome.response

        if _RICH:
            console.print(
                Panel(
                    f"[bold]confidence:[/bold] {resp.confidence}\n"
                    f"[bold]answer:[/bold] {resp.answer}\n"
                    f"[bold]refusal_reason:[/bold] {resp.refusal_reason}",
                    title="Response",
                    border_style="green",
                )
            )
            if resp.citations:
                t = Table(title="Citations", show_lines=True)
                t.add_column("#")
                t.add_column("Paragraph")
                t.add_column("Quote")
                for i, c in enumerate(resp.citations, 1):
                    t.add_row(str(i), c.paragraph_ref, c.quoted_text[:80])
                console.print(t)
            if resp.calculation_steps:
                t2 = Table(title="Calculation Steps", show_lines=True)
                t2.add_column("#")
                t2.add_column("Description")
                t2.add_column("Result")
                for s in resp.calculation_steps:
                    t2.add_row(str(s.step_number), s.description, f"{s.result} {s.unit}")
                console.print(t2)
        else:
            print(f"confidence={resp.confidence}  answer={resp.answer}")
            if resp.refusal_reason:
                print(f"refusal_reason={resp.refusal_reason}")


def main() -> None:
    queries = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUERIES
    asyncio.run(_run_demo(queries))


if __name__ == "__main__":
    main()
