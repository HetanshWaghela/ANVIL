"""Tool registry for the M6 agentic loop.

Each tool is a thin adapter over an existing Anvil module:
- `retrieve_context` ──► `HybridRetriever.retrieve`
- `graph_lookup`     ──► `GraphStore.find_by_paragraph_ref` / `expand` / `neighbors`
- `pinned_lookup`    ──► `anvil.pinned.{get_material, get_allowable_stress, get_joint_efficiency}`
- `calculate`        ──► `CalculationEngine.calculate`

The adapters never raise. Every failure mode is captured in
`ToolResult.error` so the loop can continue (or refuse) gracefully.
This is the same fail-soft contract the rest of the pipeline uses.

`TOOL_DESCRIPTIONS` is the single source of truth for what gets
exposed to the LLM — both the JSON-shaped manifest used by
instructor / OpenAI-style tool calling AND the human-readable list
the run-logger persists. Keep them in sync by editing only the dict
below.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from anvil import CalculationError
from anvil.generation.calculation_engine import (
    CalculationEngine,
    CalculationInputs,
)
from anvil.knowledge.graph_store import GraphStore
from anvil.logging_config import get_logger
from anvil.pinned import (
    get_allowable_stress,
    get_joint_efficiency,
    get_material,
)
from anvil.retrieval.hybrid_retriever import HybridRetriever
from anvil.schemas.agent import ToolCall, ToolResult
from anvil.schemas.retrieval import RetrievalQuery

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool descriptors — exposed to the LLM and to the docs/report writer.
# ---------------------------------------------------------------------------


TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "retrieve_context": {
        "description": (
            "Retrieve the most relevant passages from the parsed standard "
            "for a natural-language query. Returns a list of chunks with "
            "paragraph_ref, element_type, content, and a similarity score."
        ),
        "parameters": {
            "query": "str — natural-language query (required).",
            "top_k": "int — number of chunks to return (default 5, max 20).",
        },
    },
    "graph_lookup": {
        "description": (
            "Walk the typed knowledge graph from a known paragraph "
            "reference. Returns the seed node plus all nodes within "
            "`max_hops` and their edges. Useful for chasing cross-refs "
            "(e.g. given UG-27, expand to UW-12 and any rule that "
            "references UG-27)."
        ),
        "parameters": {
            "paragraph_ref": "str — canonical paragraph ref (e.g. 'A-27(c)(1)').",
            "max_hops": "int — graph traversal depth (default 1, max 3).",
        },
    },
    "pinned_lookup": {
        "description": (
            "Query the trusted ground-truth tables (materials, allowable "
            "stresses by temperature, joint efficiencies). NEVER infer "
            "these from retrieval — always use this tool. Returns the "
            "raw record + its source citation."
        ),
        "parameters": {
            "kind": "str — 'material' | 'allowable_stress' | 'joint_efficiency'.",
            "key": (
                "str — material spec (e.g. 'SM-516 Gr 70') for material/"
                "allowable_stress kinds; joint type label "
                "(e.g. 'Type 1') for joint_efficiency."
            ),
            "temp_c": (
                "float — required only for kind='allowable_stress'."
            ),
            "rt_level": (
                "str — required only for kind='joint_efficiency' "
                "(e.g. 'Full RT', 'Spot RT', 'No RT')."
            ),
        },
    },
    "calculate": {
        "description": (
            "Run the deterministic thickness + MAWP calculation engine. "
            "Validates pressure/material/temperature applicability, "
            "returns t_min, t_design, t_nominal, and MAWP with full "
            "citation-bearing steps. Refuses with a structured error "
            "if inputs are out of range — never silently extrapolates."
        ),
        "parameters": {
            "component": (
                "str — 'cylindrical_shell' (inside-radius A-27(c)(1)) | "
                "'cylindrical_shell_outside_radius' (A-27(c)(2)) | "
                "'spherical_shell' (A-27(d))."
            ),
            "P_mpa": "float — design pressure (MPa).",
            "design_temp_c": "float — design temperature (°C).",
            "inside_diameter_mm": "float — required for inside-radius components.",
            "outside_diameter_mm": "float — required for outside-radius components.",
            "material": "str — material spec (e.g. 'SM-516 Gr 70').",
            "joint_type": "str — joint type label (e.g. 'Type 1').",
            "rt_level": "str — radiography level ('Full RT' etc.).",
            "corrosion_allowance_mm": "float — corrosion allowance (mm).",
        },
    },
    "finalize": {
        "description": (
            "Terminate the agent loop with a final structured "
            "AnvilResponse. The agent MUST eventually call this — if "
            "the budget runs out first, the loop synthesizes a refusal."
        ),
        "parameters": {
            "response": "AnvilResponse — the final structured answer.",
        },
    },
}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


# The signature: (call.arguments) -> output dict. Tools must NOT raise.
ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


def _coerce_joint_type(raw: Any) -> int:
    """Coerce LLM-shaped joint type ('Type 1', '1', 1, 'type 1') to int.

    The pinned table is keyed by `int` (1..6). The LLM/agent layer
    naturally produces strings — having every adapter call duplicate
    the parsing logic would be a footgun. Centralize here so a
    regression test can lock the contract.
    """
    if isinstance(raw, bool):  # `bool` is a subclass of int — reject it
        raise ValueError(f"joint_type cannot be bool ({raw!r}).")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s.startswith("type"):
            s = s[len("type") :].strip()
        try:
            return int(s)
        except ValueError as exc:
            raise ValueError(
                f"Cannot coerce joint_type={raw!r} to int. "
                f"Expected an int (1–6) or a label like 'Type 1'."
            ) from exc
    raise ValueError(
        f"joint_type must be int or str (e.g. 'Type 1'); got "
        f"{type(raw).__name__}."
    )


class ToolRegistry:
    """Maps tool names to their executable adapters.

    The registry holds references to the live retriever / graph store /
    calc engine — everything is resolved at construction time so the
    per-call dispatch is a pure dict lookup.

    `finalize` is registered as a no-op here: the agent loop intercepts
    it before dispatch (it changes control flow, not state) but
    listing it keeps the public manifest complete.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        graph_store: GraphStore,
        calc_engine: CalculationEngine,
    ) -> None:
        self.retriever = retriever
        self.graph_store = graph_store
        self.calc_engine = calc_engine
        self._registry: dict[str, ToolFn] = {
            "retrieve_context": self._retrieve_context,
            "graph_lookup": self._graph_lookup,
            "pinned_lookup": self._pinned_lookup,
            "calculate": self._calculate,
            # finalize has no executor — the agent loop intercepts it.
        }

    # ---- public API -------------------------------------------------------

    def names(self) -> list[str]:
        """All callable tool names (excludes `finalize`)."""
        return list(self._registry)

    def manifest(self) -> dict[str, dict[str, Any]]:
        """Tool manifest in JSON-friendly shape; includes `finalize`."""
        return {k: dict(v) for k, v in TOOL_DESCRIPTIONS.items()}

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, returning a `ToolResult`. Never raises."""
        t0 = time.perf_counter()
        fn = self._registry.get(call.name)
        if fn is None:
            err = (
                f"Unknown tool {call.name!r}. Available: "
                f"{sorted(self._registry)}."
            )
            log.warning("agent.tool.unknown", tool=call.name)
            return ToolResult(
                name=call.name,
                arguments=call.arguments,
                error=err,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        try:
            output = fn(call.arguments)
            return ToolResult(
                name=call.name,
                arguments=call.arguments,
                output=output,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — adapter must not raise
            log.warning("agent.tool.error", tool=call.name, error=repr(exc))
            return ToolResult(
                name=call.name,
                arguments=call.arguments,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    # ---- adapters ---------------------------------------------------------

    def _retrieve_context(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("retrieve_context: 'query' is required and non-empty.")
        top_k = int(args.get("top_k") or 5)
        top_k = max(1, min(top_k, 20))
        chunks = self.retriever.retrieve(
            RetrievalQuery(text=query, top_k=top_k, enable_graph_expansion=True)
        )
        return {
            "n": len(chunks),
            "chunks": [
                {
                    "element_id": c.element_id,
                    "paragraph_ref": c.paragraph_ref,
                    "element_type": c.element_type,
                    "page_number": c.page_number,
                    "score": round(c.score, 4),
                    "retrieval_source": c.retrieval_source,
                    "content": c.content,
                }
                for c in chunks
            ],
        }

    def _graph_lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        ref = str(args.get("paragraph_ref") or "").strip()
        if not ref:
            raise ValueError("graph_lookup: 'paragraph_ref' is required.")
        max_hops = int(args.get("max_hops") or 1)
        max_hops = max(1, min(max_hops, 3))
        seeds = self.graph_store.find_by_paragraph_ref(ref)
        if not seeds:
            return {"seeds": [], "expanded": [], "edges": []}
        expanded = sorted(self.graph_store.expand(seeds, max_hops=max_hops))
        edges: list[dict[str, str]] = []
        for n in seeds:
            for tgt, edge_type, ref_text in self.graph_store.neighbors(n):
                edges.append(
                    {
                        "from": n,
                        "to": tgt,
                        "edge_type": edge_type,
                        "reference_text": ref_text,
                    }
                )
        return {
            "seeds": seeds,
            "expanded": expanded,
            "edges": edges,
            "max_hops": max_hops,
        }

    def _pinned_lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        kind = str(args.get("kind") or "").strip().lower()
        key = str(args.get("key") or "").strip()
        if not key:
            raise ValueError("pinned_lookup: 'key' is required.")
        if kind == "material":
            mat = get_material(key)
            if mat is None:
                return {"found": False, "kind": kind, "key": key}
            return {
                "found": True,
                "kind": kind,
                "key": key,
                "spec_grade": mat.key,
                "max_temp_c": mat.max_temp_c,
                "tabulated_temps_c": sorted(mat.stress_by_temp_c),
                "source": "Pinned table M-1 (materials).",
            }
        if kind == "allowable_stress":
            temp_c = float(args.get("temp_c") or 0.0)
            S = get_allowable_stress(key, temp_c)
            return {
                "found": S is not None,
                "kind": kind,
                "key": key,
                "temp_c": temp_c,
                "allowable_stress_mpa": S,
                "source": "Pinned table M-1 (allowable stress).",
            }
        if kind == "joint_efficiency":
            rt = str(args.get("rt_level") or "").strip()
            if not rt:
                raise ValueError(
                    "pinned_lookup[joint_efficiency]: 'rt_level' is required."
                )
            joint_type_int = _coerce_joint_type(key)
            E = get_joint_efficiency(joint_type_int, rt)
            return {
                "found": E is not None,
                "kind": kind,
                "key": key,
                "joint_type": joint_type_int,
                "rt_level": rt,
                "joint_efficiency": E,
                "source": "Pinned table B-12 (joint efficiencies).",
            }
        raise ValueError(
            f"pinned_lookup: unknown kind={kind!r}. "
            "Expected: material | allowable_stress | joint_efficiency."
        )

    def _calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        # Build CalculationInputs from kwargs; let Pydantic validate.
        try:
            inputs = CalculationInputs(
                component=args["component"],
                P_mpa=float(args["P_mpa"]),
                design_temp_c=float(args["design_temp_c"]),
                inside_diameter_mm=(
                    float(args["inside_diameter_mm"])
                    if args.get("inside_diameter_mm") is not None
                    else None
                ),
                outside_diameter_mm=(
                    float(args["outside_diameter_mm"])
                    if args.get("outside_diameter_mm") is not None
                    else None
                ),
                material=str(args["material"]),
                joint_type=_coerce_joint_type(args["joint_type"]),
                rt_level=str(args["rt_level"]),
                corrosion_allowance_mm=float(
                    args.get("corrosion_allowance_mm") or 0.0
                ),
            )
        except KeyError as exc:
            raise ValueError(f"calculate: missing required kwarg {exc}.") from exc

        try:
            result = self.calc_engine.calculate(inputs)
        except CalculationError as exc:
            # Engine refusal — surface as an error so the agent can
            # decide to refuse vs. retry with different inputs.
            return {
                "ok": False,
                "error": f"CalculationError: {exc}",
            }
        return {
            "ok": True,
            "S_mpa": result.S_mpa,
            "E": result.E,
            "R_mm": result.R_mm,
            "formula_ref": result.formula_ref,
            "t_min_mm": result.t_min_mm,
            "t_design_mm": result.t_design_mm,
            "t_nominal_mm": result.t_nominal_mm,
            "mawp_mpa": result.mawp_mpa,
            "n_steps": len(result.steps),
            "applicability_ok": result.applicability_ok,
        }


__all__ = ["TOOL_DESCRIPTIONS", "ToolRegistry"]
