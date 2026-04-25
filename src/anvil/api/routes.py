"""API routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from anvil.generation.calculation_engine import (
    CalculationEngine,
    CalculationInputs,
    CalculationResult,
)
from anvil.generation.generator import AnvilGenerator
from anvil.knowledge.graph_store import GraphStore
from anvil.pinned import (
    get_allowable_stress,
    get_joint_efficiency,
    get_material,
    list_materials,
)
from anvil.schemas.generation import AnvilResponse

# ---- request/response models ------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class QueryResponse(BaseModel):
    response: AnvilResponse
    retrieved_element_ids: list[str]
    citation_validation: dict[str, object]


class CalculateRequest(BaseModel):
    component: Literal[
        "cylindrical_shell",
        "cylindrical_shell_outside_radius",
        "spherical_shell",
    ]
    P_mpa: float = Field(gt=0)
    design_temp_c: float
    material: str
    joint_type: int = Field(ge=1, le=6)
    rt_level: Literal["Full RT", "Spot RT", "No RT"]
    corrosion_allowance_mm: float = Field(ge=0)
    inside_diameter_mm: float | None = None
    outside_diameter_mm: float | None = None


class CalculateResponse(BaseModel):
    formula_ref: str
    S_mpa: float
    E: float
    R_mm: float
    applicability_lhs: float
    applicability_rhs: float
    applicability_ok: bool
    t_min_mm: float
    t_design_mm: float
    t_nominal_mm: int
    mawp_mpa: float
    warnings: list[str]


class MaterialResponse(BaseModel):
    spec_no: str
    grade: str
    product_form: str
    min_tensile_mpa: float
    min_yield_mpa: float
    stress_by_temp_c: dict[int, float]
    max_temp_c: int
    notes: list[str]


class GraphNodeResponse(BaseModel):
    node_id: str
    attributes: dict[str, object]
    outgoing_edges: list[dict[str, object]]
    incoming_neighbors: list[str]


# ---- route factory ----------------------------------------------------------


def build_router(
    generator: AnvilGenerator,
    graph_store: GraphStore | None = None,
    calc_engine: CalculationEngine | None = None,
) -> APIRouter:
    router = APIRouter()
    calc_engine = calc_engine or CalculationEngine()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "retriever": "ready",
            "graph_store": "ready" if graph_store else "not_configured",
            "materials_loaded": str(len(list_materials())),
        }

    @router.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        outcome = await generator.generate(req.query, top_k=req.top_k)
        return QueryResponse(
            response=outcome.response,
            retrieved_element_ids=[c.element_id for c in outcome.retrieved_chunks],
            citation_validation={
                "total": outcome.citation_validation.total,
                "valid": outcome.citation_validation.valid,
                "accuracy": outcome.citation_validation.accuracy,
                "issues": [
                    {"index": i.citation_index, "issue": i.issue}
                    for i in outcome.citation_validation.issues
                ],
            },
        )

    @router.post("/calculate", response_model=CalculateResponse)
    def calculate(req: CalculateRequest) -> CalculateResponse:
        try:
            result: CalculationResult = calc_engine.calculate(
                CalculationInputs(
                    component=req.component,
                    P_mpa=req.P_mpa,
                    design_temp_c=req.design_temp_c,
                    material=req.material,
                    joint_type=req.joint_type,
                    rt_level=req.rt_level,
                    corrosion_allowance_mm=req.corrosion_allowance_mm,
                    inside_diameter_mm=req.inside_diameter_mm,
                    outside_diameter_mm=req.outside_diameter_mm,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CalculateResponse(
            formula_ref=result.formula_ref,
            S_mpa=result.S_mpa,
            E=result.E,
            R_mm=result.R_mm,
            applicability_lhs=result.applicability_lhs,
            applicability_rhs=result.applicability_rhs,
            applicability_ok=result.applicability_ok,
            t_min_mm=result.t_min_mm,
            t_design_mm=result.t_design_mm,
            t_nominal_mm=result.t_nominal_mm,
            mawp_mpa=result.mawp_mpa,
            warnings=result.warnings,
        )

    @router.get("/materials", response_model=list[str])
    def materials_list() -> list[str]:
        return list_materials()

    @router.get("/materials/{spec_grade:path}", response_model=MaterialResponse)
    def material_lookup(spec_grade: str) -> MaterialResponse:
        mat = get_material(spec_grade)
        if mat is None:
            raise HTTPException(404, detail=f"Material '{spec_grade}' not found")
        return MaterialResponse(
            spec_no=mat.spec_no,
            grade=mat.grade,
            product_form=mat.product_form,
            min_tensile_mpa=mat.min_tensile_mpa,
            min_yield_mpa=mat.min_yield_mpa,
            stress_by_temp_c=mat.stress_by_temp_c,
            max_temp_c=mat.max_temp_c,
            notes=mat.notes,
        )

    @router.get("/stress/{spec_grade:path}")
    def stress_lookup(spec_grade: str, temp_c: float) -> dict[str, object]:
        S = get_allowable_stress(spec_grade, temp_c)
        if S is None:
            raise HTTPException(
                404,
                detail=f"Stress unavailable for {spec_grade} at {temp_c}°C",
            )
        return {"material": spec_grade, "temp_c": temp_c, "S_mpa": S}

    @router.get("/joint_efficiency/{joint_type}/{rt_level:path}")
    def efficiency_lookup(joint_type: int, rt_level: str) -> dict[str, object]:
        E = get_joint_efficiency(joint_type, rt_level)
        if E is None:
            raise HTTPException(
                404,
                detail=f"No efficiency for Type {joint_type} / {rt_level}",
            )
        return {"joint_type": joint_type, "rt_level": rt_level, "E": E}

    @router.get("/graph/node/{node_id}", response_model=GraphNodeResponse)
    def graph_node(node_id: str) -> GraphNodeResponse:
        if graph_store is None or node_id not in graph_store.graph:
            raise HTTPException(404, detail=f"Node '{node_id}' not found")
        attrs = dict(graph_store.graph.nodes[node_id])
        outgoing: list[dict[str, object]] = [
            {"target": t, "edge_type": et, "reference_text": rt}
            for t, et, rt in graph_store.neighbors(node_id)
        ]
        incoming = list(graph_store.graph.predecessors(node_id))
        return GraphNodeResponse(
            node_id=node_id,
            attributes=attrs,
            outgoing_edges=outgoing,
            incoming_neighbors=incoming,
        )

    return router
