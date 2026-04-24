"""API integration tests via the FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from anvil.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_materials_list(client: TestClient) -> None:
    r = client.get("/materials")
    assert r.status_code == 200
    assert "SM-516 Gr 70" in r.json()


def test_material_detail(client: TestClient) -> None:
    r = client.get("/materials/SM-516 Gr 70")
    assert r.status_code == 200
    data = r.json()
    assert data["spec_no"] == "SM-516"
    assert data["grade"] == "Gr 70"
    assert data["stress_by_temp_c"]["350"] == 114


def test_material_not_found(client: TestClient) -> None:
    r = client.get("/materials/SM-999 Gr XYZ")
    assert r.status_code == 404


def test_stress_lookup(client: TestClient) -> None:
    r = client.get("/stress/SM-516 Gr 70", params={"temp_c": 275})
    assert r.status_code == 200
    assert abs(r.json()["S_mpa"] - 124.0) < 1e-6


def test_stress_lookup_over_max(client: TestClient) -> None:
    r = client.get("/stress/SM-516 Gr 70", params={"temp_c": 700})
    assert r.status_code == 404


def test_joint_efficiency_lookup(client: TestClient) -> None:
    r = client.get("/joint_efficiency/1/Full RT")
    assert r.status_code == 200
    assert r.json()["E"] == 1.0


def test_calculate_endpoint(client: TestClient) -> None:
    body = {
        "component": "cylindrical_shell",
        "P_mpa": 1.5,
        "design_temp_c": 350,
        "material": "SM-516 Gr 70",
        "joint_type": 1,
        "rt_level": "Full RT",
        "corrosion_allowance_mm": 3.0,
        "inside_diameter_mm": 1800,
    }
    r = client.post("/calculate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert abs(data["t_min_mm"] - 11.94) < 0.02
    assert data["t_nominal_mm"] == 16
    assert abs(data["mawp_mpa"] - 1.633) < 0.005


def test_calculate_validation_error(client: TestClient) -> None:
    body = {
        "component": "cylindrical_shell",
        "P_mpa": 1.0,
        "design_temp_c": 700,
        "material": "SM-516 Gr 70",
        "joint_type": 1,
        "rt_level": "Full RT",
        "corrosion_allowance_mm": 1.0,
        "inside_diameter_mm": 1000,
    }
    r = client.post("/calculate", json=body)
    assert r.status_code == 422


def test_query_endpoint(client: TestClient) -> None:
    r = client.post(
        "/query",
        json={"query": "What joint efficiency applies to a Type 1 Full RT joint?", "top_k": 5},
    )
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert data["response"]["confidence"] != "insufficient"


def test_query_ood_refused(client: TestClient) -> None:
    r = client.post("/query", json={"query": "What is the weather in San Jose?"})
    assert r.status_code == 200
    assert r.json()["response"]["confidence"] == "insufficient"
