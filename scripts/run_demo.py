"""Run the ANVIL local demo server.

Usage:
    uv run python scripts/run_demo.py

Starts a FastAPI server on http://localhost:8899 with:
  - /demo/*  API endpoints (corpus management, query, pinned data)
  - /        serves the demo frontend
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from anvil.api.demo_routes import build_demo_router


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_local_env() -> bool:
    """Load repo-root .env for the local demo without overriding shell env."""
    should_load = os.environ.get("ANVIL_DEMO_LOAD_ENV", "1").lower().strip()
    if should_load in {"0", "false", "no", "off"}:
        return False

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = _strip_env_value(value)
    return True


_DOTENV_LOADED = _load_local_env()


def create_demo_app() -> FastAPI:
    app = FastAPI(
        title="ANVIL Local Demo",
        version="0.1.0",
        description="Local interview demo for the ANVIL compliance-grade RAG system.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build_demo_router())

    static_dir = Path(__file__).resolve().parents[1] / "src" / "anvil" / "api" / "static"

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "demo.html", media_type="text/html")

    return app


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "8899"))
    dotenv_status = "loaded .env" if _DOTENV_LOADED else ".env not loaded"
    print(f"\n  ANVIL Local Demo -> http://localhost:{port} ({dotenv_status})\n")
    app = create_demo_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
