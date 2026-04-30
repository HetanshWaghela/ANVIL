# ANVIL Local Demo

A minimalist black-and-white frontend that demonstrates the full ANVIL pipeline
for interview demos. Local-only — no auth, no deployment.

## Quick Start

```bash
# 1. Put this in .env once
ANVIL_LLM_BACKEND=nvidia_nim
ANVIL_LLM_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_API_KEY=nvapi-...
ANVIL_EMBEDDER=hash

# 2. Run the demo server. scripts/run_demo.py auto-loads .env.
uv run python scripts/run_demo.py

# 3. Open http://localhost:8899 in your browser
```

The status indicator in the top nav shows green when a real backend is configured
and amber when falling back to FakeLLMBackend (queries will be refused).

The demo runner reads repo-root `.env` automatically and does not override values
already exported in your shell. To temporarily bypass `.env`, run with
`ANVIL_DEMO_LOAD_ENV=0 uv run python scripts/run_demo.py`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANVIL_LLM_BACKEND` | Yes | `nvidia_nim`, `openai_compatible`, or `instructor` |
| `ANVIL_LLM_MODEL` | Yes | Model identifier (e.g. `meta/llama-3.3-70b-instruct`) |
| `NVIDIA_API_KEY` | For nvidia_nim | NIM API key from build.nvidia.com |
| `OPENAI_COMPAT_BASE_URL` | For openai_compatible | Base URL of OpenAI-protocol endpoint |
| `OPENAI_COMPAT_API_KEY` | For openai_compatible | API key for the endpoint |
| `ANVIL_EMBEDDER` | Recommended | `hash` for low-latency live demo, `sentence_transformer` for semantic retrieval quality |
| `DEMO_PORT` | No | Server port (default: 8899) |

If any required variable is missing, the UI shows a clear error and refuses to
run queries. Upload/parse and pinned table inspection still work without an LLM.

For the interview demo, keep `ANVIL_EMBEDDER=hash` so the pre-indexed ASME VIII-1
upload returns quickly. If you switch to `sentence_transformer`, the result is
more semantically useful but first-load indexing can take tens of seconds.

## Features

### Upload & Parse
Upload a PDF or load the built-in SPES-1 synthetic standard. The backend:
- Saves the PDF under `data/private/uploads/` (gitignored)
- Recognizes the licensed ASME VIII-1 demo PDF by SHA-256 and loads the
  pre-indexed private markdown/elements/graph artifacts for a fast live demo
- Extracts markdown via pymupdf4llm for other PDFs
- Parses into typed DocumentElements
- Builds a knowledge graph and vector index
- Returns detailed stats (elements, tables, formulas, graph metrics)

### Pinned Tables
Inspects the verified material properties (Table M-1) and joint efficiencies
(Table B-12) that the calculation engine uses. These are never extracted by the
LLM — they are deterministic lookups from structured data files.

### Query Demo
Runs queries against the real ANVIL pipeline. Shows answer, confidence, citations,
calculation steps, latency, and validation results. Demo queries are:
- **SPES-1 corpus**: pre-defined queries that exercise the full calculation pipeline
- **Uploaded PDF**: queries generated from parsed section/table/formula metadata

### Pipeline Explainer
A visual walkthrough of the 8-step pipeline: Parse → Index → Retrieve → Pin →
Calculate → Generate → Validate → Refuse.

## Interview Framing

The UI makes these distinctions clear:

| Corpus | What works | What doesn't (yet) |
|---|---|---|
| **SPES-1 (synthetic)** | Full pipeline: parsing, retrieval, graph, pinned tables, deterministic calculation, LLM generation, citation validation | — |
| **Uploaded real ASME PDF** | Parsing, retrieval, graph onboarding, citation demos | Deterministic calculation (requires validated pinned tables and formula tools for the specific standard) |

This is engineering maturity, not a limitation. The architecture supports onboarding
new standards — it requires creating validated pinned tables and formula tools, plus
golden evaluation datasets.

## Privacy

- Uploaded PDFs and extracted markdown are saved under `data/private/uploads/`
- `data/private/` is gitignored — files never enter version control
- No private ASME text is baked into source files
- Run `uv run python scripts/audit_private_artifacts.py` to verify
- Check `git status` after uploading to confirm nothing leaks

## Architecture

```
scripts/run_demo.py          → entry point (loads .env, starts uvicorn)
src/anvil/api/demo_routes.py → /demo/* API endpoints
src/anvil/api/static/demo.html → single-page frontend
```

The demo routes import and use the real ANVIL pipeline components:
`parse_markdown_standard`, `build_graph`, `HybridRetriever`, `AnvilGenerator`,
`get_default_backend`, `get_default_embedder`, pinned data modules.

No separate build step. No npm. No node. Plain HTML/CSS/JS served by FastAPI.
