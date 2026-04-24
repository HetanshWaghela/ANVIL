# ANVIL — Compliance-Grade Retrieval Over Engineering Standards

## Project Overview
Compliance-grade RAG system for engineering standards (ASME BPVC-like).
Demonstrates structural document parsing → typed knowledge graph → graph-aware retrieval → evidence-enforced generation with provenance tracking.

## Commands
- `uv run pytest tests/unit/ -x -q` — Run unit tests (fast, run after every change)
- `uv run pytest tests/integration/ -x -q` — Run integration tests
- `uv run pytest tests/evaluation/ -v` — Run evaluation suite with metrics
- `uv run python scripts/ingest.py` — Ingest synthetic standard into KG + vector store
- `uv run python scripts/evaluate.py` — Full evaluation pipeline
- `uv run python scripts/demo.py` — Interactive query demo
- `uv run ruff check src/ tests/` — Lint
- `uv run mypy src/` — Type check

## Architecture
- `src/anvil/schemas/` — Pydantic models. All data flows through typed schemas.
- `src/anvil/parsing/` — PDF/Markdown → structured document elements (sections, tables, formulas)
- `src/anvil/knowledge/` — Parsed elements → typed knowledge graph (NetworkX)
- `src/anvil/retrieval/` — Hybrid retrieval: BM25 + vector (sqlite-vec) + graph expansion
- `src/anvil/generation/` — Evidence-enforced generation with mandatory citations
- `src/anvil/evaluation/` — RAGAS-inspired metrics + domain-specific metrics
- `src/anvil/pinned/` — Verified ground truth data for critical lookups (never from RAG)
- `src/anvil/api/` — FastAPI endpoints

## Conventions
- Python 3.12+. Type annotations on every function signature. No `Any` except at LLM boundaries.
- Pydantic v2 for all data models. Use `model_validator` for cross-field constraints.
- Async by default for I/O-bound operations. Sync for CPU-bound (parsing, graph ops).
- No LangChain. No LlamaIndex. Raw API calls + custom pipeline. We own every line.
- Every LLM response must be a Pydantic model with a `citations` field. No ungrounded text.
- Use `structlog` for structured logging. Every retrieval/generation step logged with metadata.

## Guardrails
- NEVER hallucinate material properties. If a value isn't in the retrieved context or pinned data, REFUSE.
- NEVER generate a calculation result without showing the formula, every input value, and its source.
- NEVER commit secrets, API keys, or model weights to the repo.
- Tests must assert on behavior, not implementation. No mocking the LLM in integration tests — use recorded responses.
- Every metric in the evaluation framework must have a clear mathematical definition in code comments.

## Domain Rules
- ASME code paragraph references follow the pattern: UG-27(c)(1), UW-12, UCS-23, etc. In the synthetic standard (SPES-1), these become A-27(c)(1), B-12, etc.
- Material specs in the SYNTHETIC standard use SM- prefix: SM-516 Gr 70, SM-240 Type 304, SM-106 Grade B (NOT real ASME SA- prefix).
- Allowable stress S is ALWAYS temperature-dependent. Never use room-temperature value for elevated temp design.
- Joint efficiency E depends on BOTH weld type AND examination extent. Both must be specified.
- Corrosion allowance is ADDED to calculated thickness, never subtracted.
