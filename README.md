# ANVIL

[![CI](https://github.com/Hetansh/ANVIL/actions/workflows/ci.yml/badge.svg)](https://github.com/Hetansh/ANVIL/actions/workflows/ci.yml)

**Compliance-grade retrieval-augmented reasoning over engineering standards.**

ANVIL is an end-to-end RAG system that answers ASME-style pressure-vessel
queries with auditable provenance: every numeric value is either a user input,
a pinned-table lookup with a citation, or a deterministic `Decimal` calculation
with its formula cited to a paragraph in the source standard. No values come
from model memory. No arithmetic comes from the model. Refusal is a first-class
output, not a fallback.

The system is benchmarked on **SPES-1** — the *Synthetic Pressure Equipment
Standard* — a fully-published, project-internal analog of ASME BPVC Section
VIII Div. 1 built so the entire pipeline (corpus, ground truth, evaluations,
ablations) can be reproduced from this repository without licensed material.

## Headline result

On the 100-example public SPES-1 benchmark, real `meta/llama-3.3-70b-instruct`
through NVIDIA NIM:

| metric | baseline | no-pinned ablation | delta |
| :--- | ---: | ---: | ---: |
| **`pass_rate`** | **0.950** | 0.520 | **−0.430** |
| **`calculation_correctness`** | **1.000** | **0.000** | **−1.000** |
| `citation_accuracy` | **1.000** | 0.997 | −0.003 |
| `faithfulness` | 0.977 | 0.900 | −0.077 |
| `refusal_calibration` | 0.980 | 0.950 | −0.030 |
| `retrieval_recall@10` | **1.000** | 1.000 | 0 |

That `calculation_correctness` line — **1.000 → 0.000** — is the design
thesis: when the LLM is allowed to extract material allowable stresses from
retrieved text instead of looking them up in a pinned, versioned table, every
single calculation answer is wrong. ANVIL forces the dependency on trusted
data and audited code, and the rest of the pipeline is built around that
constraint.

Run IDs and full per-example artifacts:
- `data/runs/2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline/`
- `data/runs/2026-04-26T04-55-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-pinned/`

Every metric has a mathematical definition in its docstring at
[`src/anvil/evaluation/metrics.py`](src/anvil/evaluation/metrics.py); every
run directory carries a `manifest.json` with the dataset hash, git SHA,
model id, and full env so a reviewer can byte-identically reproduce it.

## Architecture

```
                    SPES-1 markdown / PDF
                            │
      pymupdf4llm + table-repair + applicability extraction
                            ▼
              typed DocumentElement[] (pydantic v2)
                            │
            ┌───────────────┼─────────────────────┐
            ▼               ▼                     ▼
      bm25s index    sqlite-vec store      NetworkX KG
      (lexical)      (BAAI/bge-small)      (typed edges)
            │               │                     │
            └───── RRF + graph-expansion + reranker ───────┐
                            │                              │
                            ▼                              │
                   RetrievedChunk[]                        │
                            │                              │
       ┌────────────────────┼─────────────────────┐        │
       ▼                    ▼                     ▼        │
  refusal gate     CalculationEngine        prompt builder │
  (pre-LLM)        (Decimal, pinned         + LLM          │
                    materials + B-12)        + structured  │
                            │                  output)     │
                            └────────┬─────────────────────┘
                                     ▼
                              AnvilResponse
                                     │
                  citation_enforcer.validate_citations
                  (every quoted_text is a substring of
                   the cited element's content)
                                     ▼
                            audited final output
```

Two generation paths share the same components:

- **Fixed pipeline** (`AnvilGenerator.generate`): deterministic
  retrieve → gate → calculate → generate → validate. This is the production
  path; it produced every headline number above.
- **Agentic loop** (`AnvilAgent.run`): the LLM decides which of
  `retrieve_context | graph_lookup | pinned_lookup | calculate | finalize` to
  call, turn-by-turn. Every (call, result) is recorded in
  `AgentTranscript`, persisted next to per-example metrics, and bounded by
  `AgentBudget(max_steps=8, max_tool_errors=3)`.

## Four invariants the codebase enforces

1. **No ungrounded claims.** Every `Citation` has a `source_element_id`, a
   `paragraph_ref`, and a `quoted_text` that is a verbatim substring of the
   cited element's content.
   [`src/anvil/generation/citation_enforcer.py`](src/anvil/generation/citation_enforcer.py)
   validates each citation post-generation against retrieved chunks *and* an
   element index, so canonical-ref citations into pinned-data tables (which
   may not appear in retrieval) still have their quotes verified against the
   parsed standard.
2. **No LLM arithmetic.** The LLM selects which formula to apply and
   identifies the inputs; every numeric result comes from
   [`src/anvil/pinned/formulas.py`](src/anvil/pinned/formulas.py) with
   28-digit `Decimal` precision and half-up rounding. Each
   `CalculationStep` is typed by a `StepKey` enum so downstream consumers
   cannot misidentify a step by its description string (ADR-004).
3. **Refusal runs *before* the LLM.** A "refuse to answer" decision made
   *after* generation is already a generated response and therefore a
   citation liability. `refusal_gate` rejects out-of-domain queries,
   unknown materials, out-of-range temperatures, and incomplete
   calculation inputs before the LLM is ever called.
4. **Pinned ground truth, not retrieved values.** Material allowable
   stresses and joint efficiencies live in versioned JSON
   (`data/synthetic/table_1a_materials.json`,
   `joint_efficiency_table.json`). The retrieval layer indexes Tables M-1
   and B-12 so generation has *textual provenance* for citations, but the
   actual values used in calculations come from pinned data through
   `get_allowable_stress(spec_grade, temp_c)` with linear interpolation
   inside the tabulated range and explicit refusal outside it. A
   regression test (`test_markdown_table_m1_matches_pinned_data`)
   guarantees the two sources cannot drift.

These invariants are not aspirational; each maps to a regression test in
`tests/unit/` that fails if the corresponding behavior breaks. CI runs the
281+ test suite under `--strict` mypy on every push.

## Quickstart

```bash
uv sync --extra dev                                  # install
uv run pytest tests/ -q                              # 285 tests, ~1.5s
cp .env.example .env && set -a && source .env && set +a   # configure NIM keys
uv run anvil nim-check                               # confirm catalog reachability
uv run anvil ingest                                  # parse SPES-1 → KG + element artifacts
uv run anvil query "What does A-27(c)(1) require?"
uv run anvil calculate --component cylindrical_shell --P 1.5 --temp 350 \
  --material "SM-516 Gr 70" --joint-type 1 --rt-level "Full RT" \
  --ca 3.0 --inside-diameter-mm 1800
uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline
uv run uvicorn anvil.api.app:create_app --factory    # FastAPI demo
```

The production path is NVIDIA NIM with the sentence-transformer embedder —
every headline number above came from that configuration. Any unrecognized
value for `ANVIL_LLM_BACKEND` / `ANVIL_EMBEDDER` raises at startup rather
than silently coercing to a deterministic offline backend, so a misconfigured
production deploy fails loudly close to the configuration site instead of
shipping degraded responses far from it.

## Reproducing the headline results

```bash
# 1. Configure a real NIM backend.
cp .env.example .env                                                 # edit NVIDIA_API_KEY
set -a; source .env; set +a
export ANVIL_EMBEDDER=sentence_transformer

# 2. Confirm the NIM catalog is reachable + drift-checked.
uv run anvil nim-check --json --list

# 3. Run the full 4-row real ablation matrix on 100 public SPES-1 examples.
for abl in baseline no-pinned no-refusal no-citation-enforcer; do
  uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct \
    --ablation $abl --min-pass-rate 0.0
done

# 4. (Optional) Run the agent loop head-to-head with the fixed pipeline.
uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8

# 5. (Optional) Refresh the parser benchmark with Reducto + local pymupdf4llm.
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer,reducto
```

If an interrupted eval needs to be resumed:

```bash
uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct \
  --ablation baseline --resume-run <run_id>
```

`EvaluationRunner` keeps a per-run `checkpoint.json` so completed examples are
never re-charged against the NIM quota.

## Reviewer artifacts

| Artifact | What it shows |
| :--- | :--- |
| [`docs/report.md`](docs/report.md) | Workshop-paper writeup tying every real result to its run ID, dataset hash, git SHA, and design decisions |
| [`docs/headline_results.md`](docs/headline_results.md) | The current real-backend results table |
| [`docs/ablations.md`](docs/ablations.md) | Component ablations (BM25 vs vector vs hybrid; pinned-on vs pinned-off; refusal-on vs refusal-off; citation-enforcer-on vs off) with full interpretation |
| [`docs/trust_calibration.md`](docs/trust_calibration.md) | 9-point sweep of `RELEVANCE_THRESHOLD`; defended operating point of 0.05 with perfect refusal precision/recall on the golden OOD cohort |
| [`docs/parser_benchmark.md`](docs/parser_benchmark.md) | PDF parser benchmark on SPES-1 and NASA SP-8007 across `pymupdf4llm`, `naive_pdfminer`, and `reducto`; defended default of local `pymupdf4llm` |
| [`docs/agent_loop.md`](docs/agent_loop.md) | Tool surface, budget enforcement, and host-side guardrails for the agentic loop |
| [`docs/agent_results.md`](docs/agent_results.md) | Real 100-example agent run with full transcripts and the reasoning for keeping the fixed pipeline as the application path |
| [`docs/design_decisions.md`](docs/design_decisions.md) | 14 ADR-style entries: synthetic standard, pinned data, no LLM arithmetic, `StepKey` enum, fail-loud retrieval, citation builder, etc. |
| [`docs/cost_budget.md`](docs/cost_budget.md) | NIM request budget, quota guardrails, and recommended live-run sequence |
| [`docs/private_asme.md`](docs/private_asme.md) | Private licensed-ASME validation boundary: private inputs/runs only, sanitized aggregate metrics only |

## Production backends

ANVIL ships with four `LLMBackend` implementations behind a single Pydantic
contract; switching is one environment variable.

```bash
# NVIDIA NIM (default recommendation; what the headline numbers used)
export ANVIL_EMBEDDER=sentence_transformer
export ANVIL_LLM_BACKEND=nvidia_nim
export NVIDIA_API_KEY=nvapi-...
export ANVIL_LLM_MODEL=meta/llama-3.3-70b-instruct
# optional reasoning controls for capable models:
export ANVIL_NIM_REASONING=1
export ANVIL_NIM_REASONING_EFFORT=high

# Any OpenAI-compatible endpoint (Together, Fireworks, vLLM, Ollama, LM Studio)
export ANVIL_LLM_BACKEND=openai_compatible
export OPENAI_COMPAT_BASE_URL=https://api.together.xyz/v1
export OPENAI_COMPAT_API_KEY=...
export ANVIL_LLM_MODEL=deepseek-ai/deepseek-v3

# Native Anthropic / OpenAI via instructor.from_provider
export ANVIL_LLM_BACKEND=instructor
export ANVIL_LLM_MODEL=anthropic/claude-3-5-sonnet-latest
```

The NIM backend has per-key request throttling, fallback-key rotation, and
typed retry classification (`RetryableGenerationError` distinguishes
transient transport noise from permanent contract violations).

## Repository layout

```
ANVIL/
├── data/
│   ├── synthetic/                 # SPES-1 standard + pinned ground truth
│   │   ├── standard.md            # markdown version of the standard
│   │   ├── table_1a_materials.json    # allowable stress table (pinned)
│   │   ├── joint_efficiency_table.json
│   │   └── design_examples.json   # 10 hand-verified worked examples
│   ├── runs/                      # stamped per-run artifacts (manifest, per_example, summary, raw)
│   └── parser_benchmark/          # PDF corpus + ground-truth annotations
├── src/anvil/                     # ~10k LOC across 9 subpackages
│   ├── schemas/                   # Pydantic v2 schemas — the typed spine
│   ├── parsing/                   # PDF/Markdown → DocumentElement[]
│   ├── knowledge/                 # DocumentElement[] → typed KG (NetworkX)
│   ├── retrieval/                 # BM25s + sqlite-vec + graph + RRF + rerank
│   ├── generation/                # prompt, LLM backends, calc engine, citation enforcer, agent loop
│   ├── pinned/                    # ground-truth data + Decimal formula functions
│   ├── evaluation/                # RAGAS-inspired + domain-specific metrics, ablation harness, run logger
│   ├── api/                       # FastAPI app + routes + middleware
│   ├── cli.py                     # `anvil` entry point: ingest / query / calculate / eval / compare / nim-check
│   └── pipeline.py                # build_pipeline() factory
├── tests/                         # 285 tests — unit / integration / evaluation / fail-loud regressions
├── scripts/                       # batch drivers: run_ablations, run_calibration, run_agent_eval, run_parser_benchmark, ...
└── docs/                          # architecture, ADRs, evaluation methodology, cost budget, parser benchmark
```

Licensed ASME PDFs, extracted text, private indexes, prompts, raw responses,
and private run artifacts are intentionally excluded from git
(`scripts/audit_private_artifacts.py` enforces this and is part of CI).
Public real-world parser stress uses NASA pressure-system standards via
`data/parser_benchmark/public_pressure_sources.json` and
`scripts/download_public_pressure_docs.py`.

## Example: a calculation query

```python
import asyncio
from anvil.pipeline import build_pipeline

async def main():
    pipeline = build_pipeline()
    outcome = await pipeline.generator.generate(
        "Calculate thickness for cylindrical shell ID=1800 mm, P=1.5 MPa, "
        "T=350°C, SM-516 Gr 70, Type 1 with full RT, CA=3.0 mm.",
    )
    r = outcome.response
    print(r.answer)
    for step in r.calculation_steps:
        print(f"  step {step.step_number} [{step.result_key}] "
              f"{step.formula} → {step.result} {step.unit}  "
              f"({step.citation.paragraph_ref})")
    for c in r.citations:
        print(f"  cite {c.paragraph_ref}: {c.quoted_text[:80]!r}")

asyncio.run(main())
```

Produces:

```
Minimum required thickness t_min = 11.94 mm; design thickness (with corrosion allowance) = 14.94 mm; selected nominal plate = 16 mm; MAWP (back-calculated) = 1.633 MPa.
  step 1 [allowable_stress] S = Table M-1(SM-516 Gr 70, 350.0°C) → 114.0 MPa  (Table M-1)
  step 2 [joint_efficiency] E = Table B-12(Type 1, Full RT) → 1.0 dimensionless  (Table B-12)
  step 3 [applicability_check] P ≤ 0.385 × S × E → 43.89 MPa  (A-27(c)(1))
  step 4 [min_thickness]       t = (P × R) / (S × E − 0.6 × P) → 11.94 mm  (A-27(c)(1))
  step 5 [design_thickness]    t_design = t_min + CA → 14.94 mm  (A-25)
  step 6 [nominal_plate]       next_standard_plate(t_design) → 16 mm  (A-27(c)(1))
  step 7 [mawp]                P = (S × E × t_corr) / (R + 0.6 × t_corr) → 1.633 MPa  (A-27(c)(1))
```

Every numeric value is either (a) a user input, (b) a pinned-data lookup with a
citation into Table M-1 or B-12, or (c) a deterministic calculation whose
formula is cited to the paragraph it comes from.

## What refusal looks like

```python
await pipeline.generator.generate("What is the weather in San Jose today?")
# confidence = INSUFFICIENT
# refusal_reason = "No sufficiently relevant content found in the standard (max relevance 0.000 < threshold 0.050)."

await pipeline.generator.generate("Stress for SM-999 Gr XYZ at 300°C?")
# refusal_reason = "Material 'SM-999 Gr XYZ' is not in the SPES-1 pinned materials table..."

await pipeline.generator.generate("Thickness for SM-516 Gr 70 at 700°C, P=1 MPa, ID=1000 mm, Type 1 Full RT, CA=2 mm")
# refusal_reason = "Design temperature 700.0°C exceeds the maximum tabulated temperature for SM-516 Gr 70 (500°C per M-23)."
```

## CLI surface

| Command | Purpose |
| :--- | :--- |
| `anvil nim-check` | Probe the NVIDIA NIM model catalog; report latency, reachability, and drift vs. the locked default catalog |
| `anvil ingest` | Parse the bundled SPES-1 standard and write KG + element artifacts |
| `anvil query "<question>"` | Full retrieval → refusal → generation → citation-enforcer pipeline |
| `anvil calculate ...` | Deterministic calculation engine from explicit inputs (no LLM in the loop) |
| `anvil eval --backend nvidia_nim --model ... --ablation ...` | Stamped `data/runs/<run_id>/` evaluation directory with full provenance |
| `anvil compare <run_dir> ...` | Render a Markdown comparison table from run summaries |

Batch studies still live in `scripts/`: `run_ablations.py`,
`run_calibration.py`, `run_nim_headlines.py`, `run_parser_benchmark.py`,
`run_agent_eval.py`, `audit_private_artifacts.py`.

## Agentic loop

`AnvilAgent` exposes the four primitives as typed tools and lets a real LLM
pick one each turn:

```
retrieve_context(query, top_k)             → BM25+vector+graph chunks
graph_lookup(paragraph_ref, max_hops)      → KG seed + n-hop neighborhood
pinned_lookup(kind, key, temp_c?, rt?)     → trusted table value + row citation
calculate(component, P, T, material, ...)  → Decimal calculation + steps + citations
finalize(response: AnvilResponse)          → emit final structured answer
```

The loop is bounded (`AgentBudget(max_steps=8, max_tool_errors=3)`) and
fail-soft (every tool returns a `ToolResult`; nothing raises out of the
loop). Three deterministic auto-finalization paths sit between the LLM and
the response:

- **After a successful `calculate`** — the host assembles the
  `AnvilResponse` directly from the calculation steps and citations the
  trusted engine produced. The model is not asked to re-summarize
  arithmetic it did not perform.
- **After a successful `pinned_lookup` for a direct table question** — the
  host assembles the answer from the pinned row plus its `CitationBuilder`
  reference. No additional LLM turn is required.
- **After 2 successful `retrieve_context` / `graph_lookup` calls without a
  deterministic tool firing** — the host hands the agent-curated chunks to
  `AnvilGenerator.synthesize_from_chunks(...)`, which is steps 4–6 of the
  fixed pipeline (prompt → LLM → citation validation) executed on the
  agent-selected context. This eliminates the runaway-retrieval failure
  mode (the model rephrasing its query 8 times instead of finalizing) that
  capped the previous agent run at 0/20 lookup and 0/20 cross_reference
  passes.

Every `(call, result)` lands in `AgentTranscript.steps`, persisted at
`data/runs/<run_id>__agent/agent_transcripts.json` so a reviewer can replay
the agent's decisions deterministically. Regression tests in
`tests/unit/test_agent.py` lock each guardrail.

## Deployment

A minimal `Dockerfile` and `fly.toml` are included for a read-only FastAPI
demo. To deploy a live NIM-backed demo, set `NVIDIA_API_KEY`,
`ANVIL_LLM_BACKEND=nvidia_nim`, `ANVIL_EMBEDDER=sentence_transformer`, and
an explicit `ANVIL_LLM_MODEL` as platform secrets. Rationale and tradeoffs
in ADR-013 (`docs/design_decisions.md`).

## Quality gates (run on every push)

```bash
uv run ruff check src/ tests/ scripts/   # lint
uv run mypy src/                         # --strict, 61 source files
uv run pytest tests/ -q                  # 285 tests, ~1.5s
```

All three pass on `main`. CI is a GitHub Actions workflow
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) running `ruff`,
`mypy --strict`, and `pytest`, plus an optional `nim-check` job that runs
only when `NVIDIA_API_KEY` is configured as a repo secret.

## Honest limitations

The real-backend validation path is complete for the fixed pipeline: NIM
health checks, 100-example Meta baseline, three real ablations, Reducto
parser benchmark, private-ASME artifact audit, and a 100-example agent run
all have local artifacts.

Intentionally scoped gaps:

1. DeepSeek V4 Flash and V4 Pro timed out in the targeted 2026-04-26 NIM
   probe, so they were skipped. Reported as such, not papered over.
2. The most recent committed agent transcripts (`run_id` ending in
   `06-35-00Z_..._abl-baseline__agent`) predate the retrieval-saturation
   fix; they remain in `data/runs/` as historical evidence of the
   pre-fix failure mode. A re-run on the fixed code is the next item in
   `docs/cost_budget.md`.
3. Reducto's adapter works but its SPES-1 formula recovery underperforms
   `pymupdf4llm` (the defended local default — see ADR-014).
4. A licensed-ASME validation pass is local-only; sanitized aggregate
   metrics are the only thing that may leave the private workspace.

## License

MIT. The synthetic standard in `data/synthetic/` is an original work
inspired by public ASME BPVC conventions; no proprietary content.
