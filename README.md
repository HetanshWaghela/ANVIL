# ANVIL

[![CI](https://github.com/Hetansh/ANVIL/actions/workflows/ci.yml/badge.svg)](https://github.com/Hetansh/ANVIL/actions/workflows/ci.yml)

**Compliance-grade retrieval over engineering standards.**

Anvil is a retrieval-augmented reasoning system designed for the failure modes
that matter in regulated engineering domains: hallucinated material values,
fabricated code paragraph references, silent calculation errors, and quietly
ignored applicability conditions. It demonstrates an end-to-end pipeline
(parsing → typed knowledge graph → hybrid retrieval → evidence-enforced
generation → deterministic calculation) over the **Synthetic Pressure
Equipment Standard (SPES-1)** — a fully-published analog of ASME BPVC
Section VIII Div. 1 built specifically for this project.

## Why another RAG?

General RAG systems are tuned for answering "what does this say?" In regulated
engineering, the correct question is "what does the code *require*, with which
provenance, and is every input value trustworthy?" Four properties are
non-negotiable:

1. **No ungrounded claims.** Every factual statement in a response carries a
   `Citation` with a real `source_element_id`, a real `paragraph_ref`, and a
   `quoted_text` that is a substring of the retrieved content. Citations are
   validated post-generation by `citation_enforcer.py`.
2. **No LLM arithmetic.** The LLM selects the formula and identifies inputs.
   Actual arithmetic runs through `pinned/formulas.py` with `Decimal`
   precision. Every numeric result is a typed `CalculationStep` with its own
   citation.
3. **Refusal is a first-class response.** The `refusal_gate` rejects queries
   with no relevant context, unknown materials, or temperatures outside
   tabulated ranges — *before* the LLM is even invoked.
4. **Pinned ground truth.** Material allowable stresses and joint efficiencies
   live in versioned JSON, not in the RAG index. The index retrieves *textual
   provenance* for those values; the values themselves come from pinned data.

## Quickstart

```bash
uv sync --extra dev                              # install
uv run pytest tests/ -q                          # offline quality gate
uv run anvil ingest                              # parse SPES-1 → KG + element artifacts
uv run anvil eval --backend fake                 # run 100-example public benchmark
uv run anvil query "What does A-27(c)(1) require?"
uv run anvil calculate --component cylindrical_shell --P 1.5 --temp 350 \
  --material "SM-516 Gr 70" --joint-type 1 --rt-level "Full RT" \
  --ca 3.0 --inside-diameter-mm 1800
uv run anvil nim-check                           # key-gated NIM connectivity check
uv run uvicorn anvil.api.app:create_app --factory # FastAPI
```

Everything defaults to a deterministic, offline-capable backend
(`DeterministicHashEmbedder` + `FakeLLMBackend`) so CI and local dev require no
model downloads or API keys. **Both defaults log a WARNING** at startup —
a prod deploy that forgot to configure a real backend sees this
immediately, rather than shipping silent fake responses.

The `anvil` CLI is the preferred reviewer entry point. The legacy scripts in
`scripts/` remain available for batch studies and backwards-compatible CI jobs.

## Headline numbers

Application headline claims use **real backend runs**, not the deterministic
fake backend. Every metric below is from a stamped run directory under
`data/runs/` with `manifest.json`, `summary.json`, `per_example.json`, and a
run-local `report.md`.

### Real NVIDIA NIM validation (100-example public SPES-1 benchmark)

Latest full real-backend validation pass: **2026-04-26** with
`ANVIL_EMBEDDER=sentence_transformer` and
`meta/llama-3.3-70b-instruct`.

| run | pass_rate | calculation_correctness | citation_accuracy | faithfulness | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Meta baseline | **0.950** | **1.000** | **1.000** | 0.977 | 0.980 | `2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline` |
| Meta no-pinned | 0.520 | **0.000** | 0.997 | 0.900 | 0.950 | `2026-04-26T04-55-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-pinned` |
| Meta no-refusal | 0.950 | **1.000** | 0.988 | 0.994 | 0.970 | `2026-04-26T05-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-refusal` |
| Meta no-citation-enforcer | 0.970 | **1.000** | 0.998 | 0.994 | 0.980 | `2026-04-26T06-06-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-citation-enforcer` |

The strongest evidence is the pinned-data ablation: removing pinned material
and joint-efficiency tables drops calculation correctness from **1.000** to
**0.000** on the same 100-example public benchmark. That is the core
DeepMechanix-facing result: engineering facts are pulled from audited tables
and deterministic calculation code, not left to model memory.

### Trust-boundary calibration

`RELEVANCE_THRESHOLD` was swept across nine values; the chosen
operating point is **0.05** (perfect refusal precision, perfect refusal
recall, perfect non-refusal coverage on the golden OOD cohort). Full
sweep + defended choice in [`docs/trust_calibration.md`](docs/trust_calibration.md).

### Live NIM rows

Full NIM integration guide → [`docs/nim_integration.md`](docs/nim_integration.md).
Current real-result table → [`docs/headline_results.md`](docs/headline_results.md).

The 2026-04-26 NIM health check found `meta/llama-3.3-70b-instruct` reachable,
while `deepseek-ai/deepseek-v4-flash` and `deepseek-ai/deepseek-v4-pro` timed
out in the targeted probe and were skipped rather than reported with invented
numbers.

The fake backend remains for deterministic CI/testing only:
`ANVIL_LLM_BACKEND=fake ANVIL_EMBEDDER=hash uv run pytest tests/ -q`.
Fake-backend metrics can be used as offline regression evidence, not as
application performance proof.

```bash
cp .env.example .env                                    # then edit
export NVIDIA_API_KEY=nvapi-...                          # or `source .env`
uv run anvil nim-check                                  # confirm catalog
uv run anvil nim-check --json --list                    # + drift report
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline --min-pass-rate 0.0
```

Additional reviewer artifacts:

| Artifact | Status |
| :--- | :--- |
| [`docs/report.md`](docs/report.md) | Workshop-paper-style writeup tying current real results to run IDs and design decisions |
| [`docs/cost_budget.md`](docs/cost_budget.md) | NIM request budget, quota guardrails, and recommended live-run sequence |
| [`docs/agent_results.md`](docs/agent_results.md) | Complete 100-example real agent run; diagnostic, not a headline win |
| [`docs/parser_benchmark.md`](docs/parser_benchmark.md) | PDF parser benchmark and defended default parser choice, now including real Reducto rows |
| [`docs/trust_calibration.md`](docs/trust_calibration.md) | Refusal-threshold sweep and operating-point rationale |
| [`docs/ablations.md`](docs/ablations.md) | Component ablations with interpretation and limitations |
| [`docs/private_asme.md`](docs/private_asme.md) | Licensed-ASME validation guardrails: private inputs/runs only, sanitized aggregate metrics only |

### Production backends

**NVIDIA NIM (default recommendation):**

```bash
export ANVIL_EMBEDDER=sentence_transformer
export ANVIL_LLM_BACKEND=nvidia_nim
export NVIDIA_API_KEY=nvapi-...
export ANVIL_LLM_MODEL=deepseek-ai/deepseek-v3.1   # or any NIM model
# optional — enable chain-of-thought for reasoning-capable models:
export ANVIL_NIM_REASONING=1
export ANVIL_NIM_REASONING_EFFORT=high
```

**Any other OpenAI-compatible endpoint** (Together, Fireworks, vLLM, Ollama, LM Studio):

```bash
export ANVIL_LLM_BACKEND=openai_compatible
export OPENAI_COMPAT_BASE_URL=https://api.together.xyz/v1
export OPENAI_COMPAT_API_KEY=...
export ANVIL_LLM_MODEL=deepseek-ai/deepseek-v3
```

**Anthropic / OpenAI native via instructor.from_provider:**

```bash
export ANVIL_LLM_BACKEND=instructor
export ANVIL_LLM_MODEL=anthropic/claude-3-5-sonnet-latest
# or openai/gpt-4o-mini, etc.
```

Any unrecognized value for `ANVIL_LLM_BACKEND` or `ANVIL_EMBEDDER` raises
at startup — we refuse to silently coerce typos into the fake backend.

## Repository layout

```
anvil/
├── data/synthetic/               # SPES-1 standard + pinned ground truth
│   ├── standard.md               # markdown version of the standard
│   ├── table_1a_materials.json   # allowable stress table (pinned)
│   ├── joint_efficiency_table.json
│   └── design_examples.json      # 10 hand-verified worked examples
├── src/anvil/
│   ├── parsing/          # PDF/Markdown → DocumentElement[]
│   ├── knowledge/        # DocumentElement[] → typed KG (NetworkX)
│   ├── retrieval/        # BM25 + vector + graph + RRF + rerank
│   ├── generation/       # prompt, LLM, citation enforcement, calc engine
│   ├── pinned/           # ground-truth data + Decimal formula functions
│   ├── evaluation/       # RAGAS-inspired + domain-specific metrics
│   ├── api/              # FastAPI app + routes + middleware
│   └── schemas/          # Pydantic v2 schemas — the typed spine
├── tests/                # unit / integration / evaluation / fail-loud regressions
└── scripts/              # ingest.py, evaluate.py, demo.py
```

Licensed ASME PDFs, extracted text, private indexes, prompts, raw responses,
and private run artifacts are intentionally excluded from git. Public
real-world parser stress uses NASA pressure-system standards via
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

## CLI command surface

The installed `anvil` command now covers the main project workflow:

| Command | Purpose |
| :--- | :--- |
| `anvil nim-check` | Probe the active NVIDIA NIM model catalog and report latency / reachability |
| `anvil ingest` | Parse the bundled SPES-1 standard and write graph / element artifacts |
| `anvil query "<question>"` | Run the full retrieval → refusal → generation → citation pipeline |
| `anvil calculate ...` | Run the deterministic calculation engine from explicit inputs |
| `anvil eval --backend fake` | Produce a stamped `data/runs/<run_id>/` evaluation directory |
| `anvil compare <run_dir> ...` | Render a Markdown comparison table from run summaries |

The batch scripts in `scripts/` still drive the longer studies:
`run_ablations.py`, `run_calibration.py`, `run_nim_headlines.py`,
`run_parser_benchmark.py`, and `run_agent_eval.py`.

## Deployment

A minimal `Dockerfile` and `fly.toml` are included for read-only FastAPI demo
deployment. The default container uses the fake backend and hash embedder so it
requires no secrets. To deploy a live NIM-backed demo, set `NVIDIA_API_KEY`,
`ANVIL_LLM_BACKEND=nvidia_nim`, `ANVIL_EMBEDDER=sentence_transformer`, and an
explicit `ANVIL_LLM_MODEL` as platform secrets.

## Evaluation results

The current public benchmark is the 100-example
`tests/evaluation/golden_dataset.json` set. The best real baseline artifact is:

| Metric | Score | Notes |
| :--- | ---: | :--- |
| `pass_rate` | 0.950 | 95 / 100 examples pass every threshold |
| `calculation_correctness` | 1.000 | all calculation examples match deterministic ground truth |
| `faithfulness` | 0.977 | most claims are supported by context or deterministic calculation |
| `citation_accuracy` | 1.000 | every emitted citation validates |
| `entity_grounding` | 0.989 | named entities are nearly always grounded |
| `refusal_calibration` | 0.980 | refusal behavior matches expectation on almost all examples |
| `retrieval_recall@10` | 1.000 | all expected refs are retrieved |
| `structural_completeness` | 1.000 | expected refs are surfaced somewhere in the response chain |
| `retrieval_precision@10` | 0.400 | graph expansion intentionally includes adjacent evidence |

Run ID:
`2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline`.

Every metric has a mathematical definition in its docstring
(`src/anvil/evaluation/metrics.py`) and a pass threshold tuned to catch real
failure modes rather than chasing a number.

## Current remaining work

The real-backend validation path is now complete for the fixed pipeline:
NIM health checks, 100-example Meta baseline, three real ablations, Reducto
parser benchmark, private-ASME artifact audit, and a complete 100-example
agent run all have local artifacts.

Remaining gaps are intentionally scoped:

1. DeepSeek V4 Flash and V4 Pro timed out in the targeted 2026-04-26 NIM probe,
   so they were skipped rather than reported.
2. The agent loop has full transcripts but is not competitive with the fixed
   pipeline yet; lookup and cross-reference behavior need tool-policy work.
3. Docker/Fly deployment is optional and is not part of the headline evidence.

## License

MIT. The synthetic standard in `data/synthetic/` is an original work
inspired by public ASME BPVC conventions; no proprietary content.
