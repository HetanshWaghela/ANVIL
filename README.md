# ANVIL

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
uv sync --extra dev                     # install
uv run pytest tests/ -q                 # 134 tests, ~0.5s
uv run python scripts/ingest.py         # parse SPES-1 → KG + vector index
uv run python scripts/evaluate.py       # run 30-example golden dataset
uv run python scripts/demo.py           # interactive query demo (Rich UI)
uv run uvicorn anvil.api.app:create_app --factory   # FastAPI
```

Everything defaults to a deterministic, offline-capable backend
(`DeterministicHashEmbedder` + `FakeLLMBackend`) so CI and local dev require no
model downloads or API keys. **Both defaults log a WARNING** at startup —
a prod deploy that forgot to configure a real backend sees this
immediately, rather than shipping silent fake responses.

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

## Evaluation results

Running the 30-example golden dataset (`tests/evaluation/golden_dataset.json`):

| Metric                     | Score | Notes                                   |
| :------------------------- | ----: | :-------------------------------------- |
| `pass_rate` (all metrics)  | 0.967 | 29 / 30 examples pass every threshold   |
| `calculation_correctness`  | 1.000 | all numeric outputs match ground truth  |
| `faithfulness`             | 1.000 | every claim supported by context or calc|
| `citation_accuracy`        | 1.000 | all citations validate                   |
| `entity_grounding`         | 1.000 | every named entity appears in context   |
| `refusal_calibration`      | 1.000 | refusal matches expectation everywhere  |
| `retrieval_recall@10`      | 0.987 | 1 example misses an expected paragraph  |
| `structural_completeness`  | 0.987 | every expected ref surfaced             |
| `retrieval_precision@10`   | 0.368 | threshold scales with `len(expected_refs)`|

Every metric has a mathematical definition in its docstring
(`src/anvil/evaluation/metrics.py`) and a pass threshold tuned to catch
real failure modes rather than chasing a number.

## License

MIT. The synthetic standard in `data/synthetic/` is an original work
inspired by public ASME BPVC conventions; no proprietary content.
