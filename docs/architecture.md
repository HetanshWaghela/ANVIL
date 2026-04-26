# Architecture

This document describes ANVIL's architecture as a set of layered, typed
contracts. The system is small enough to fit in one diagram and large enough
that the seams between layers matter — every layer above the parsing layer
treats the layer below as immutable, typed input.

## High-level pipeline

```
                ┌─────────────────────────────────────────────┐
                │  data/synthetic/standard.md (SPES-1)        │
                │  data/synthetic/*.json (pinned ground truth)│
                └──────────────────┬──────────────────────────┘
                                   │
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  parsing/        markdown_parser, table_extractor,           │
   │                  formula_extractor, section_linker           │
   │                  → list[DocumentElement]                     │
   └──────────────────┬──────────────────────────────┬────────────┘
                      │                              │
                      ▼                              ▼
   ┌──────────────────────────────────┐  ┌─────────────────────────┐
   │  knowledge/                      │  │  retrieval/             │
   │   graph_builder, graph_store,    │  │   embedder,             │
   │   node_types, edge_types         │  │   vector_store,         │
   │   → networkx.DiGraph             │  │   bm25 index,           │
   │                                  │  │   graph_retriever,      │
   │                                  │  │   hybrid_retriever      │
   └─────────┬────────────────────────┘  └────────────┬────────────┘
             │                                        │
             └──────────────────┬─────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  generation/                                                 │
   │   refusal_gate  → calculation_engine → prompt_builder →      │
   │   llm_backend (NVIDIA NIM / Instructor / OpenAI-compat) →    │
   │   citation_enforcer                                          │
   │   → AnvilResponse (Pydantic, with Citations)                 │
   └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  evaluation/                                                 │
   │   metrics, dataset, runner, regression                       │
   │                                                              │
   │  api/        FastAPI routes — /query /calculate /materials   │
   └──────────────────────────────────────────────────────────────┘
```

## Module map

| Layer | Module | Responsibility |
| :--- | :--- | :--- |
| Schemas | `src/anvil/schemas/` | The typed spine — every cross-layer object is a Pydantic model. No business logic. |
| Parsing | `src/anvil/parsing/` | Markdown / PDF → `DocumentElement` with tables, formulas, cross-references. |
| Knowledge | `src/anvil/knowledge/` | `DocumentElement`s → typed `networkx.DiGraph` with node + edge types. |
| Pinned | `src/anvil/pinned/` | Hand-verified material properties, joint efficiencies, and Decimal-backed formulas. **Calculations always source numbers here.** |
| Retrieval | `src/anvil/retrieval/` | BM25 + vector (`sqlite-vec`) + graph expansion fused via Reciprocal Rank Fusion. |
| Generation | `src/anvil/generation/` | Refusal gate → calc engine → prompt builder → LLM → citation enforcer. |
| Evaluation | `src/anvil/evaluation/` | Domain-specific metrics (faithfulness, citation accuracy, calc correctness, refusal calibration) over a 30+ example golden dataset. |
| API | `src/anvil/api/` | FastAPI app + structured request logging. |

## Provenance contract

Every numerical claim in the system is traceable through a chain of typed
records:

```
AnvilResponse.calculation_steps[i]
   .result          : float               # the number
   .result_key      : StepKey             # stable enum (NEVER a description string)
   .formula         : str                 # human-readable formula
   .inputs[symbol]  : InputValue          # symbol, value, unit, source
       .citation    : Citation | None     # provenance for this input
   .citation        : Citation            # provenance for the formula itself
       .source_element_id : str           # real DocumentElement id
       .paragraph_ref     : str           # canonical SPES-1 ref
       .quoted_text       : str           # substring of source content
       .page_number       : int
```

Two enforcement mechanisms keep this honest:

1. **`CitationBuilder.from_elements`** (in `calculation_engine.py`) refuses
   to fabricate a citation. If a paragraph or material row is not present
   in the parsed standard, it raises `CalculationError` rather than
   constructing a citation from a generic section intro.
2. **`citation_enforcer.validate_citations`** runs after every generation.
   For every citation it confirms either (a) the cited element is in the
   retrieved chunks AND `quoted_text` is a substring of that chunk's
   content, or (b) the `paragraph_ref` is canonical AND the cited element
   exists in the parsed standard AND `quoted_text` is a substring of that
   element's content. The "(b) without `quoted_text` validation" path that
   used to exist was removed during the audit — it was a hallucination
   escape hatch.

## Refusal gate

`generation/refusal_gate.py` runs **before** the LLM, with two distinct
checks:

* `should_refuse(query, chunks)` — refuses on (1) low max relevance, (2) a
  generic non-`SM-` material mention, (3) an `SM-` material not in pinned
  data, (4) a temperature exceeding the material's `max_temp_c`.
* `check_calculation_evidence(chunks)` — when an NL query parses into a
  calculation request but the LLM would derive inputs from retrieved
  context, refuses unless retrieval surfaced **all** of the formula
  paragraph (A-27), the stress table (M-1), and the joint table (B-12).
  Bypassed when an API caller passes explicit `CalculationInputs`, because
  the engine then sources values from pinned data deterministically.

## Calculation engine

`generation/calculation_engine.py` is the only place that produces numbers.

* All arithmetic uses `decimal.Decimal` via `pinned/formulas.py`.
* `_formula_for(component)` selects the formula triple
  `(applicability_fn, thickness_fn, mawp_fn)` for the component.
* Applicability is checked twice: pressure (`P ≤ 0.385·S·E` cyl /
  `0.665·S·E` sphere) and geometry (`t ≤ R/2`). Either failure raises
  `CalculationError` — we never silently apply thin-wall formulas in the
  thick-wall regime.
* Each `CalculationStep` carries a `result_key: StepKey` enum so that
  downstream consumers (metrics, UI, auditors) match on stable identifiers,
  not on `description` strings.

## Retrieval

* **BM25** (`bm25s` if installed; pure-Python fallback otherwise — fail-loud
  if `bm25s` errors at index time).
* **Vector** (`sqlite-vec` if installed; NumPy fallback only on
  `ImportError`. Any other failure raises — silent degradation is forbidden).
* **Graph expansion** — given the fused top-K, follows graph edges
  (`REQUIRES`, `REFERENCES`, `DEFINES`, …) up to 1 hop. This is what makes
  "wall thickness formula" pull in `Table M-1` and `Table B-12` even if
  vector + BM25 alone miss them.
* **Reciprocal Rank Fusion** with `k=60`, scores normalized so the top
  chunk scores 1.0. The score is then capped by `top_vec_sim × overlap_frac`
  so out-of-domain queries collapse to near-zero scores.

## Evaluation

Each metric in `evaluation/metrics.py` carries an explicit mathematical
definition in its docstring. Notable choices:

* **`retrieval_precision_at_k`** — uses a query-dependent threshold
  `min(0.3, max(0.08, 0.8·|expected|/k))` so single-target lookup queries
  aren't punished for "polluting" precision with their other top-10
  results.
* **`calculation_correctness`** — keys on `StepKey` enum values, NOT on
  step `description` strings. Renaming a step description never silently
  breaks a metric.
* **`refusal_calibration`** — strict 0/1 score per example.

## Backends

| Env value | Backend | Notes |
| :--- | :--- | :--- |
| `nvidia_nim` (recommended for production) | NVIDIA NIM via `OpenAICompatibleBackend` | Requires `NVIDIA_API_KEY`; supports per-key throttling and fallback-key rotation. |
| `openai_compatible` | Any OpenAI-protocol endpoint | Requires `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `ANVIL_LLM_MODEL`. |
| `instructor` | `instructor.from_provider` | Requires `ANVIL_LLM_MODEL` like `openai/gpt-4o-mini`. |

Every misconfigured selection (missing env var, unknown value, missing key)
raises `GenerationError` at construction time — a deploy that forgot to set
secrets fails immediately rather than silently degrading.
