# Evaluation Methodology

This document explains exactly *what* ANVIL measures, *why* those measures
map to real failure modes, and *how* the numbers are computed. Every
metric implementation in `src/anvil/evaluation/metrics.py` carries a
mathematical definition in its docstring; this document collects them in
one place and explains the tradeoffs.

## What we are measuring

Generic RAG benchmarks reward fluency and surface-level relevance. None of
those signals matter in regulated engineering — a fluent answer with a
hallucinated stress value is a design liability. ANVIL's metrics target
the failure modes that actually matter:

| Failure mode | Metric |
| :--- | :--- |
| Hallucinated stress value | `calculation_correctness` (vs. pinned ground truth) |
| Fabricated paragraph reference | `entity_grounding` |
| Quote that doesn't appear in the cited source | `citation_accuracy` |
| Missing cross-reference (e.g. cited UG-27 but not UW-12 or Table M-1) | `structural_completeness` |
| Confident answer to an out-of-domain query | `refusal_calibration` |
| Numeric drift across answer prose | `faithfulness` |
| Misranked or missing required evidence | `retrieval_precision_at_k`, `retrieval_recall_at_k` |

## Evidence tiers

ANVIL separates evaluation evidence into two tiers:

Application/headline claims use the production backend (`nvidia_nim` or
another OpenAI-compatible provider). The 285-test suite is a separate
regression layer that locks every behavior the metrics layer scores; it
runs offline so CI can guard against regressions without burning provider
quota.

## Golden dataset

`tests/evaluation/golden_dataset.json` contains 100 public SPES-1 examples
across five categories:

| Category | Count | Purpose |
| :--- | :--- | :--- |
| `calculation` | 34 | Worked answers with `expected_values` for every step. Anchors `calculation_correctness`. |
| `lookup` | 20 | "What is the allowable stress of SM-516 Gr 70 at 300°C?" — single-value retrieval. |
| `cross_reference` | 20 | "What inputs are needed for A-27(c)(1)?" — exercises graph expansion. |
| `out_of_domain` | 12 | Unsupported materials, real-ASME requests, unrelated prompts, and missing inputs — must trigger refusal. |
| `edge_case` | 14 | Temperature interpolation, exact tabulated temperatures, outside-radius formulas, low joint efficiency, and high pressure. |

Each `GoldenExample` carries:

* `expected_paragraph_refs` — the canonical refs that retrieval must
  surface in the top-10 (used by precision/recall).
* `expected_values` — typed `dict[str, float]` keyed by canonical metric
  names (`S_mpa`, `E`, `t_min_mm`, `t_design_mm`, `t_nominal_mm`,
  `mawp_mpa`). Compared by `calculation_correctness`.
* `expected_refusal` — boolean for `refusal_calibration`.
* `numeric_tolerance` — per-example bound (default 0.02, tightened to
  0.001 for the worked calculation examples).

Calculation expected values in the expanded public set were generated from
`CalculationEngine`, not by hand, then written into JSON as reviewed expected
outputs. This keeps the benchmark deterministic while avoiding query-specific
answer branches in `src/`.

Licensed ASME validation uses a separate local-only path: datasets and parsed
standards stay under `data/private/`, outputs stay under `data/private_runs/`,
and only sanitized aggregate metrics may be shared. See `docs/private_asme.md`.

## Metric definitions

### `retrieval_precision_at_k`

```
P@K = |retrieved[:k] ∩ relevant| / |retrieved[:k]|
```

A retrieved chunk counts as relevant if its `paragraph_ref` matches an
expected_ref under the sub-paragraph boundary rule (`A-27(c)(1)` covers
`A-27`, but `A-2` does NOT cover `A-27`).

**Threshold.** Computed dynamically as
`min(0.3, max(0.08, 0.8·|expected|/k))`. The rationale: a query with one
expected ref in top-10 should not be penalized for "polluting" the rest
of the slate with adjacent content; that's exactly what hybrid retrieval
plus graph expansion *should* do.

### `retrieval_recall_at_k`

```
R@K = |found_expected| / |expected|
```

Strict 99% threshold — every expected paragraph must appear in the top-K.
The graph-expansion step is what makes this achievable: when the user
asks for "the wall thickness formula" the formula chunk pulls in M-1 and
B-12 along the `REQUIRES` edges.

### `calculation_correctness`

```
correctness = matched / total       (exact match required: threshold = 1.0)
```

For each expected key (`S_mpa`, `E`, `t_min_mm`, …) we look up the
matching `CalculationStep.result_key` and compare with the per-example
`numeric_tolerance`. The match is keyed off `StepKey` enum, NOT off
`description` strings — see ADR-004.

This is the metric that catches an off-by-an-interpolation bug or a
formula that confused inside vs. outside radius.

### `citation_accuracy`

```
accuracy = valid_citations / total_citations
```

Computed by `validate_citations`. A citation is valid if either:

1. Its `source_element_id` is in the retrieved chunks AND
   `paragraph_refs_compatible(citation.ref, source.ref)` AND
   `quoted_text` substantially matches `source.content`; or
2. Its `paragraph_ref` is canonical AND the cited element resolves in the
   parsed standard via the threaded `element_index` AND `quoted_text`
   substantially matches the resolved element's content.

The "substantially matches" predicate is exact-substring OR ≥60% word
overlap (a permissive band that tolerates whitespace / punctuation drift
without admitting fabricated quotes).

### `faithfulness` (RAGAS-style)

```
faithfulness = supported_sentences / total_sentences
```

Decompose the answer into sentences. A sentence is supported if any
numeric token appears in the retrieved context or in the deterministic
calculation steps' results/inputs, OR (for non-numeric sentences) if at
least 40% of substantive words overlap with the retrieved context.

This is a deterministic proxy for the NLI-based RAGAS metric. It's
intentionally cheap so it can run on every CI job.

### `entity_grounding`

```
grounding = grounded_entities / total_entities
```

For every named entity (paragraph ref like `A-27(c)(1)`, table ref like
`Table M-1`, material spec like `SM-516 Gr 70`) we require it to appear
in the retrieved context OR be a canonical SPES-1 ref. Threshold 0.9.

### `structural_completeness`

```
completeness = covered_expected_refs / |expected_refs|
```

For calculation queries, every `expected_paragraph_ref` must appear in
either (a) `response.citations`, (b) `response.calculation_steps[].citation`,
or (c) `retrieved_chunks`. Threshold 1.0 (every required ref must be
covered somewhere in the response chain).

### `refusal_calibration`

Strict 0/1: did the actual refusal decision match `example.expected_refusal`?

This is the metric that catches "system confidently answered an OOD
query" and "system refused a legitimate query."

## Aggregation and pass criteria

`EvaluationRunner.run()` returns:

* **`per_example`** — list of `EvaluationResult` with the metric vector.
* **`aggregate`** — mean of each metric across all examples.
* **`pass_rate`** — fraction of examples where every metric passed its
  threshold.

The integration test `test_run_full_eval_pass_rate` asserts
`pass_rate >= 0.7` on the bundled offline regression fixtures; production
deploys should set thresholds higher and treat regressions
(`evaluation/regression.py`) as CI-blocking.

## Run artifacts and resume

Every full evaluation writes a stamped directory under `data/runs/<run_id>/`
with `manifest.json`, `summary.json`, `per_example.json`, `raw_responses.jsonl`,
and `report.md`. Real-provider runs also write `checkpoint.json` as examples
complete, so interrupted evaluations can resume without replaying completed
requests:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval \
  --backend nvidia_nim \
  --model meta/llama-3.3-70b-instruct \
  --ablation baseline \
  --min-pass-rate 0.0 \
  --resume-run <run_id>
```

The NIM backend applies per-key request throttling. If fallback keys are
configured in `.env`, retryable provider errors try each configured key before
the runner enters cooldown. Manifests redact all configured secret environment
variables.

## Regression detection

`evaluation/regression.compare_runs(baseline, current, tolerance=0.01)`
flags any aggregate metric that dropped by more than `tolerance`. A real
deploy keeps the most recent passing run as `baseline.json` and runs the
comparison on every PR; a non-empty `regressed_metrics` list blocks merge.

The meta-test
`test_regression_comparison_detects_drops` proves the comparator catches
a deliberate 0.2-magnitude drop, so the regression suite cannot itself
silently regress.

## What we deliberately do NOT measure

* **BLEU / ROUGE** — measures n-gram overlap with a reference text.
  Useless for engineering answers where two correct answers can differ
  in wording but agree numerically.
* **Token-level perplexity** — fluency proxy that punishes terse
  cited-prose answers.
* **MMLU-style multiple choice** — reduces an open-ended compliance task
  to a 4-option recognizer.

These aren't bad metrics in general; they are bad metrics *for this
domain*. The metrics we ship are domain-appropriate, deterministic, and
each maps to a concrete failure mode an auditor would care about.
