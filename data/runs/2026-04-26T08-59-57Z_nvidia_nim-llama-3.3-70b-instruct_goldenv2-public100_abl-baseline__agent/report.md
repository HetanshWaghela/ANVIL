# Run report — `2026-04-26T08-59-57Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent`

**Pass rate: `64.0%`** (64 / 100 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `ccc6a1930f6f…` |
| n_examples | 100 |
| git_sha | `fbdaf85b69317dfdda0e4a9f7d9980e6d152e26a` |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-26T09:22:29Z` |
| finished_at_utc | `2026-04-26T09:22:29Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.820 | 1.000 | 82/100 |
| `retrieval_precision_at_k` | 0.398 | 0.240 | 85/88 |
| `retrieval_recall_at_k` | 0.962 | 0.990 | 84/88 |
| `faithfulness` | 0.790 | 0.800 | 69/88 |
| `citation_accuracy` | 0.966 | 0.800 | 85/88 |
| `entity_grounding` | 0.977 | 1.000 | 84/88 |
| `structural_completeness` | 0.962 | 1.000 | 84/88 |
| `calculation_correctness` | 0.789 | 1.000 | 32/45 |

## Failing examples (worst-first)

### `gold-calc-012` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-calc-034` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-xref-013` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

### `gold-calc-010` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-lookup-003` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.100 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
