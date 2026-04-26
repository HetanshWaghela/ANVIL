# Run report — `2026-04-26T08-39-30Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent`

**Pass rate: `65.0%`** (65 / 100 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `ccc6a1930f6f…` |
| n_examples | 100 |
| git_sha | `a0fb040ec7ec0f2bbd59074f57656d5046b3373f` |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-26T08:57:40Z` |
| finished_at_utc | `2026-04-26T08:57:40Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.840 | 1.000 | 84/100 |
| `retrieval_precision_at_k` | 0.361 | 0.240 | 69/88 |
| `retrieval_recall_at_k` | 0.784 | 0.990 | 69/88 |
| `faithfulness` | 0.682 | 0.800 | 60/88 |
| `citation_accuracy` | 0.989 | 0.800 | 87/88 |
| `entity_grounding` | 0.955 | 1.000 | 80/88 |
| `structural_completeness` | 0.909 | 1.000 | 80/88 |
| `calculation_correctness` | 0.856 | 1.000 | 36/45 |

## Failing examples (worst-first)

### `gold-lookup-002` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

### `gold-lookup-003` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

### `gold-lookup-004` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

### `gold-lookup-011` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

### `gold-lookup-014` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
