# Run report — `2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent`

**Pass rate: `46.0%`** (46 / 100 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `ccc6a1930f6f…` |
| n_examples | 100 |
| git_sha | `d1a500747d134ec7526f8b0c4de7773ad6ab7414` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-26T07:55:30Z` |
| finished_at_utc | `2026-04-26T07:55:31Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.610 | 1.000 | 61/100 |
| `retrieval_precision_at_k` | 0.360 | 0.240 | 70/88 |
| `retrieval_recall_at_k` | 0.786 | 0.990 | 68/88 |
| `faithfulness` | 0.511 | 0.800 | 45/88 |
| `citation_accuracy` | 0.989 | 0.800 | 87/88 |
| `entity_grounding` | 0.983 | 1.000 | 85/88 |
| `structural_completeness` | 0.843 | 1.000 | 73/88 |
| `calculation_correctness` | 0.878 | 1.000 | 36/45 |

## Failing examples (worst-first)

### `gold-ood-006` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-ood-010` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-calc-015` — _calculation_

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

### `gold-lookup-001` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.000 | — | ✗ |
| `retrieval_recall_at_k` | 0.000 | 0.990 | ✗ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 0.000 | 1.000 | ✗ |

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

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
