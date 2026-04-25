# Run report — `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

**Pass rate: `53.3%`** (16 / 30 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `0b6db32f3d8c…` |
| n_examples | 30 |
| git_sha | `8bcbf95f956ff517f4f0617366ce79cb38e605a9` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T17:51:28Z` |
| finished_at_utc | `2026-04-25T17:51:28Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.567 | 1.000 | 17/30 |
| `retrieval_precision_at_k` | 0.306 | 0.240 | 14/25 |
| `retrieval_recall_at_k` | 0.560 | 0.990 | 14/25 |
| `faithfulness` | 0.480 | 0.800 | 12/25 |
| `citation_accuracy` | 1.000 | 0.800 | 25/25 |
| `entity_grounding` | 1.000 | 1.000 | 25/25 |
| `structural_completeness` | 0.560 | 1.000 | 14/25 |
| `calculation_correctness` | 0.917 | 1.000 | 11/12 |

## Failing examples (worst-first)

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

### `gold-lookup-005` — _lookup_

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
