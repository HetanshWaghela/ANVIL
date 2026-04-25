# Run report — `2026-04-25T13-10-10Z_nvidia_nim-llama-3.3-nemotron-super-49b-v1.5_goldenv1_abl-baseline`

**Pass rate: `86.7%`** (26 / 30 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| ablation | `baseline` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `0b6db32f3d8c…` |
| n_examples | 30 |
| git_sha | `d5742e2a71ad943bd9e3a8b072e80fd5d9215b38` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T13:32:07Z` |
| finished_at_utc | `2026-04-25T13:32:07Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | 30/30 |
| `retrieval_precision_at_k` | 0.368 | 0.240 | 25/25 |
| `retrieval_recall_at_k` | 0.987 | 0.990 | 24/25 |
| `faithfulness` | 0.980 | 0.800 | 24/25 |
| `citation_accuracy` | 1.000 | 0.800 | 25/25 |
| `entity_grounding` | 0.952 | 0.900 | 23/25 |
| `structural_completeness` | 0.987 | 1.000 | 24/25 |
| `calculation_correctness` | 1.000 | 1.000 | 12/12 |

## Failing examples (worst-first)

### `gold-lookup-005` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.200 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 0.000 | 0.900 | ✗ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-xref-004` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.100 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.500 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-xref-001` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 0.667 | 0.990 | ✗ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 0.667 | 1.000 | ✗ |

### `gold-calc-009` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 0.800 | 0.900 | ✗ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 1.000 | 1.000 | ✓ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
