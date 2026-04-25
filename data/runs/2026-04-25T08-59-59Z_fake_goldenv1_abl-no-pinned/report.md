# Run report — `2026-04-25T08-59-59Z_fake_goldenv1_abl-no-pinned`

**Pass rate: `56.7%`** (17 / 30 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `fake` |
| ablation | `no-pinned` |
| ablation_config.name | `no-pinned` |
| ablation_config.retrieval_mode | `hybrid` |
| ablation_config.use_citation_enforcer | `True` |
| ablation_config.use_pinned_data | `False` |
| ablation_config.use_refusal_gate | `True` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `0b6db32f3d8c…` |
| n_examples | 30 |
| git_sha | `de54f4fcd526144ad89ff6f26dbbeac10cd8e346` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T09:00:00Z` |
| finished_at_utc | `2026-04-25T09:00:00Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | 30/30 |
| `retrieval_precision_at_k` | 0.368 | 0.240 | 25/25 |
| `retrieval_recall_at_k` | 0.987 | 0.990 | 24/25 |
| `faithfulness` | 1.000 | 0.800 | 25/25 |
| `citation_accuracy` | 1.000 | 0.800 | 25/25 |
| `entity_grounding` | 1.000 | 0.900 | 25/25 |
| `structural_completeness` | 0.987 | 1.000 | 24/25 |
| `calculation_correctness` | 0.000 | 1.000 | 0/12 |

## Failing examples (worst-first)

### `gold-calc-001` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-calc-002` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-calc-004` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-calc-005` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

### `gold-calc-006` — _calculation_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |
| `calculation_correctness` | 0.000 | 1.000 | ✗ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
