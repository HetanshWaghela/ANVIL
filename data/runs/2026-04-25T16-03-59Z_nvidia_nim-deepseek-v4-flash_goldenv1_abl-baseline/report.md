# Run report — `2026-04-25T16-03-59Z_nvidia_nim-deepseek-v4-flash_goldenv1_abl-baseline`

**Pass rate: `86.7%`** (26 / 30 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `deepseek-ai/deepseek-v4-flash` |
| ablation | `baseline` |
| ablation_config.name | `baseline` |
| ablation_config.retrieval_mode | `hybrid` |
| ablation_config.use_citation_enforcer | `True` |
| ablation_config.use_pinned_data | `True` |
| ablation_config.use_refusal_gate | `True` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `0b6db32f3d8c…` |
| n_examples | 30 |
| git_sha | `8bcbf95f956ff517f4f0617366ce79cb38e605a9` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T16:07:31Z` |
| finished_at_utc | `2026-04-25T16:07:31Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.967 | 1.000 | 29/30 |
| `retrieval_precision_at_k` | 0.388 | 0.240 | 25/25 |
| `retrieval_recall_at_k` | 0.987 | 0.990 | 24/25 |
| `faithfulness` | 0.930 | 0.800 | 22/25 |
| `citation_accuracy` | 1.000 | 0.800 | 25/25 |
| `entity_grounding` | 0.960 | 1.000 | 24/25 |
| `structural_completeness` | 0.987 | 1.000 | 24/25 |
| `calculation_correctness` | 1.000 | 1.000 | 12/12 |

## Failing examples (worst-first)

### `gold-edge-003` — _edge_case_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.300 | 0.160 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

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

### `gold-xref-001` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.500 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 0.667 | 0.990 | ✗ |
| `faithfulness` | 0.500 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 0.667 | 1.000 | ✗ |

### `gold-xref-004` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.100 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.750 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
