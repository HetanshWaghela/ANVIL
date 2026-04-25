# Run report — `2026-04-25T18-20-58Z_fake_goldenv2-public100_abl-baseline`

**Pass rate: `94.0%`** (94 / 100 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `fake` |
| ablation | `baseline` |
| ablation_config.name | `baseline` |
| ablation_config.retrieval_mode | `hybrid` |
| ablation_config.use_citation_enforcer | `True` |
| ablation_config.use_pinned_data | `True` |
| ablation_config.use_refusal_gate | `True` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `2028f9f05c37…` |
| n_examples | 100 |
| git_sha | `4e0ed57b9dce9066bfd5b8b74166bc1ef7bd8f27` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T18:20:58Z` |
| finished_at_utc | `2026-04-25T18:20:58Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.960 | 1.000 | 96/100 |
| `retrieval_precision_at_k` | 0.380 | 0.240 | 87/88 |
| `retrieval_recall_at_k` | 0.991 | 0.990 | 86/88 |
| `faithfulness` | 0.989 | 0.800 | 87/88 |
| `citation_accuracy` | 1.000 | 0.800 | 88/88 |
| `entity_grounding` | 1.000 | 1.000 | 88/88 |
| `structural_completeness` | 0.991 | 1.000 | 86/88 |
| `calculation_correctness` | 1.000 | 1.000 | 45/45 |

## Failing examples (worst-first)

### `gold-ood-006` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-ood-010` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-ood-011` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-xref-015` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.200 | 0.160 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-xref-014` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.100 | 0.160 | ✗ |
| `retrieval_recall_at_k` | 0.500 | 0.990 | ✗ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 0.500 | 1.000 | ✗ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
