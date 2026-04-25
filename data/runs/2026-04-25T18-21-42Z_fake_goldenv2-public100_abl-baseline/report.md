# Run report — `2026-04-25T18-21-42Z_fake_goldenv2-public100_abl-baseline`

**Pass rate: `97.0%`** (97 / 100 examples passed every metric threshold).

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
| dataset_hash | `aeeb382af8ec…` |
| n_examples | 100 |
| git_sha | `4e0ed57b9dce9066bfd5b8b74166bc1ef7bd8f27` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T18:21:42Z` |
| finished_at_utc | `2026-04-25T18:21:42Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.980 | 1.000 | 98/100 |
| `retrieval_precision_at_k` | 0.384 | 0.240 | 88/88 |
| `retrieval_recall_at_k` | 1.000 | 0.990 | 88/88 |
| `faithfulness` | 0.989 | 0.800 | 87/88 |
| `citation_accuracy` | 0.997 | 0.800 | 87/88 |
| `entity_grounding` | 1.000 | 1.000 | 88/88 |
| `structural_completeness` | 1.000 | 1.000 | 88/88 |
| `calculation_correctness` | 1.000 | 1.000 | 45/45 |

## Failing examples (worst-first)

### `gold-ood-011` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-xref-011` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.400 | 0.240 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 0.000 | 0.800 | ✗ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-xref-015` — _cross_reference_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.300 | 0.160 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 0.750 | 0.800 | ✗ |
| `entity_grounding` | 1.000 | 0.900 | ✓ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
