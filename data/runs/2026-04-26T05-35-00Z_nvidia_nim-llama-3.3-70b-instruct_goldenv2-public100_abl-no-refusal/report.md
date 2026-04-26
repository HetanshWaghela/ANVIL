# Run report — `2026-04-26T05-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-refusal`

**Pass rate: `95.0%`** (95 / 100 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `no-refusal` |
| ablation_config.name | `no-refusal` |
| ablation_config.retrieval_mode | `hybrid` |
| ablation_config.use_citation_enforcer | `True` |
| ablation_config.use_pinned_data | `True` |
| ablation_config.use_refusal_gate | `False` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `ccc6a1930f6f…` |
| n_examples | 100 |
| git_sha | `d1a500747d134ec7526f8b0c4de7773ad6ab7414` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-26T06:05:41Z` |
| finished_at_utc | `2026-04-26T06:05:42Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.970 | 1.000 | 97/100 |
| `retrieval_precision_at_k` | 0.400 | 0.240 | 88/88 |
| `retrieval_recall_at_k` | 1.000 | 0.990 | 88/88 |
| `faithfulness` | 0.994 | 0.800 | 87/88 |
| `citation_accuracy` | 0.988 | 0.800 | 87/88 |
| `entity_grounding` | 0.981 | 1.000 | 86/88 |
| `structural_completeness` | 1.000 | 1.000 | 88/88 |
| `calculation_correctness` | 1.000 | 1.000 | 45/45 |

## Failing examples (worst-first)

### `gold-ood-006` — _out_of_domain_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |

### `gold-lookup-010` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.200 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 0.000 | 0.900 | ✗ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-lookup-008` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.000 | 1.000 | ✗ |
| `retrieval_precision_at_k` | 0.200 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 1.000 | 0.800 | ✓ |
| `entity_grounding` | 0.333 | 0.900 | ✗ |
| `structural_completeness` | 1.000 | 1.000 | ✓ |

### `gold-lookup-001` — _lookup_

| metric | value | threshold | passed |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | ✓ |
| `retrieval_precision_at_k` | 0.200 | 0.080 | ✓ |
| `retrieval_recall_at_k` | 1.000 | 0.990 | ✓ |
| `faithfulness` | 1.000 | 0.800 | ✓ |
| `citation_accuracy` | 0.000 | 0.800 | ✗ |
| `entity_grounding` | 1.000 | 1.000 | ✓ |
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

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
