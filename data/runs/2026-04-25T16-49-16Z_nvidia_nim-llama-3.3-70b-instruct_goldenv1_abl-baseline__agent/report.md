# Run report — `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

**Pass rate: `50.0%`** (5 / 10 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `95e165a1de4f…` |
| n_examples | 10 |
| git_sha | `8bcbf95f956ff517f4f0617366ce79cb38e605a9` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T16:54:05Z` |
| finished_at_utc | `2026-04-25T16:54:05Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 0.500 | 1.000 | 5/10 |
| `retrieval_precision_at_k` | 0.250 | 0.240 | 5/10 |
| `retrieval_recall_at_k` | 0.500 | 0.990 | 5/10 |
| `faithfulness` | 0.500 | 0.800 | 5/10 |
| `citation_accuracy` | 1.000 | 0.800 | 10/10 |
| `entity_grounding` | 1.000 | 1.000 | 10/10 |
| `structural_completeness` | 0.500 | 1.000 | 5/10 |
| `calculation_correctness` | 0.500 | 1.000 | 5/10 |

## Failing examples (worst-first)

### `gold-calc-006` — _calculation_

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

### `gold-calc-007` — _edge_case_

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

### `gold-calc-008` — _calculation_

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

### `gold-calc-009` — _calculation_

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

### `gold-calc-010` — _calculation_

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

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
