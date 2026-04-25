# Run report — `2026-04-25T16-01-30Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

**Pass rate: `100.0%`** (3 / 3 examples passed every metric threshold).

## Run configuration

| field | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| ablation | `baseline-agent` |
| dataset | `tests/evaluation/golden_dataset.json` |
| dataset_hash | `c9f5aa8c12e7…` |
| n_examples | 3 |
| git_sha | `8bcbf95f956ff517f4f0617366ce79cb38e605a9` |
| git_dirty | **yes — manifest is from a dirty worktree** |
| git_branch | `main` |
| python | `3.14.4` |
| started_at_utc | `2026-04-25T16:02:29Z` |
| finished_at_utc | `2026-04-25T16:02:29Z` |

## Aggregate metrics

| metric | mean | threshold (per-example) | per-example pass |
| :--- | ---: | ---: | :---: |
| `refusal_calibration` | 1.000 | 1.000 | 3/3 |
| `retrieval_precision_at_k` | 0.500 | 0.240 | 3/3 |
| `retrieval_recall_at_k` | 1.000 | 0.990 | 3/3 |
| `faithfulness` | 1.000 | 0.800 | 3/3 |
| `citation_accuracy` | 1.000 | 0.800 | 3/3 |
| `entity_grounding` | 1.000 | 1.000 | 3/3 |
| `structural_completeness` | 1.000 | 1.000 | 3/3 |
| `calculation_correctness` | 1.000 | 1.000 | 3/3 |

## Failing examples

_None — every example passed every metric threshold._

## Reproducibility

To reproduce, check out the recorded git sha and re-run with the same env vars (the manifest captures the allowlisted set with secret values redacted).

## Artifacts

- [`manifest.json`](manifest.json)
- [`per_example.json`](per_example.json)
- [`summary.json`](summary.json)

Raw per-request logs (`raw_responses.jsonl`, `prompts.jsonl`, `request_log.jsonl`) are produced locally but gitignored — see `.gitignore` and the plan §3.1 storage policy.
