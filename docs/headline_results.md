# Headline Results — placeholder

This file is **auto-generated** by `scripts/run_nim_headlines.py` once
a valid `NVIDIA_API_KEY` is provided. Until then it carries this
placeholder so the README link does not 404.

## How to populate

```bash
export NVIDIA_API_KEY=...   # https://build.nvidia.com
uv run anvil nim-check       # confirm the catalog is reachable
uv run python scripts/run_nim_headlines.py --include-fake
```

The script will:

1. Run the **baseline** ablation (full pipeline) against the
   `FakeLLMBackend` (control row).
2. Run the same ablation against each of the three locked NIM
   models — `meta/llama-3.3-70b-instruct`,
   `deepseek-ai/deepseek-v3.1`,
   `nvidia/llama-3.1-nemotron-70b-instruct`.
3. Drop a stamped run directory under `data/runs/` per model.
4. Regenerate this file via `scripts/compare_runs.py` so the table
   below replaces this placeholder.

## Expected shape (filled by the script)

| run | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fake (control) | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ |
| nvidia_nim / llama-3.3-70b-instruct | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ |
| nvidia_nim / deepseek-v3.1 | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ |
| nvidia_nim / llama-3.1-nemotron-70b-instruct | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ | _filled_ |

Each row links to a committed `data/runs/<run_id>/summary.json` and
`report.md`. Numbers in the application copy (see the plan §15) MUST
be footnoted with the run-id that produced them.
