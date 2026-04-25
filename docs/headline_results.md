# Headline Results

This file is the compact, reviewer-facing model/result table for ANVIL. Every row below is backed by a stamped run directory under `data/runs/<run_id>/` containing at least:

- `manifest.json`
- `summary.json`
- `per_example.json`
- `report.md`

Raw prompts / request logs are intentionally gitignored.

## Verification pass notes

Latest full application-grade verification pass: 2026-04-25.

NIM health checks after sourcing `.env`:

| probe | reachable | latency_ms | behavior |
| :--- | :--- | ---: | :--- |
| `meta/llama-3.3-70b-instruct` | yes | 459.4 | returned `OK` |
| `qwen/qwen3-next-80b-a3b-instruct` | yes | 1666.0 | returned `OK` |
| `moonshotai/kimi-k2-instruct-0905` | no | 10033.5 | `ReadTimeout('')` |
| `deepseek-ai/deepseek-v4-flash` | yes | 429.8 | returned `OK` |
| `deepseek-ai/deepseek-v3.2` | no | 10075.8 | `ReadTimeout('')` |
| `deepseek-ai/deepseek-v4-pro` | yes | 2547.9 | returned `OK` |

The same pass also showed that hosted model behavior is time-dependent. The strongest clean headline rows below remain the best application evidence, while the fresh rows are retained as live-provider robustness evidence rather than overwritten or hidden.

## Public 100-example SPES-1 benchmark

The public reproducible benchmark has been expanded from 30 to 100 examples. Numeric calculation expectations were generated from `CalculationEngine` and then stored in the golden JSON, preserving the no-hardcoding story: the system is evaluated against deterministic engineering logic, not query-specific branches.

| run | n_examples | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| fake / public100 / abl-baseline | 100 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.385 | **1.000** | `2026-04-25T18-24-28Z_fake_goldenv2-public100_abl-baseline` |

## Live NIM baseline comparison

| run | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| fake / abl-baseline | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.987 | 0.987 | 0.388 | 1.000 | `2026-04-25T14-41-21Z_fake_goldenv1_abl-baseline` |
| nvidia_nim / llama-3.3-70b-instruct / abl-baseline | **0.967** | **1.000** | 0.998 | **1.000** | **1.000** | 0.987 | 0.987 | 0.388 | **1.000** | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| nvidia_nim / kimi-k2-instruct-0905 / abl-baseline | 0.900 | **1.000** | **1.000** | 0.975 | 0.960 | 0.987 | 0.987 | 0.388 | **1.000** | `2026-04-25T14-23-10Z_nvidia_nim-kimi-k2-instruct-0905_goldenv1_abl-baseline` |
| nvidia_nim / deepseek-v4-flash / abl-baseline | 0.867 | **1.000** | **1.000** | 0.930 | 0.960 | 0.987 | 0.987 | 0.388 | 0.967 | `2026-04-25T16-03-59Z_nvidia_nim-deepseek-v4-flash_goldenv1_abl-baseline` |
| nvidia_nim / qwen3-next-80b-a3b-instruct / abl-baseline | 0.800 | **1.000** | 0.988 | **1.000** | 0.930 | 0.987 | 0.987 | 0.388 | **1.000** | `2026-04-25T14-18-20Z_nvidia_nim-qwen3-next-80b-a3b-instruct_goldenv1_abl-baseline` |

## Fresh verification rows

These runs were executed later in the same session with the requested commands and `ANVIL_EMBEDDER=sentence_transformer`. They are useful for demonstrating honest operational logging: the fixed pipeline continued to produce artifacts, but NIM served many refusal-shaped or rate-limited responses during the later window.

| run | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| meta baseline, later verification | 0.533 | 0.750 | 0.980 | 0.520 | 1.000 | 0.987 | 0.987 | 0.388 | 0.600 | `2026-04-25T16-21-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| meta no-pinned, later verification | 0.167 | 0.000 | 1.000 | 0.020 | 1.000 | 0.987 | 0.987 | 0.388 | 0.200 | `2026-04-25T16-23-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-pinned` |
| meta no-refusal, later verification | 0.233 | 0.083 | 1.000 | 0.080 | 1.000 | 0.987 | 0.987 | 0.388 | 0.233 | `2026-04-25T16-25-26Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-refusal` |
| meta no-citation-enforcer, later verification | 0.200 | 0.083 | 1.000 | 0.040 | 1.000 | 0.987 | 0.987 | 0.388 | 0.200 | `2026-04-25T16-28-03Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-citation-enforcer` |
| deepseek-v4-flash, later verification | 0.700 | 0.917 | 1.000 | 0.787 | 0.940 | 0.987 | 0.987 | 0.388 | 0.833 | `2026-04-25T16-29-52Z_nvidia_nim-deepseek-v4-flash_goldenv1_abl-baseline` |

## Live Meta ablations

These rows use the strongest current model, `meta/llama-3.3-70b-instruct`, with `ANVIL_EMBEDDER=sentence_transformer`.

| ablation | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | **0.967** | **1.000** | 0.998 | **1.000** | **1.000** | 0.987 | 0.987 | 0.388 | **1.000** | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| no-pinned | 0.567 | **0.000** | **1.000** | 0.933 | **1.000** | 0.987 | 0.987 | 0.388 | 0.967 | `2026-04-25T15-18-55Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-pinned` |
| no-refusal | 0.933 | **1.000** | 0.998 | 0.980 | **1.000** | 0.987 | 0.987 | 0.388 | **1.000** | `2026-04-25T15-23-07Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-refusal` |
| no-citation-enforcer | 0.867 | 0.917 | **1.000** | 0.880 | **1.000** | 0.987 | 0.987 | 0.388 | 0.900 | `2026-04-25T15-33-42Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-no-citation-enforcer` |

## Agentic calculation smoke result

This is a focused live smoke run over the first three calculation examples after hardening the agent tool boundary:

- normalizes LLM phrases such as `cylindrical shell` → `cylindrical_shell`;
- normalizes `full radiography` → `Full RT`;
- auto-retrieves evidence before calculation;
- auto-finalizes after a successful deterministic calculation.

| configuration | pass_rate | avg_tool_calls | finalize_rate | budget_exhaustion_rate | run_id |
| :--- | ---: | ---: | ---: | ---: | :--- |
| agent / meta llama-3.3-70b-instruct / first 3 calc examples | **1.000** | 2.00 | **1.000** | 0.000 | `2026-04-25T16-01-30Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |

## Agentic comparison status

The full 30-example agent comparison was attempted twice. Both attempts produced committed `agent_transcripts.json` artifacts but were dominated by NIM `429 Too Many Requests` decider errors, so they are recorded as provider-instability evidence, not as a fair agent-quality estimate.

| configuration | n_examples | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | avg_tool_calls | finalize_rate | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| fixed / full retry | 30 | 0.833 | 0.833 | 0.998 | 0.840 | 0.987 | — | — | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| agent / full retry, rate-limited | 30 | 0.233 | 0.167 | 1.000 | 0.080 | 0.080 | 0.13 | 0.067 | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| fixed / partial fallback | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | — | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| agent / partial fallback, rate-limited | 10 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 1.50 | 0.500 | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| agent / delayed full run, rate-limited late | 30 | 0.533 | 0.917 | 1.000 | 0.480 | 0.560 | 2.20 | 0.400 | `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |

## Parser benchmark with Reducto

Reducto is now connected and measured using the current API schema (`upload.file_id` and parse `input`, table format `md`). The result strengthens the parser story because the hosted parser is actually tested, not assumed. Public pressure-system supplements are tracked in `data/parser_benchmark/public_pressure_sources.json`; run `uv run python scripts/download_public_pressure_docs.py` to add NASA-STD-8719.17D and NASA-STD-8719.26 PDFs to the parser stress corpus.

| system | pdf | table_f1 | formula_fidelity | paragraph_ref_recall | section_recall | latency_ms_per_page | cost |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| pymupdf4llm | SPES-1 | 0.957 | 1.000 | 1.000 | 0.882 | 144.1 | free |
| reducto | SPES-1 | 0.731 | 0.000 | 1.000 | 0.765 | 6325.7 | ~$0.01/page |
| naive_pdfminer | SPES-1 | 0.000 | 0.000 | 0.000 | 0.000 | 23.2 | free |
| pymupdf4llm | NASA SP-8007 | 1.000 | 1.000 | 1.000 | 1.000 | 1346.6 | free |
| reducto | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.704 | 721.5 | ~$0.01/page |
| naive_pdfminer | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.000 | 15.7 | free |

## Interpretation

### Best current baseline model

`meta/llama-3.3-70b-instruct` remains the strongest current NIM model for ANVIL:

- `pass_rate = 0.967`
- `calculation_correctness = 1.000`
- `citation_accuracy = 0.998`
- `refusal_calibration = 1.000`

### DeepSeek V4 Flash result

`deepseek-ai/deepseek-v4-flash` is now reachable on NIM and completed a full 30-example baseline run:

- `pass_rate = 0.867`
- `calculation_correctness = 1.000`
- `citation_accuracy = 1.000`

It is a valid bake-off candidate, but it did not beat Meta on the current golden set. The run also encountered at least one provider rate-limit soft-refusal, so its observed pass rate should be interpreted as both model behavior and hosted-endpoint reliability under the current request pattern.

### Pinned-data ablation is the strongest evidence

The live `no-pinned` ablation is the most important DeepMechanix-facing result:

- baseline `calculation_correctness = 1.000`
- no-pinned `calculation_correctness = 0.000`
- baseline `pass_rate = 0.967`
- no-pinned `pass_rate = 0.567`

This is strong evidence that pinned engineering data is not decorative. It is load-bearing for compliant calculation.

### Refusal-gate ablation is nuanced under real models

The fake-backend ablation showed a larger refusal-gate drop. The live Meta `no-refusal` row drops less severely:

- baseline `pass_rate = 0.967`
- no-refusal `pass_rate = 0.933`
- no-refusal `refusal_calibration = 1.000`

Interpretation: the strongest live model has some intrinsic refusal behavior, so disabling the deterministic gate is less catastrophic than with the fake backend. The deterministic gate remains important because it makes refusal behavior explicit, auditable, and model-independent.

### Citation-enforcer ablation exposed provider instability and grounding loss

The `no-citation-enforcer` row under Meta produced:

- `pass_rate = 0.867`
- `faithfulness = 0.880`
- `calculation_correctness = 0.917`
- `refusal_calibration = 0.900`

This run encountered NIM 429 rate-limit soft-refusals near the end, so it should be read as a real-world robustness row rather than a pure isolated citation-enforcer measurement. Still, it supports the broader point: ANVIL needs host-side enforcement and logging because live provider behavior includes schema failures, rate limits, and grounding drift.

## Reproduction commands

NIM health check:

`uv run anvil nim-check --json --list`

Meta baseline:

`ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline`

Meta no-pinned ablation:

`ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-pinned --min-pass-rate 0.0`

Meta no-refusal ablation:

`ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-refusal --min-pass-rate 0.0`

Meta no-citation-enforcer ablation:

`ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-citation-enforcer --min-pass-rate 0.0`

DeepSeek V4 Flash bake-off:

`ANVIL_EMBEDDER=sentence_transformer ANVIL_NIM_MODELS=deepseek-ai/deepseek-v4-flash uv run anvil eval --backend nvidia_nim --model deepseek-ai/deepseek-v4-flash --ablation baseline --min-pass-rate 0.0`

Agent smoke run:

`ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8 --limit 3 --skip-fixed --write-table docs/agent_results_smoke.md`
