# Headline Results

This is the reviewer-facing result page for the current validation pass.
Application claims below use real NVIDIA NIM backend runs only.

Every metric row maps to a stamped directory under `data/runs/<run_id>/`
containing `manifest.json`, `summary.json`, `per_example.json`, raw local
responses, and a run-local `report.md`.

## Current Real Backend Pass

Validation date: 2026-04-26.

Backend settings:

- Backend: `nvidia_nim`
- Primary evaluated model: `meta/llama-3.3-70b-instruct`
- Embedder: `sentence_transformer`
- Dataset: `tests/evaluation/golden_dataset.json`
- Dataset size: 100 public SPES-1 examples
- Dataset hash: `ccc6a1930f6f7ab3627b8c428ea7682449d9c6f843363bc4a3cf17f5c58e472c`
- NIM request policy: per-key request throttling, fallback-key rotation, and checkpoint/resume enabled.

NIM connectivity after sourcing `.env`:

| probe | result |
| :--- | :--- |
| `uv run anvil nim-check --json --list` | default catalog reachable; live `/models` returned 136 models |
| `meta/llama-3.3-70b-instruct` | reachable in the targeted check, about 267 ms |
| `deepseek-ai/deepseek-v4-flash` | targeted check timed out at about 10 s; skipped for evaluation |
| `deepseek-ai/deepseek-v4-pro` | targeted check timed out at about 10 s; skipped for evaluation |

## Real 100-Example Pipeline Results

These are the current application-grade headline rows.

| run | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Meta baseline | **0.950** | **1.000** | **1.000** | 0.977 | 0.989 | **1.000** | **1.000** | 0.400 | 0.980 | `2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline` |
| Meta no-pinned | 0.520 | **0.000** | 0.997 | 0.900 | 0.982 | **1.000** | **1.000** | 0.400 | 0.950 | `2026-04-26T04-55-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-pinned` |
| Meta no-refusal | 0.950 | **1.000** | 0.988 | 0.994 | 0.981 | **1.000** | **1.000** | 0.400 | 0.970 | `2026-04-26T05-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-refusal` |
| Meta no-citation-enforcer | **0.970** | **1.000** | 0.998 | 0.994 | 0.989 | **1.000** | **1.000** | 0.400 | 0.980 | `2026-04-26T06-06-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-citation-enforcer` |

Category pass counts for the baseline run:

| category | passed / total |
| :--- | :--- |
| calculation | 34 / 34 |
| lookup | 17 / 20 |
| cross_reference | 18 / 20 |
| out_of_domain | 12 / 12 |
| edge_case | 14 / 14 |

## Interpretation

`meta/llama-3.3-70b-instruct` is the best current real model for ANVIL because
it completed the full 100-example public benchmark with the strongest clean
baseline artifact: 0.950 pass rate, perfect calculation correctness, perfect
citation accuracy, and near-perfect refusal calibration.

The `no-pinned` ablation is the clearest engineering result. Removing pinned
material and joint-efficiency data drops calculation correctness from 1.000 to
0.000 and pass rate from 0.950 to 0.520. This shows that the pinned engineering
tables are load-bearing, not decorative.

The `no-refusal` ablation is more nuanced under a strong real model. Pass rate
stays at 0.950, but refusal calibration drops from 0.980 to 0.970 and citation
accuracy drops from 1.000 to 0.988. The deterministic refusal gate remains
valuable because it makes the boundary explicit, auditable, and independent of
provider behavior.

The `no-citation-enforcer` row has the highest pass rate in this pass, but it
is not a reason to remove the enforcer. It is a single hosted-provider run where
the model happened to produce mostly valid citations. The enforcer still
defines the safety boundary: quote validation, paragraph compatibility, and
fail-closed behavior belong in host code, not in model goodwill.

## Agent Comparison

The real agent comparison completed over all 100 examples and wrote full
transcripts.

| configuration | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | avg_tool_calls | finalize_rate | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| agent / Meta / 100 examples | 0.460 | 0.878 | 0.989 | 0.511 | 0.786 | 5.02 | 0.530 | `2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent` |

Agent category pass counts:

| category | passed / total |
| :--- | :--- |
| calculation | 27 / 34 |
| lookup | 0 / 20 |
| cross_reference | 0 / 20 |
| out_of_domain | 10 / 12 |
| edge_case | 9 / 14 |

This is a complete artifact, not a partial run, but it is not a headline win.
The transcripts show repeated retrieval loops, invalid finalization payloads,
and weaker lookup/cross-reference behavior than the fixed pipeline. The value
of this row is diagnostic: the fixed pipeline is currently the production path,
while the agent is an auditable prototype with real traces.

## Parser Benchmark With Reducto

Reducto was included in the parser benchmark using the configured
`REDUCTO_API_KEY`. It used cached provider outputs during this pass and did not
require schema-adapter changes.

| system | pdf | table_f1 | formula_fidelity | paragraph_ref_recall | section_recall | latency_ms_per_page | cost |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| pymupdf4llm | SPES-1 | 0.957 | 1.000 | 1.000 | 0.882 | 144.1 | free |
| reducto | SPES-1 | 0.731 | 0.000 | 1.000 | 0.765 | 6325.7 | about $0.01/page |
| naive_pdfminer | SPES-1 | 0.000 | 0.000 | 0.000 | 0.000 | 23.2 | free |
| pymupdf4llm | NASA SP-8007 | 1.000 | 1.000 | 1.000 | 1.000 | 1346.6 | free |
| reducto | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.704 | 721.5 | about $0.01/page |
| naive_pdfminer | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.000 | 15.7 | free |

The defended default remains `pymupdf4llm`: it is local, free, structurally
strong on SPES-1, and does not require sending licensed standards to a hosted
parser.

## Reproduction Commands

```bash
set -a
source .env
set +a

uv run anvil nim-check --json --list
uv run anvil nim-check --models meta/llama-3.3-70b-instruct,deepseek-ai/deepseek-v4-flash,deepseek-ai/deepseek-v4-pro --json

ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-pinned --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-refusal --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-citation-enforcer --min-pass-rate 0.0

ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer,reducto
uv run python scripts/audit_private_artifacts.py
```
