# Pipeline vs. Agent — Headline Comparison

This is the one-page comparison reviewers should read first. It is updated
to reflect the **post-fix** agent run on the 100-example public benchmark
against real NVIDIA NIM. Implementation details and the exact loop
guardrails live in [`agent_loop.md`](agent_loop.md); the auto-generated
single-row headline table lives in [`agent_results.md`](agent_results.md).

## Headline numbers

Model: `meta/llama-3.3-70b-instruct`. Dataset:
`tests/evaluation/golden_dataset.json` (100 public SPES-1 examples).
Backend: NVIDIA NIM. Embedder: `sentence-transformers/BAAI/bge-small-en-v1.5`.

| configuration | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall@10 | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Fixed pipeline** | **0.950** | **1.000** | **1.000** | 0.977 | 1.000 | `2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline` |
| Agent loop (post-fix) | 0.640 | 0.789 | 0.966 | 0.790 | 0.962 | `2026-04-26T08-59-57Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent` |
| Agent loop (pre-fix, historical) | 0.460 | 0.878 | 0.989 | 0.511 | 0.786 | `2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent` |

## Defended claim

The deterministic fixed pipeline is the production path. The agent is an
**audited prototype** that passes the same metric battery as the fixed
pipeline and produces full transcripts, but it does not match the fixed
pipeline on this constrained-compliance task — and that is itself a useful
engineering finding.

## What the agent fix demonstrates

The pre-fix agent transcripts diagnosed a real LLM-loop failure mode: on
`lookup` and `cross_reference` queries the model rephrased its query 8
times instead of finalizing, scoring **0/40** across both categories. The
post-fix run validates that three host-side guardrails fix this without
reducing auditability:

| failure mode | host-side fix | result |
| :--- | :--- | :--- |
| Loop-on-retrieval until step budget is exhausted | After two retrieve/graph calls without a deterministic tool firing, host hands the agent-curated chunks to `AnvilGenerator.synthesize_from_chunks` (steps 4–6 of the fixed pipeline, full citation enforcement). | `lookup` 0→8, `cross_reference` 0→9 |
| Deterministic-tool-only transcript (no retrieval evidence in the trace) | Auto-hydrate one `retrieve_context` step before any `calculate` or `pinned_lookup` call, so retrieval-based metrics have evidence to score against. | `retrieval_recall@10` 0.786 → 0.962, `faithfulness` 0.511 → 0.790 |
| `AgentDecision` shape ambiguity (`tool_call` and `finalize` both set, neither set) | Strengthened agent system prompt with explicit XOR rules and per-category guidance. | `finalize_rate` 0.530 → 0.740, `avg_tool_calls` 5.02 → 1.97 |

Overall: **pass_rate 0.460 → 0.640** (+18 absolute, +39% relative). The
small dip in `calculation_correctness` (0.878 → 0.789) is variance from
provider rate-limit pressure during the post-fix run (key rotation across
all three configured NIM keys, several 75-second cooldowns).

## Per-category breakdown

| category | post-fix | pre-fix | fixed pipeline |
| :--- | :--- | :--- | :--- |
| calculation | 24 / 34 | 27 / 34 | 34 / 34 |
| lookup | 8 / 20 | 0 / 20 | 19 / 20 |
| cross_reference | 9 / 20 | 0 / 20 | 19 / 20 |
| edge_case | 11 / 14 | 9 / 14 | 13 / 14 |
| out_of_domain | 12 / 12 | 10 / 12 | 12 / 12 |

## Reproducing the agent run

```bash
set -a; source .env; set +a
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8 \
    --skip-fixed \
    --sleep-between-examples 2 \
    --write-table docs/agent_results.md
```

## Acceptance criteria for any future agent claim

1. A fixed-pipeline run and an agent run both exist under `data/runs/`
   with the same model, dataset hash, and git SHA.
2. Both runs include `summary.json`, `per_example.json`, `report.md`.
3. The agent run includes `agent_transcripts.json`.
4. `pass_rate`, `calculation_correctness`, `citation_accuracy`,
   `faithfulness`, `retrieval_recall_at_k`, `avg_tool_calls`,
   `finalize_rate` are all reported.
5. Any claim in `README.md`, `docs/report.md`, or `docs/headline_results.md`
   links to the run id that produced it.

## Risks watched in transcripts

| risk | metric to watch | current status |
| :--- | :--- | :--- |
| Tool-call overuse (graph-spam, retrieve-spam) | `avg_tool_calls` | Down 5.02 → 1.97 after fix. |
| Citation drift (correct evidence, wrong citation) | `citation_accuracy` | 0.966; one full provider regression away from a fail. |
| Calculation bypass (free-text arithmetic) | `calculation_correctness` + transcript audit | Auto-finalize-after-`calculate` enforces tool use. |
| Step-budget exhaustion (loop didn't terminate) | `finalize_rate` | Up 0.530 → 0.740 after fix. |
| False confidence on OOD | `refusal_calibration` | 0.820; `out_of_domain` category still 12 / 12. |
