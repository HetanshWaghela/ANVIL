# Agent vs. Fixed Pipeline — Headline Comparison

Model: `meta/llama-3.3-70b-instruct`. Dataset: `tests/evaluation/golden_dataset.json` (30 examples). Budget: max_steps=8.

This full agent-only run used `--sleep-between-examples 10` to reduce provider pressure. It is the best current full agent artifact, but it still encountered late-run NIM 429s, so it should not be presented as a clean agent win over the fixed pipeline.

| configuration | pass_rate | calc_correctness | citation_accuracy | faithfulness | retrieval_recall | avg_tool_calls | finalize_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| agent / meta/llama-3.3-70b-instruct | 0.533 | 0.917 | 1.000 | 0.480 | 0.560 | 2.20 | 0.400 |

Run ID:

- agent: `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

_See `data/runs/` for full per-example artifacts and `agent_transcripts.json` for the agent's tool-call traces._
