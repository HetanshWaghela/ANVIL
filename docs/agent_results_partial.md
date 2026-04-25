# Agent vs. Fixed Pipeline — Headline Comparison

Model: `meta/llama-3.3-70b-instruct`. Dataset: `tests/evaluation/golden_dataset.json` (10 examples). Budget: max_steps=8.

This fallback was run after two full 30-example attempts remained rate-limit affected. It covers the first 10 calculation examples only and should be cited as partial.

| configuration | pass_rate | calc_correctness | citation_accuracy | faithfulness | retrieval_recall | avg_tool_calls | finalize_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fixed / meta/llama-3.3-70b-instruct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | — |
| agent / meta/llama-3.3-70b-instruct | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 1.50 | 0.500 |

Run IDs:

- fixed: `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed`
- agent: `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

_See `data/runs/` for full per-example artifacts and `agent_transcripts.json` for the agent's tool-call traces._
