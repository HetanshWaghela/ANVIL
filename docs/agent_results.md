# Agent vs. Fixed Pipeline — Headline Comparison

Model: `meta/llama-3.3-70b-instruct`. Dataset: `tests/evaluation/golden_dataset.json` (30 examples). Budget: max_steps=8.

This is the second full-run attempt after cooldown. It is not a clean agent-quality headline because the agent decider encountered repeated NIM `429 Too Many Requests` errors. The artifacts are still useful for provider-reliability analysis.

| configuration | pass_rate | calc_correctness | citation_accuracy | faithfulness | retrieval_recall | avg_tool_calls | finalize_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fixed / meta/llama-3.3-70b-instruct | 0.833 | 0.833 | 0.998 | 0.840 | 0.987 | — | — |
| agent / meta/llama-3.3-70b-instruct | 0.233 | 0.167 | 1.000 | 0.080 | 0.080 | 0.13 | 0.067 |

Run IDs:

- fixed: `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed`
- agent: `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`

_See `data/runs/` for full per-example artifacts and `agent_transcripts.json` for the agent's tool-call traces._
