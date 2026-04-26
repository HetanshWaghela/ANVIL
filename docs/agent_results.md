# Agent vs. Fixed Pipeline — Current Real Run

Model: `meta/llama-3.3-70b-instruct`. Dataset: `tests/evaluation/golden_dataset.json` (100 examples). Budget: max_steps=8.
Run ID: `2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent`.

| configuration | pass_rate | calc_correctness | citation_accuracy | faithfulness | retrieval_recall | avg_tool_calls | finalize_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| agent / meta/llama-3.3-70b-instruct | 0.460 | 0.878 | 0.989 | 0.511 | 0.786 | 5.02 | 0.530 |

This is a complete real-backend artifact, not a partial fallback. It is also a weak result: lookup and cross-reference categories fail frequently because the agent loop repeats retrieval or emits invalid finalization payloads. The fixed retrieve-calculate-generate pipeline remains the application path.

_See `data/runs/2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent/agent_transcripts.json` for the tool-call traces._
