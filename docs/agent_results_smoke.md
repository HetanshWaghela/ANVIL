# Agent vs. Fixed Pipeline — Headline Comparison

Model: `meta/llama-3.3-70b-instruct`. Dataset: `tests/evaluation/golden_dataset.json` (3 examples). Budget: max_steps=8.

| configuration | pass_rate | calc_correctness | citation_accuracy | faithfulness | retrieval_recall | avg_tool_calls | finalize_rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| agent / meta/llama-3.3-70b-instruct | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.00 | 1.000 |

_See `data/runs/` for full per-example artifacts and `agent_transcripts.json` for the agent's tool-call traces._
