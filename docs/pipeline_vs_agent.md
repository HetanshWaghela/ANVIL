# Pipeline vs. Agent Comparison Status

## Purpose

This document tracks the status of ANVIL’s fixed deterministic pipeline versus the agentic tool-calling loop.

The original build-out plan made the agentic calculation loop a first-class deliverable, but with an important constraint: the project should not claim agentic improvement until a real evaluation run produces committed artifacts under `data/runs/`.

Current status:

- Fixed pipeline: implemented, evaluated, and documented.
- Agentic loop: implemented, test-covered, and hardened after live NIM debugging.
- Live agentic calculation smoke result: complete for the first 3 calculation examples, with committed run artifacts.
- Full 30-example live agent-vs-fixed comparison: attempted twice; both runs are artifacted but dominated by NIM `429 Too Many Requests` decider errors.
- Partial 10-example fallback: completed and artifacted, still rate-limit affected but shows successful agent tool use on the examples that were served.

This document is therefore both a status guide and a record of the hardened agent boundary discovered during live testing.

---

## Configurations Compared

### Fixed Pipeline

The fixed pipeline follows the deterministic ANVIL flow:

1. Retrieve evidence with BM25 + vector + graph expansion.
2. Apply the pre-generation refusal gate.
3. Run deterministic calculation when applicable.
4. Generate a structured `AnvilResponse`.
5. Validate citations after generation.

This is the production-default path and the path used for the headline NIM results.

### Agentic Tool-Calling Loop

The agentic loop exposes ANVIL primitives as bounded tools. A model decides which tool to call at each step until it finalizes or refuses. After live NIM testing, the loop also includes host-side compliance guardrails: if the model jumps straight to calculation, ANVIL automatically hydrates the transcript with retrieval evidence first; if deterministic calculation succeeds, ANVIL auto-finalizes from trusted calculation steps rather than spending extra model turns.

Implemented tool categories include:

| Tool category | Purpose |
| :--- | :--- |
| Retrieval | Retrieve context from the hybrid retriever |
| Graph lookup | Inspect paragraph / neighbor structure |
| Pinned lookup | Query verified material and weld-efficiency data |
| Calculation | Call the deterministic calculation engine |
| Finalization | Emit the final structured response or refusal |

The agent loop is bounded by a step budget and records a transcript of tool calls for review.

---

## Current Evidence

### Fixed Pipeline Evidence

The fixed pipeline has committed live NIM baseline runs.

| Model | pass_rate | calculation_correctness | citation_accuracy | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `meta/llama-3.3-70b-instruct` | 0.967 | 1.000 | 0.998 | 1.000 | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| `moonshotai/kimi-k2-instruct-0905` | 0.900 | 1.000 | 1.000 | 1.000 | `2026-04-25T14-23-10Z_nvidia_nim-kimi-k2-instruct-0905_goldenv1_abl-baseline` |
| `qwen/qwen3-next-80b-a3b-instruct` | 0.800 | 1.000 | 0.988 | 1.000 | `2026-04-25T14-18-20Z_nvidia_nim-qwen3-next-80b-a3b-instruct_goldenv1_abl-baseline` |

The strongest current fixed-pipeline row is `meta/llama-3.3-70b-instruct`.

### Agent Evidence

The agentic loop now has both test coverage and a committed live NIM smoke run.

The first live run exposed three real boundary failures:

| Failure observed | Fix applied |
| :--- | :--- |
| Model repeatedly called `calculate` after successful calculation | Auto-finalize after successful deterministic calculation |
| Model emitted `component="cylindrical shell"` instead of `cylindrical_shell` | Component phrase normalization in the calculation tool adapter |
| Model emitted `rt_level="full radiography"` instead of `Full RT` | Radiography phrase normalization in the calculation tool adapter |
| Agent calculated without first retrieving evidence | Automatic retrieval hydration before calculation |

After hardening, the live smoke run over the first three calculation examples passed all metrics:

| Evidence type | Status |
| :--- | :--- |
| Tool surface implemented | Complete |
| Bounded loop implemented | Complete |
| Transcript schema / persistence | Complete |
| Scripted backend tests | Complete |
| Live NIM agent smoke run | Complete |
| Committed `agent_transcripts.json` artifact | Complete for smoke, full attempts, and partial fallback |
| Full 30-example live comparison | Attempted twice; rate-limit dominated |
| Partial 10-example fallback | Complete; still rate-limit affected |

The current defensible claim is: ANVIL includes an implemented, bounded, test-covered agentic calculation loop that successfully handled 3/3 live calculation examples after tool-boundary hardening. A full live comparison now exists as artifacts, but it should be read as hosted-provider instability evidence because repeated `429` decider failures prevented a clean agent-quality measurement. ANVIL does **not** claim a full golden-set agent win over the deterministic pipeline.

---

## Current Comparison Table

The full 30-example agent comparison was run twice after a cooldown. Both attempts preserved run artifacts and transcripts, but the retry remained rate-limited. A 10-example fallback was then run and clearly labeled as partial.

| Configuration | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | avg_tool_calls | finalize_rate | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Fixed pipeline / `meta/llama-3.3-70b-instruct` / full golden set | 0.967 | 1.000 | 0.998 | 1.000 | 0.987 | — | — | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| Agent loop / `meta/llama-3.3-70b-instruct` / first 3 calculation examples | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.00 | 1.000 | `2026-04-25T16-01-30Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| Fixed pipeline / full retry | 0.833 | 0.833 | 0.998 | 0.840 | 0.987 | — | — | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| Agent loop / full retry, rate-limited | 0.233 | 0.167 | 1.000 | 0.080 | 0.080 | 0.13 | 0.067 | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| Fixed pipeline / first 10 calculation examples | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | — | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| Agent loop / first 10 calculation examples, rate-limited fallback | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 1.50 | 0.500 | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| Agent loop / full golden set, 10s cooldown, rate-limited late | 0.533 | 0.917 | 1.000 | 0.480 | 0.560 | 2.20 | 0.400 | `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |

---

## How to Run the Comparison

The live comparison is key-gated and should be run only when `NVIDIA_API_KEY` is available. Source `.env` first if the key is stored there; the terminal process used for testing did not inherit the key until `.env` was explicitly sourced.

Recommended model:

`meta/llama-3.3-70b-instruct`

Smoke command used for the committed hardened-agent result:

`ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8 --limit 3 --skip-fixed --write-table docs/agent_results_smoke.md`

Full comparison command:

`ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8`

Partial fallback command used after the retry remained unstable:

`ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8 --limit 10 --write-table docs/agent_results_partial.md`

Cooldown command used for the best full agent artifact:

`ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8 --skip-fixed --sleep-between-examples 10 --write-table docs/agent_results_clean.md`

Expected outputs:

| Artifact | Purpose |
| :--- | :--- |
| `data/runs/<run_id>__fixed/summary.json` | Fixed-pipeline aggregate metrics |
| `data/runs/<run_id>__fixed/per_example.json` | Fixed-pipeline per-example metrics |
| `data/runs/<run_id>__fixed/report.md` | Fixed-pipeline human-readable run summary |
| `data/runs/<run_id>__agent/summary.json` | Agent aggregate metrics |
| `data/runs/<run_id>__agent/per_example.json` | Agent per-example metrics |
| `data/runs/<run_id>__agent/report.md` | Agent human-readable run summary |
| `data/runs/<run_id>__agent/agent_transcripts.json` | Full tool-call trace for each example |
| `docs/agent_results.md` | Generated comparison table |

After the run completes, copy the generated comparison table into this document or replace this document with the generated artifact if it contains the full interpretation.

---

## Acceptance Criteria for Claiming Agent Results

The agent comparison should be considered complete only when all of the following are true:

1. A fixed-pipeline run and an agent run exist under `data/runs/`.
2. Both runs use the same model, dataset, git SHA, and evaluation settings.
3. Both runs include committed `summary.json`, `per_example.json`, and `report.md`.
4. The agent run includes `agent_transcripts.json`.
5. The comparison reports:
   - pass rate;
   - calculation correctness;
   - citation accuracy;
   - faithfulness;
   - retrieval recall;
   - average tool calls per query;
   - finalize rate;
   - step-budget exhaustion rate, if available.
6. Any claim in `README.md` or `docs/report.md` links to the run IDs that produced it.

Until a full run completes without provider-side 429 saturation, the correct public claim is:

> ANVIL includes an implemented, bounded, test-covered agentic calculation loop. A live NIM smoke run over 3 calculation examples passes all metrics after hardening the tool boundary. Full and partial live comparisons have been artifacted, but the full comparison is not a clean agent-quality result because NIM returned repeated `429 Too Many Requests` errors in the decider path.

---

## Interpretation Guidance

The agent loop is not automatically better than the fixed pipeline. The comparison should be interpreted in terms of tradeoffs:

| Possible outcome | Interpretation |
| :--- | :--- |
| Agent improves pass rate | Tool selection helps on examples where the fixed pipeline under-retrieves or under-plans |
| Agent matches pass rate with more latency | Fixed pipeline remains preferable for production; agent may still be useful for exploratory workflows |
| Agent lowers pass rate | Deterministic pipeline is more reliable for this constrained compliance task |
| Agent improves retrieval but hurts citation accuracy | Tool use helps evidence gathering but increases grounding risk |
| Agent exhausts step budget often | Prompt/tool policy or max-step budget needs redesign |
| Agent refuses more often | Agent is more conservative, but may reduce in-domain coverage |

The most valuable result is not necessarily “agent wins.” For a compliance-grade system, an honest finding that the deterministic pipeline is more reliable is still a strong engineering conclusion.

---

## Risks to Watch

### 1. Tool-call overuse

The agent may call retrieval or graph lookup repeatedly without improving evidence quality. This should be visible through average tool-call count and transcripts.

### 2. Citation drift

The agent may find correct evidence but finalize with weaker or mismatched citations. Citation accuracy must be compared directly against the fixed pipeline.

### 3. Calculation bypass

The agent must not perform arithmetic in free text. Calculation results should still come from the deterministic calculation tool.

### 4. Budget exhaustion

If many examples terminate because the agent hits `max_steps`, the loop is not ready for headline claims.

### 5. False confidence

The agent should refuse unsupported queries rather than attempting to answer from partial context.

---

## Current Conclusion

As of this document, the fixed deterministic pipeline remains the defended production path for the full 30-example golden set.

The agentic loop is now stronger than a paper prototype: live NIM testing exposed realistic structured-output failures, and the code now handles them through component normalization, radiography normalization, evidence hydration, and deterministic auto-finalization after successful calculation. The best clean result remains the scoped 3-example smoke run. The best full agent artifact is the 10-second-cooldown run, which improves over the saturated full attempts but still hits provider 429s late. The conclusion is clear and useful: the agent path works for calculation orchestration, but the fixed deterministic pipeline remains the production headline because it is less exposed to hosted-provider rate limits and unnecessary decider turns.