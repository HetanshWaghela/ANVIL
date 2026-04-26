# M6 — Agentic Tool-Calling Loop

## Why an agent at all?

The fixed `AnvilGenerator` pipeline (`retrieve → refusal-gate → calculate → generate`) is **deliberately rigid**: every input-driven query takes the same path, and every step is observable. That rigidity is the point — it's what makes the citation enforcer and the deterministic calculation engine defensible against hallucination.

But it has a ceiling. There are queries where the *right* answer requires deciding when to call which subroutine — for example, "What's the maximum design temperature for SM-516 Gr 70 used in a Type 2 joint with full RT?" That requires a `pinned_lookup(material)` followed by a `pinned_lookup(joint_efficiency)` — not a single retrieve+calc.

M6 adds an **agentic loop** that exposes the existing primitives as tools and lets a real LLM (NVIDIA NIM) pick which to call, turn-by-turn, until it has enough information to finalize. We compare it head-to-head against the fixed pipeline on the same golden dataset under the same metrics — so we can quantify whether agentic flexibility is worth the cost in tokens / latency / reliability.

## Hardened guardrails from live NIM testing

The first live NIM agent smoke run exposed realistic structured-output boundary failures that did not appear in scripted tests:

| Observed failure | Hardened behavior |
| :--- | :--- |
| Model emitted `component="cylindrical shell"` instead of `cylindrical_shell` | Tool adapter normalizes component phrases into strict `CalculationInputs.component` enum values |
| Model emitted `rt_level="full radiography"` instead of `Full RT` | Tool adapter normalizes radiography phrases into pinned Table B-12 labels |
| Model emitted verbose joint labels such as `Type 2 with full radiography` | Joint-type coercion extracts and validates the integer joint type |
| Model called `calculate` without first retrieving evidence | Agent auto-hydrates the transcript with `retrieve_context(query, top_k=10)` before calculation |
| Model repeatedly called `calculate` after a successful calculation | Agent auto-finalizes from deterministic calculation steps and citations, avoiding extra LLM turns and rate-limit exposure |
| Model repeatedly called `pinned_lookup` for direct table questions | Agent auto-finalizes from the pinned table value plus its `CitationBuilder` row reference; no additional LLM turn is required |
| Model looped on `retrieve_context` with paraphrased queries on lookup / cross_reference questions until `budget_steps_exhausted` | After `_RETRIEVAL_SATURATION_THRESHOLD` (=2) successful retrieve/graph calls without a deterministic tool firing, the host hands the agent-curated chunks to `AnvilGenerator.synthesize_from_chunks(...)` — same prompt/LLM/citation-enforcement code path as the fixed pipeline, but executed on the chunks the agent itself selected |

These guardrails are not hardcoded answers. They are host-side contract enforcement around typed tools: the model may decide to calculate, but the host normalizes tool arguments, retrieves evidence, executes trusted deterministic code, and assembles the final `AnvilResponse` from calculation-step provenance.

The retrieval-saturation gate is the most recent addition. It directly targets the failure mode that capped the agent at 0/20 lookup and 0/20 cross_reference passes in the 2026-04-26 100-example run: the LLM rephrased its retrieval query 8 times in a row instead of finalizing. The fix preserves agency — the agent still chooses *which* chunks to retrieve via its own tool calls — while delegating *answer assembly* to trusted host code that already has citation enforcement. Regression tests in `tests/unit/test_agent.py` lock in the trigger conditions and the backwards-compat path (no synthesizer attached → previous behavior).

## Tool surface

| name | adapter | purpose |
| :--- | :--- | :--- |
| `retrieve_context` | `HybridRetriever.retrieve` | BM25 + vector + graph-expansion retrieval |
| `graph_lookup` | `GraphStore.{find_by_paragraph_ref, expand}` | direct-ref + n-hop neighborhood |
| `pinned_lookup` | `pinned.{get_material, get_allowable_stress, get_joint_efficiency}` | verified ground-truth tables |
| `calculate` | `CalculationEngine.calculate` | deterministic ASME formula evaluation |
| `finalize` | (control flow) | emits the final `AnvilResponse` and exits |

Every tool returns a `ToolResult` — `output` on success, `error` on failure — so the loop never sees an exception. Engine-level refusals (out-of-range temp, unknown material) surface as `output={"ok": False, "error": ...}` so the agent can interpret them and refuse intelligently rather than treating them as crashes.

## Loop guarantees

1. **Bounded.** `AgentBudget` defaults to `max_steps=8`, `max_tool_errors=3`. Both are enforced before each `decide()` call. A runaway loop on real-NIM costs at most `8 × decide_latency` per example.
2. **Fail-soft.** Decider exceptions, unknown tools, invalid arguments, and calculation-engine refusals all map to refusal-shaped or structured-error outputs — no uncaught exception tears down the eval run.
3. **Observable.** Every `(call, result)` lands in `AgentTranscript.steps`. The eval runner persists transcripts to `data/runs/<run_id>__agent/agent_transcripts.json` next to the per-example metrics, so any reviewer can replay decisions deterministically.
4. **Evidence-hydrated.** If the model calls `calculate` before retrieving context, the host inserts a `retrieve_context(query, top_k=10)` step first. This preserves the compliance invariant that calculation answers are grounded in retrieved standard evidence.
5. **Deterministically auto-finalized after calculation.** Once `calculate` returns `ok=True`, the host assembles the final `AnvilResponse` from trusted `CalculationStep`s and citations. The LLM is not asked to re-summarize arithmetic it did not perform.
6. **No new metric escape hatches.** Agent runs go through the same metric battery as fixed runs (`AgentEvaluationRunner._score` is byte-identical to `EvaluationRunner._score`). The retrieval set used for `retrieval_recall_at_k` etc. is the **union of every `retrieve_context` call's chunks**, dedup by `element_id` — chunks pulled in via `graph_lookup` do **not** count, so graph-spam can't inflate retrieval scores.

## Decider implementations

* `ScriptedAgentBackend` — deterministic, queue-of-decisions. Used in unit tests so we can assert on loop behavior without hitting any LLM.
* `LLMAgentBackend` — wraps an `LLMBackend` and uses instructor's structured-output mode to choose between `ToolCall` and `FinalAnswer` each turn. Dispatches between the `client.create(...)` (instructor.from_provider) and `client.chat.completions.create(...)` (OpenAI-compatible / NIM) shapes by attribute presence.

## Running

Live agent smoke run used after hardening the tool boundary (requires `NVIDIA_API_KEY`):

```bash
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8 \
    --limit 3 \
    --skip-fixed \
    --write-table docs/agent_results_smoke.md
```

Full agent vs. fixed-pipeline comparison:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8
```

Partial fallback used after full-run retry remained rate-limited:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8 \
    --limit 10 \
    --write-table docs/agent_results_partial.md
```

Cooldown run used for the best full agent artifact:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8 \
    --skip-fixed \
    --sleep-between-examples 10 \
    --write-table docs/agent_results_clean.md
```

Outputs:

* `data/runs/<run_id>__fixed/` — fixed pipeline run (full RunLogger artifact tree)
* `data/runs/<run_id>__agent/` — agent run (RunLogger tree + `agent_transcripts.json`)
* `docs/agent_results.md` — full comparison table
* `docs/agent_results_partial.md` — rate-limit fallback over the first 10 calculation examples
* `docs/agent_results_clean.md` — delayed full agent-only run
* `docs/agent_results_smoke.md` — committed live smoke result over 3 calculation examples after hardening

CI / no-network sanity test:

```bash
uv run pytest tests/integration/test_agent_eval.py -v
```

drives the runner with a `ScriptedAgentBackend` over real golden examples — locks the runner plumbing without needing an API key.

## Tests

Unit (`tests/unit/test_agent.py`, 18 tests):

* Happy path — retrieve → finalize, transcript shape locked.
* Step-budget exhaustion → refusal, termination=`budget_steps_exhausted`.
* Tool-error budget → refusal, termination=`tool_error_limit`.
* Decider exception → caught, termination=`decider_error`.
* Each tool adapter round-trips (retrieve, graph, pinned ×3, calculate happy + engine-refusal, unknown-tool).
* `AgentDecision` XOR invariant (both/neither set both raise).
* Joint-type coercion regression (`'Type 1'`, `'1'`, `1`, `2.0` all → int; bools, `2.5`, lists rejected).
* Empty script → loud `IndexError` (no silent infinite loops).

Integration (`tests/integration/test_agent_eval.py`, 2 tests):

* `AgentEvaluationRunner` over 3 representative golden examples (calc / lookup / OOD-refusal) — full metric coverage parity with `EvaluationRunner`.
* Budget exhaustion on real examples → all responses come back as refusals, `finalize_rate=0.0`.

## Limitations

* The cleanest live committed agent result is currently a 3-example calculation smoke run. It passes all metrics and proves the hardened tool contract.
* Full 30-example comparisons have now been attempted and artifacted, but the original agent rows are dominated by NIM `429 Too Many Requests` decider failures. Treat them as provider-reliability evidence, not a fair measurement of agent reasoning quality.
* A delayed full agent-only run with `--sleep-between-examples 10` is the best current full artifact: pass rate 0.533, calculation correctness 0.917, citation accuracy 1.000, average 2.20 tool calls, finalize rate 0.400. It still hit late-run 429s.
* A 10-example fallback over calculation examples completed with partial success: fixed pipeline pass rate 1.000, agent pass rate 0.500, with successful auto-finalization on served examples and repeated 429s on the rest.
* The fixed deterministic pipeline remains the defended production path for the full golden set.
* The agent gets no tool-result history compaction — for queries that genuinely need 8 steps, the prompt grows linearly. A summarization step would be needed before extending the budget materially.
* No full token/cost model yet — `avg_tool_calls` is reported, and the hardened smoke run averages 2 tool calls per calculation query (`retrieve_context` + `calculate`), but dollar/token budgeting remains future work.
* Hosted NIM rate limits matter. The hardening work specifically reduces repeated decision turns after successful calculation to avoid avoidable 429 failures.
