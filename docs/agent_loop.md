# M6 — Agentic Tool-Calling Loop

## Why an agent at all?

The fixed `AnvilGenerator` pipeline (`retrieve → refusal-gate → calculate → generate`) is **deliberately rigid**: every input-driven query takes the same path, and every step is observable. That rigidity is the point — it's what makes the citation enforcer and the deterministic calculation engine defensible against hallucination.

But it has a ceiling. There are queries where the *right* answer requires deciding when to call which subroutine — for example, "What's the maximum design temperature for SM-516 Gr 70 used in a Type 2 joint with full RT?" That requires a `pinned_lookup(material)` followed by a `pinned_lookup(joint_efficiency)` — not a single retrieve+calc.

M6 adds an **agentic loop** that exposes the existing primitives as tools and lets a real LLM (NVIDIA NIM) pick which to call, turn-by-turn, until it has enough information to finalize. We compare it head-to-head against the fixed pipeline on the same golden dataset under the same metrics — so we can quantify whether agentic flexibility is worth the cost in tokens / latency / reliability.

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
2. **Fail-soft.** Decider exceptions, unknown tools, and invalid arguments all map to a refusal-shaped `AnvilResponse` with `confidence=insufficient` — same shape the fixed pipeline produces — so downstream metrics treat agent failures the same way.
3. **Observable.** Every `(call, result)` lands in `AgentTranscript.steps`. The eval runner persists transcripts to `data/runs/<run_id>__agent/agent_transcripts.json` next to the per-example metrics, so any reviewer can replay decisions deterministically.
4. **No new metric escape hatches.** Agent runs go through the same metric battery as fixed runs (`AgentEvaluationRunner._score` is byte-identical to `EvaluationRunner._score`). The retrieval set used for `retrieval_recall_at_k` etc. is the **union of every `retrieve_context` call's chunks**, dedup by `element_id` — chunks pulled in via `graph_lookup` do **not** count, so graph-spam can't inflate retrieval scores.

## Decider implementations

* `ScriptedAgentBackend` — deterministic, queue-of-decisions. Used in unit tests so we can assert on loop behavior without hitting any LLM.
* `LLMAgentBackend` — wraps an `LLMBackend` and uses instructor's structured-output mode to choose between `ToolCall` and `FinalAnswer` each turn. Dispatches between the `client.create(...)` (instructor.from_provider) and `client.chat.completions.create(...)` (OpenAI-compatible / NIM) shapes by attribute presence.

## Running

Live agent vs. fixed-pipeline comparison (requires `NVIDIA_API_KEY`):

```bash
uv run python scripts/run_agent_eval.py \
    --model meta/llama-3.3-70b-instruct \
    --max-steps 8
```

Outputs:

* `data/runs/<run_id>__fixed/` — fixed pipeline run (full RunLogger artifact tree)
* `data/runs/<run_id>__agent/` — agent run (RunLogger tree + `agent_transcripts.json`)
* `docs/agent_results.md` — comparison table (pass-rate, calc-correctness, citation-accuracy, faithfulness, retrieval-recall, avg tool-calls, finalize-rate)

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

* `LLMAgentBackend` is wired but only exercised end-to-end against a real model (no recorded-response replay yet — that's a future M-tier item).
* The agent gets no tool-result history compaction — for queries that genuinely need 8 steps, the prompt grows linearly. A summarization step would be needed before extending the budget materially.
* No tool-call cost model — `avg_tool_calls` is reported but not budgeted in dollars / tokens. The headline `run_agent_eval.py` script reports it; a defended-tradeoff section in `docs/report.md` (M8) is where this gets pinned.
