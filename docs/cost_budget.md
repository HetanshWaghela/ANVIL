# NIM Usage and Cost Budget

## Purpose

This document records the expected NVIDIA NIM usage for ANVIL evaluation runs, the current cost posture, and the guardrails used to keep real-inference experiments reproducible and inexpensive.

ANVIL’s default development and CI paths do **not** require a live model key. They use deterministic local components plus `FakeLLMBackend`. Real NVIDIA NIM calls are reserved for explicitly requested runs such as:

- `anvil nim-check`
- `anvil eval --backend nvidia_nim --model <model-id>`
- `scripts/run_nim_headlines.py`
- `scripts/run_agent_eval.py`

This separation keeps normal development free, deterministic, and safe while still allowing reviewer-facing live-model evidence when `NVIDIA_API_KEY` is available.

---

## Current NIM Model Catalog

The active NIM catalog is intentionally small and health-checkable:

| Label | Model ID | Role |
| :--- | :--- | :--- |
| Meta baseline | `meta/llama-3.3-70b-instruct` | Strongest current ANVIL NIM row |
| Qwen replacement candidate | `qwen/qwen3-next-80b-a3b-instruct` | Reachable cross-family model |
| Kimi replacement candidate | `moonshotai/kimi-k2-instruct-0905` | Reachable cross-family model |

The original plan included DeepSeek and Nemotron model IDs, but live probes showed catalog drift and reliability issues. Those decisions are documented in `docs/design_decisions.md` and `docs/nim_integration.md`.

Override the catalog for a bake-off with:

`ANVIL_NIM_MODELS=model/a,model/b,model/c`

---

## Current Recorded Live Usage

The committed headline results currently include baseline NIM runs for three live models.

| Run type | Models | Examples per model | Approx. live requests |
| :--- | ---: | ---: | ---: |
| NIM health check | 3 | 1 tiny probe | 3 |
| Baseline headline eval | 3 | 30 golden examples | 90 |
| Fake control | 0 | 30 examples | 0 |

Approximate live NIM requests used for the committed headline table:

**93 requests**

This excludes any local development retries, failed probes, or one-off exploratory checks.

---

## Planned Experiment Budget

The original build-out plan described several experiment groups. The table below estimates the request budget if each group is run live against NIM.

| Experiment | Models | Configurations | Golden examples | Estimated NIM requests |
| :--- | ---: | ---: | ---: | ---: |
| Health check | 3 | 1 | 1 tiny prompt | 3 |
| Headline baseline | 3 | 1 | 30 | 90 |
| Full NIM ablation matrix | 3 | 7 | 30 | 630 |
| Minimal NIM ablations | 1 | 3 | 30 | 90 |
| Trust calibration with generation | 3 | 9 thresholds | 30 | 810 |
| Agent vs. fixed pipeline | 1 | 2 paths | 30 | 60+ |
| Parser benchmark | 0 | — | — | 0 NIM requests |

The most expensive planned items are the full NIM ablation matrix and any calibration sweep that calls the LLM at every threshold. ANVIL therefore treats these as optional live runs. The defended default is:

1. Run all ablations with `FakeLLMBackend`.
2. Run only the most informative ablations live on NIM.
3. Run the full live matrix only if quota/time allow.

---

## Recommended Live-Run Budget

For the public portfolio artifact, the recommended budget is:

| Priority | Run | Estimated requests | Rationale |
| :--- | :--- | ---: | :--- |
| Must run | `anvil nim-check` | 3 | Confirms model availability |
| Must run | 3-model headline baseline | 90 | Produces README headline rows |
| Should run | Meta no-pinned ablation | 30 | Validates pinned-data claim with strongest model |
| Should run | Meta no-refusal ablation | 30 | Validates refusal-gate claim with strongest model |
| Should run | Meta no-citation-enforcer ablation | 30 | Tests quote-fabrication behavior with real LLM |
| Optional | Agent vs. fixed on Meta | 60+ | Produces agentic comparison artifact |
| Optional | Full 3×7 NIM ablations | 630 | Best evidence, but quota-heavy |

Recommended minimum live budget:

**183 requests**

Recommended fuller but still controlled budget:

**243–303 requests**

Full matrix budget:

**723+ requests** including health checks and baselines.

---

## Cost Posture

At the time of these runs, ANVIL uses NVIDIA NIM through the hosted endpoint:

`https://integrate.api.nvidia.com/v1`

The project assumes a key-gated hosted usage model and does not hardcode prices, because exact NIM pricing/free-tier quota can change. The cost posture is therefore documented as:

| Path | Cost expectation | Notes |
| :--- | :--- | :--- |
| CI tests | $0 | Fake backend only |
| Local unit/integration tests | $0 | Fake backend only |
| `scripts/evaluate.py` default | $0 | Fake backend unless backend override is supplied |
| `anvil nim-check` | Free-tier / key quota | 3 tiny probes by default |
| 3-model headline eval | Free-tier / key quota | ~90 requests |
| Full NIM ablations | Free-tier if quota allows; otherwise paid/provider-dependent | ~630 requests |
| Agent eval | Free-tier if quota allows; token-heavier than fixed pipeline | Tool loop may require multiple LLM calls per example |

If a paid NIM account is used, compute projected cost with:

`estimated_cost = input_tokens × input_price_per_token + output_tokens × output_price_per_token`

Run logs should capture token counts where the provider response includes usage metadata.

---

## Token Budget Estimate

A typical ANVIL golden-set request includes:

- a system prompt with response-format and grounding rules;
- retrieved context chunks;
- the user query;
- optional deterministic calculation summary;
- structured response output.

The exact token count varies by query category and model response style. For planning purposes:

| Query type | Expected prompt size | Expected completion size | Notes |
| :--- | :--- | :--- | :--- |
| OOD refusal | Low | Low | Refusal path often short-circuits |
| Lookup | Medium | Low/medium | Several citations, short answer |
| Cross-reference | Medium/high | Medium | More retrieved paragraphs |
| Calculation | Medium/high | Medium/high | Includes calculation summary and steps |
| Agentic calculation | High | Variable | Multiple tool-decision turns |

Practical rule:

- Fixed-pipeline golden evals are request-count bounded.
- Agentic evals are both request-count and token-count bounded because each example may require multiple decision turns.

---

## Quota Guardrails

ANVIL uses the following guardrails to avoid accidental quota burn:

1. **Fake backend by default.** No real LLM call happens unless `ANVIL_LLM_BACKEND=nvidia_nim` or an explicit CLI/backend flag is used.

2. **Key-gated live paths.** If `NVIDIA_API_KEY` is missing, NIM health checks return structured failures and live eval scripts abort clearly.

3. **Small locked model catalog.** The default catalog contains three models, not an open-ended provider sweep.

4. **Sequential health checks.** `anvil nim-check` probes models sequentially to reduce rate-limit noise.

5. **Stamped run directories.** Every real run writes a manifest and summary so duplicate runs are visible.

6. **Raw logs gitignored.** Full prompts and request logs can be inspected locally without committing large or sensitive traces.

7. **Ablations can run fake first.** The full 7-config ablation study is reproducible offline. NIM replay is reserved for selected high-value rows.

8. **Agent step budget.** The agent loop uses a max-step budget so a runaway tool-calling sequence cannot generate unbounded requests.

---

## What to Commit

Commit these small, reviewable artifacts from live runs:

- `data/runs/<run_id>/manifest.json`
- `data/runs/<run_id>/summary.json`
- `data/runs/<run_id>/per_example.json`
- `data/runs/<run_id>/report.md`
- generated docs tables such as `docs/headline_results.md`

Do **not** commit:

- `raw_responses.jsonl`
- `prompts.jsonl`
- `request_log.jsonl`
- API keys
- `.env`
- provider secrets

The `.gitignore` is configured to keep raw per-request traces local.

---

## Recommended Run Sequence

For a fresh machine with a valid key:

1. Confirm connectivity:

`uv run anvil nim-check`

2. Run fake baseline to verify local pipeline health:

`uv run anvil eval --backend fake`

3. Run the 3-model NIM headline script:

`uv run python scripts/run_nim_headlines.py --include-fake`

4. Run selected high-value NIM ablations on the strongest model:

`uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-pinned`

`uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-refusal`

`uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-citation-enforcer`

5. If quota remains, run the agent comparison:

`uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8`

---

## Current Cost Summary

Current committed live headline evidence required approximately:

**93 NIM requests**

The recommended next live additions require approximately:

**90 more NIM requests** for the three most informative Meta ablations.

The optional agent-vs-fixed run likely requires:

**60+ NIM requests**, plus extra tool-decision turns depending on agent behavior.

The project is therefore comfortably designed around a low-hundreds request budget for the public artifact, while keeping all CI and local test paths at zero live-model cost.

---

## Reviewer Notes

A reviewer should be able to verify cost discipline by checking:

1. `README.md` for the exact headline run IDs.
2. `docs/headline_results.md` for model-level results.
3. `data/runs/<run_id>/manifest.json` for backend/model settings.
4. `data/runs/<run_id>/summary.json` for aggregate metrics.
5. `.gitignore` for raw request-log exclusions.
6. `docs/nim_integration.md` for connectivity and catalog-drift handling.

The main cost-control principle is simple:

> Run everything offline first; spend NIM requests only on the small set of experiments whose live-model behavior is part of a public claim.