# ANVIL: Real-Backend Validation Report

Validation date: 2026-04-26.

ANVIL is a compliance-grade retrieval and calculation system for engineering
standards. It is evaluated publicly on SPES-1, a synthetic ASME-like pressure
equipment standard that can be committed and reproduced without licensed ASME
text. The system parses standards into typed elements, builds a knowledge graph,
retrieves with BM25/vector/graph signals, gates unsupported queries before
generation, performs calculations with deterministic pinned data, and validates
citations after generation.

This report records the current DeepMechanix-facing validation pass. Headline
claims use real NVIDIA NIM runs only. `FakeLLMBackend` remains a deterministic
CI/regression backend and is not used as primary performance evidence.

## Core Claim

The fixed ANVIL pipeline can answer and refuse SPES-1 engineering-standard
queries with auditable provenance under a real hosted LLM backend. On the
100-example public benchmark, `meta/llama-3.3-70b-instruct` reaches:

| metric | score |
| :--- | ---: |
| pass_rate | 0.950 |
| calculation_correctness | 1.000 |
| citation_accuracy | 1.000 |
| faithfulness | 0.977 |
| entity_grounding | 0.989 |
| structural_completeness | 1.000 |
| retrieval_recall_at_k | 1.000 |
| retrieval_precision_at_k | 0.400 |
| refusal_calibration | 0.980 |

Run ID:
`2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline`.

Category pass counts:

| category | passed / total |
| :--- | :--- |
| calculation | 34 / 34 |
| lookup | 17 / 20 |
| cross_reference | 18 / 20 |
| out_of_domain | 12 / 12 |
| edge_case | 14 / 14 |

## Evaluation Setup

Dataset: `tests/evaluation/golden_dataset.json`.

Dataset hash:
`ccc6a1930f6f7ab3627b8c428ea7682449d9c6f843363bc4a3cf17f5c58e472c`.

Dataset composition:

| category | count |
| :--- | ---: |
| calculation | 34 |
| lookup | 20 |
| cross_reference | 20 |
| out_of_domain | 12 |
| edge_case | 14 |

Runtime settings:

| setting | value |
| :--- | :--- |
| backend | `nvidia_nim` |
| model | `meta/llama-3.3-70b-instruct` |
| embedder | `sentence_transformer` |
| public corpus | SPES-1 |
| expected calculation values | generated from `CalculationEngine`, stored in golden JSON |
| arithmetic | deterministic Decimal-backed calculation code |
| material/joint data | pinned JSON-backed tables |
| run artifacts | `data/runs/<run_id>/` |

The current implementation supports NIM request throttling, fallback-key
rotation, and checkpoint/resume. The rate limiter is per configured key, and
retryable provider failures try all configured keys before cooldown.

## NIM Health

Health checks were run after sourcing `.env`.

| command | outcome |
| :--- | :--- |
| `uv run anvil nim-check --json --list` | default catalog reachable; live `/models` returned 136 models |
| targeted Meta probe | `meta/llama-3.3-70b-instruct` reachable, about 267 ms |
| targeted DeepSeek V4 Flash probe | timed out at about 10 s; skipped |
| targeted DeepSeek V4 Pro probe | timed out at about 10 s; skipped |

DeepSeek rows are not reported because the targeted real probes failed. This is
intentional: unreachable models are operational evidence, not missing numbers to
fill in.

## Real Ablation Study

All rows below use the same public 100-example dataset, real NIM backend, and
sentence-transformer embedder.

| ablation | pass_rate | calculation_correctness | citation_accuracy | faithfulness | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | 0.950 | 1.000 | 1.000 | 0.977 | 0.980 | `2026-04-26T04-05-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline` |
| no-pinned | 0.520 | 0.000 | 0.997 | 0.900 | 0.950 | `2026-04-26T04-55-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-pinned` |
| no-refusal | 0.950 | 1.000 | 0.988 | 0.994 | 0.970 | `2026-04-26T05-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-refusal` |
| no-citation-enforcer | 0.970 | 1.000 | 0.998 | 0.994 | 0.980 | `2026-04-26T06-06-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-no-citation-enforcer` |

The pinned-data ablation is the strongest design evidence. Removing pinned
engineering tables collapses calculation correctness from 1.000 to 0.000. This
validates the core ANVIL design choice: material stresses and joint
efficiencies are trusted table lookups, not free-form model extraction.

The refusal-gate ablation is less dramatic under Meta than under deterministic
fake-backend regression tests, but it still drops refusal calibration and
citation accuracy. The gate remains part of the defended architecture because
refusal boundaries should be explicit, audited host behavior.

The citation-enforcer ablation scoring 0.970 does not justify removing the
enforcer. It shows that this model run mostly produced valid citations without
host repair. The enforcer is still the mechanism that makes quote validation and
paragraph compatibility fail closed.

## Agent Status

The agentic tool-calling loop completed a real 100-example run, including
`agent_transcripts.json`.

| configuration | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | avg_tool_calls | finalize_rate | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| agent / Meta / max_steps=8 | 0.460 | 0.878 | 0.989 | 0.511 | 0.786 | 5.02 | 0.530 | `2026-04-26T06-35-00Z_nvidia_nim-llama-3.3-70b-instruct_goldenv2-public100_abl-baseline__agent` |

Agent category pass counts:

| category | passed / total |
| :--- | :--- |
| calculation | 27 / 34 |
| lookup | 0 / 20 |
| cross_reference | 0 / 20 |
| out_of_domain | 10 / 12 |
| edge_case | 9 / 14 |

This is a full run, not a partial fallback. It is also not a headline win. The
transcripts expose repeated retrieval loops, invalid finalization payloads, and
weak lookup/cross-reference behavior. The fixed pipeline remains the
application path; the agent is useful as an audited prototype and debugging
surface.

## Parser Benchmark

The parser benchmark includes Reducto through the configured `REDUCTO_API_KEY`.
During this pass, Reducto used cached outputs and did not require adapter
changes.

| system | pdf | table_f1 | formula_fidelity | paragraph_ref_recall | section_recall | latency_ms_per_page | cost |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| pymupdf4llm | SPES-1 | 0.957 | 1.000 | 1.000 | 0.882 | 144.1 | free |
| reducto | SPES-1 | 0.731 | 0.000 | 1.000 | 0.765 | 6325.7 | about $0.01/page |
| naive_pdfminer | SPES-1 | 0.000 | 0.000 | 0.000 | 0.000 | 23.2 | free |
| pymupdf4llm | NASA SP-8007 | 1.000 | 1.000 | 1.000 | 1.000 | 1346.6 | free |
| reducto | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.704 | 721.5 | about $0.01/page |
| naive_pdfminer | NASA SP-8007 | 0.000 | 1.000 | 1.000 | 0.000 | 15.7 | free |

The defended default remains `pymupdf4llm`: local, free, reproducible, and
stronger on controlled SPES-1 structure than Reducto in this pass.

## Private-ASME Boundary

The public repo does not contain licensed ASME text, screenshots, indexes,
prompts, raw responses, or private run artifacts. The audit command reported:

```text
No unsafe private ASME artifacts visible to git.
```

The private runner was also checked for fail-closed behavior:

| check | outcome |
| :--- | :--- |
| public `--standard` path | rejected because the standard was outside `data/private` |
| public `--output-root` path | rejected because outputs must live under `data/private_runs` |

This supports the leakage conclusion: current committed artifacts are public
SPES-1/NASA artifacts only; private licensed ASME validation is local-only and
sanitized.

## Reproducibility

Quality gates:

```bash
uv run ruff check src/ tests/ scripts/
uv run mypy src/
ANVIL_LLM_BACKEND=fake ANVIL_EMBEDDER=hash uv run pytest tests/ -q
```

NIM health:

```bash
set -a
source .env
set +a
uv run anvil nim-check --json --list
uv run anvil nim-check --models meta/llama-3.3-70b-instruct,deepseek-ai/deepseek-v4-flash,deepseek-ai/deepseek-v4-pro --json
```

Real fixed-pipeline evaluation:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-pinned --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-refusal --min-pass-rate 0.0
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation no-citation-enforcer --min-pass-rate 0.0
```

Resume an interrupted run without wasting completed calls:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run anvil eval --backend nvidia_nim --model meta/llama-3.3-70b-instruct --ablation baseline --min-pass-rate 0.0 --resume-run <run_id>
```

Real agent evaluation:

```bash
ANVIL_EMBEDDER=sentence_transformer uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8
```

Parser and private-artifact checks:

```bash
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer,reducto
uv run python scripts/audit_private_artifacts.py
```

## Remaining Gaps

1. DeepSeek V4 Flash and V4 Pro should be rechecked later because they timed
   out during this validation window.
2. The agent loop needs a stricter tool policy for lookup/cross-reference
   examples before it can compete with the fixed pipeline.
3. The Reducto adapter works, but provider-specific normalization could improve
   SPES-1 formula recovery.
4. A private licensed-ASME validation pass can be run locally by someone with a
   licensed copy, but only sanitized aggregate metrics should leave the private
   workspace.
