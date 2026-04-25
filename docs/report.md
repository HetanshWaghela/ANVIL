# ANVIL: Compliance-Grade Retrieval and Calculation over Engineering Standards

**Workshop-paper-style project report**

## Abstract

ANVIL is a compliance-grade retrieval-augmented reasoning system for engineering standards, built around the failure modes that matter in regulated static-equipment workflows: hallucinated material properties, fabricated paragraph references, ungrounded calculations, and overconfident answers outside the applicable standard. The system uses a legally clean synthetic pressure-equipment standard, SPES-1, whose paragraph structure mirrors ASME BPVC-style references while avoiding copyrighted text. ANVIL parses standards into typed document elements, constructs a knowledge graph, retrieves evidence with BM25 + vector + graph expansion, enforces citations after generation, and performs all engineering arithmetic through deterministic Decimal-backed calculation code using pinned material and weld-efficiency data. Evaluation on the expanded 100-example public SPES-1 set reaches 1.000 pass rate with deterministic offline generation, while the strongest NVIDIA NIM row on the earlier 30-example live set remains 0.967 pass rate with `meta/llama-3.3-70b-instruct`, 1.000 calculation correctness, 0.998 citation accuracy, and 1.000 refusal calibration. Ablations show that disabling pinned data collapses calculation correctness from 1.000 to 0.000, while disabling the refusal gate reduces refusal calibration in deterministic ablation from 1.000 to 0.867. A later verification pass also records real hosted-provider instability, including NIM 429s during agent runs. These results support ANVIL’s core claim: regulated-domain RAG should treat retrieval, calculation, citation, refusal, provider reliability, and licensing boundaries as auditable engineering components rather than prompt-only behavior.

---

## 1. Introduction

Retrieval-augmented generation is often evaluated on whether a model can produce a fluent answer from a set of documents. In regulated engineering domains, that bar is too low. The relevant question is not merely whether the answer sounds plausible, but whether every factual claim can be traced to a cited source, every numerical input is trustworthy, every formula is applicable, and the system refuses when the standard does not support an answer.

Static-equipment design is a useful stress test for this problem. A pressure-vessel calculation may look simple in natural language — “calculate required shell thickness” — but the safe answer depends on multiple constraints:

- allowable stress is material- and temperature-dependent;
- weld joint efficiency depends on joint type and examination extent;
- corrosion allowance is added to calculated thickness;
- formulas apply only inside pressure and thin-wall limits;
- a missing material or out-of-range temperature must cause refusal, not extrapolation;
- every formula and tabular value must be traceable to a paragraph or table.

General-purpose RAG pipelines tend to collapse these concerns into one LLM prompt. ANVIL separates them into explicitly typed stages:

1. Parse a standard into structured document elements.
2. Build a typed knowledge graph over paragraphs, formulas, tables, and cross-references.
3. Retrieve evidence with lexical, vector, and graph signals.
4. Gate unsupported queries before LLM generation.
5. Run calculations deterministically from pinned data.
6. Ask the LLM only to produce a structured, cited response.
7. Validate citations after generation.

The project targets a portfolio/research-artifact standard rather than a production pressure-vessel code tool. It intentionally uses SPES-1, a synthetic ASME-like standard, so the full corpus, evaluation set, and expected answers can be committed publicly and reproduced without legal ambiguity.

---

## 2. Contributions

ANVIL contributes the following engineering artifacts:

1. **A legally clean ASME-style benchmark corpus.** SPES-1 mirrors pressure-equipment standard structure with paragraph references such as `A-27(c)(1)`, material specifications such as `SM-516 Gr 70`, formula applicability conditions, temperature-dependent stress tables, and joint-efficiency tables.

2. **A typed compliance-grade RAG pipeline.** Every data flow passes through Pydantic models for document elements, retrieval chunks, citations, responses, and calculation steps.

3. **A deterministic calculation engine.** The LLM never performs arithmetic. It may contextualize the result, but numerical values come from pinned data and Decimal-backed formulas.

4. **Evidence-enforced generation.** The system validates that cited `quoted_text` appears in the retrieved or canonical source element, including pinned-data citations that may not be in the top-k retrieval set.

5. **A reproducible evaluation harness.** Every evaluation writes a stamped run directory containing `manifest.json`, `summary.json`, `per_example.json`, and a human-readable `report.md`.

6. **Quantified ablations.** The study measures the contribution of BM25, vector retrieval, graph expansion, pinned data, the refusal gate, and the citation enforcer.

7. **Trust-boundary calibration.** The refusal threshold is swept across nine values and defended by precision, recall, and in-domain coverage.

8. **NVIDIA NIM integration.** ANVIL can run against NIM-hosted OpenAI-compatible models through a health-checking CLI and recorded run artifacts.

9. **Parser benchmark.** The project evaluates local PDF parsing on a controlled SPES-1 PDF, a public-domain NASA engineering report, and optional public NASA pressure-system standards.

10. **Agentic-loop prototype.** The codebase includes a bounded tool-calling agent surface for retrieval, graph lookup, pinned lookup, and calculation, with deterministic scripted tests and a live-run script gated on real LLM credentials.

---

## 3. Related Work

ANVIL is not intended to introduce a new neural architecture. Its contribution is in system design, evaluation, and provenance discipline for regulated-domain RAG. The project is informed by the following evaluation and RAG-system literature.

**RAGAS** motivates the use of faithfulness and answer-grounding metrics. ANVIL implements deterministic analogues tailored to SPES-1: a claim is faithful only if it is supported by retrieved context or by a deterministic calculation step whose inputs and formula are cited.

**RAGVUE** motivates attribution of failures to retrieval, reasoning, and grounding stages. ANVIL’s run artifacts preserve per-example retrieval chunks, citation-validation issues, refusal decisions, and calculation steps so failures can be attributed to a pipeline component rather than treated as a single aggregate score.

**Coverage, Not Averages** motivates stratified evaluation. ANVIL’s golden set includes calculation, lookup, cross-reference, out-of-domain, and edge-case categories. Ablation interpretation explicitly calls out when aggregate metrics hide category-specific limitations, such as graph expansion being hard to measure on the small SPES-1 corpus.

**MVES** motivates tiered evaluation for agentic workflows. ANVIL applies the same principle by evaluating the deterministic pipeline, then separately exercising the agentic tool-calling loop with scripted and live backends.

**Evalugator** motivates future scaling of calibration beyond manually labeled golden examples. ANVIL now ships a 100-example public SPES-1 dataset; a natural future step is LLM-judge-assisted expansion with human review for larger private/license-holder validation sets.

**Tripartite-KG** motivates combining knowledge-graph structure with vector retrieval. ANVIL’s graph stores paragraph hierarchy, table/formula relationships, and cross-references, then uses graph expansion to surface neighboring evidence.

**SARG / ASME IDETC 2025** provides domain precedent for RAG over engineering standards and motivates the focus on static-equipment style references, calculations, and citation provenance.

**RAGSmith** frames RAG design as architecture search. ANVIL’s ablation study follows that framing: each component is treated as a design choice that must be defended by data.

**Industrial-scale RAG in automotive engineering** motivates practical, component-level evaluation over only end-to-end answer ratings. ANVIL follows this by reporting retrieval, citation, faithfulness, calculation, refusal, and parsing metrics separately.

---

## 4. System Overview

ANVIL’s architecture is intentionally explicit. It avoids LangChain and LlamaIndex so every stage can be inspected, tested, and logged.

### 4.1 Corpus and parsing

The default corpus is `data/synthetic/standard.md`, a Markdown rendering of SPES-1. The parser emits typed `DocumentElement` objects with fields such as:

- `element_id`
- `element_type`
- `paragraph_ref`
- `title`
- `content`
- `page_number`
- parsed table metadata
- parsed formula metadata
- cross-reference metadata

The parser extracts sections, tables, formulas, paragraph references, and applicability conditions. For PDF input, the default path converts PDF to Markdown through `pymupdf4llm` and then reuses the same Markdown parser. This keeps downstream behavior independent of whether the standard began as Markdown or PDF.

### 4.2 Knowledge graph

Parsed elements are converted into a NetworkX-backed graph. Nodes represent structural entities such as paragraphs, formulas, and tables. Edges encode relationships such as hierarchy, references, formula support, and table support. The graph is used during retrieval to expand from a directly matched chunk to adjacent evidence that may be required for a compliant answer.

For example, a query about cylindrical shell thickness may directly match `A-27(c)(1)`, while the correct calculation also needs material stress from Table M-1 and joint efficiency from Table B-12. Graph expansion is designed to make those neighboring dependencies visible.

### 4.3 Retrieval

Retrieval is hybrid:

- BM25 handles exact paragraph references and standard vocabulary.
- Vector retrieval handles paraphrase and semantic similarity.
- Reciprocal rank fusion combines lexical and vector rankings.
- Graph expansion adds structurally adjacent evidence.
- A lexical-overlap score cap reduces false relevance for unrelated queries.

The benchmark currently shows BM25 is very strong on SPES-1 because the corpus is small and vocabulary is controlled. The hybrid design is retained because larger real standards contain synonymy, cross-references, and lexically distant dependencies that pure BM25 can miss.

### 4.4 Refusal gate

ANVIL refuses before generation when retrieved context is insufficient. This is a deliberate safety boundary: an out-of-domain query should not be sent to an LLM and later “cleaned up.” Refusal is represented as a normal `AnvilResponse` with `confidence="insufficient"` and a structured `refusal_reason`.

The default relevance threshold is calibrated at 0.05 on the golden set. At that threshold, the refusal gate achieves:

- refusal precision: 1.000
- refusal recall: 1.000
- non-refusal coverage: 1.000

### 4.5 Pinned data and deterministic calculations

Material allowable stresses and joint efficiencies are not extracted by the LLM. They are loaded from pinned JSON-backed data and accessed through typed lookup functions.

The calculation engine performs:

1. material lookup;
2. allowable-stress lookup at design temperature;
3. weld joint-efficiency lookup;
4. formula selection;
5. applicability checking;
6. minimum thickness calculation;
7. corrosion allowance addition;
8. nominal plate selection;
9. MAWP back-calculation;
10. citation-bearing `CalculationStep` emission.

Each step has a stable `StepKey`, so evaluation and UI logic do not parse human-readable descriptions.

### 4.6 Generation and citation enforcement

The LLM receives retrieved evidence and, when applicable, a calculation summary. It returns a Pydantic response model. The host injects deterministic calculation steps; the LLM is not allowed to invent them.

Post-generation citation validation checks that:

- the cited element exists in retrieved context, or resolves through the canonical parsed-element index;
- `quoted_text` is present in the source content;
- paragraph references are compatible;
- calculation-step citations also validate.

An audit fix removed a previous silent escape hatch where canonical paragraph references were trusted without validating their quotes.

### 4.7 Run logging

Evaluation runs are stored under `data/runs/<run_id>/`. The reviewable files are:

- `manifest.json`
- `summary.json`
- `per_example.json`
- `report.md`

Raw prompts and request logs are produced locally but gitignored to avoid committing large prompt traces or provider payloads.

---

## 5. Evaluation Methodology

### 5.1 Golden dataset

The public golden dataset contains 100 SPES-1 examples across five categories:

- 34 calculation;
- 20 lookup;
- 20 cross-reference;
- 12 out-of-domain;
- 14 edge case.

Each example records expected references, expected refusal behavior, expected calculation values when applicable, and metric thresholds. The expanded calculation expected values were generated from `CalculationEngine`, then serialized into JSON as reviewed expected outputs; the evaluated runtime does not branch on golden queries or import expected values.

### 5.2 Metrics

ANVIL reports the following metrics:

| Metric | Meaning |
| :--- | :--- |
| `pass_rate` | Fraction of examples passing all configured metric thresholds |
| `calculation_correctness` | Agreement between emitted calculation steps and expected numeric values |
| `citation_accuracy` | Fraction of citations whose quoted text validates against source content |
| `faithfulness` | Whether claims are supported by retrieved context or deterministic calculation |
| `entity_grounding` | Whether named entities in the response are grounded in context |
| `structural_completeness` | Whether expected paragraph references are surfaced |
| `retrieval_recall_at_k` | Fraction of expected references retrieved in top-k |
| `retrieval_precision_at_k` | Precision of retrieved references relative to expected references |
| `refusal_calibration` | Agreement between refusal behavior and expected refusal labels |

Every metric has a mathematical definition in code comments or docstrings. This matters because “faithfulness” or “grounding” claims without implementation detail are too vague for a compliance-grade system.

### 5.3 Reproducibility

Each run manifest captures:

- backend;
- model;
- ablation;
- dataset path and checksum;
- git SHA;
- Python version;
- selected environment variables with secrets redacted;
- ablation configuration.

The headline NIM rows and ablation rows in this report are linked to committed run IDs in `data/runs/`.

### 5.4 Private ASME validation boundary

Real ASME standards are copyright-controlled, so the public repo must not commit ASME PDFs, extracted text, parser outputs, indexes, prompts, raw responses, retrieved chunks, or agent transcripts. The private validation path keeps licensed inputs under `data/private/`, private run outputs under `data/private_runs/`, and publishes only sanitized aggregate metrics. The runnable entry point is `scripts/run_private_asme_eval.py`; safety checks are in `scripts/audit_private_artifacts.py` and documented in `docs/private_asme.md`.

### 5.5 Integrity / hardcoding audit

The 2026-04-25 verification pass audited `src/` for golden-answer leakage and query-specific branching. Searches covered exact golden query prefixes, expected-value imports, `golden_dataset` references, `expected_values` references, suspicious `if query ...` patterns, and terms such as `cheat`, `hardcod`, and static `return` patterns involving domain values. Findings:

- No exact golden query strings were found in `src/`.
- No generation or retrieval code imports `tests/evaluation/golden_dataset.json` or expected answers.
- `expected_values` appears only in evaluation schemas, metrics, and runners.
- Domain constants in `src/anvil/pinned/`, `src/anvil/generation/calculation_engine.py`, and parser/graph regexes are standard/pinned-data logic, not query-specific answer leakage.

The conclusion is that the evaluated system is not passing by memorizing the golden queries. The strongest numeric results come from pinned engineering data and deterministic calculation, which are intentional system components with citations and tests.

---

## 6. Results

### 6.1 Headline NIM results

The current live NIM results use sentence-transformer retrieval and the active NIM catalog.

| Backend / model | pass_rate | calculation_correctness | citation_accuracy | faithfulness | entity_grounding | structural_completeness | retrieval_recall_at_k | retrieval_precision_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| fake / baseline | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 0.987 | 0.987 | 0.388 | 1.000 | `2026-04-25T14-41-21Z_fake_goldenv1_abl-baseline` |
| nvidia_nim / meta llama-3.3-70b-instruct | 0.967 | 1.000 | 0.998 | 1.000 | 1.000 | 0.987 | 0.987 | 0.388 | 1.000 | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| nvidia_nim / qwen3-next-80b-a3b-instruct | 0.800 | 1.000 | 0.988 | 1.000 | 0.930 | 0.987 | 0.987 | 0.388 | 1.000 | `2026-04-25T14-18-20Z_nvidia_nim-qwen3-next-80b-a3b-instruct_goldenv1_abl-baseline` |
| nvidia_nim / kimi-k2-instruct-0905 | 0.900 | 1.000 | 1.000 | 0.975 | 0.960 | 0.987 | 0.987 | 0.388 | 1.000 | `2026-04-25T14-23-10Z_nvidia_nim-kimi-k2-instruct-0905_goldenv1_abl-baseline` |
| nvidia_nim / deepseek-v4-flash | 0.867 | 1.000 | 1.000 | 0.930 | 0.960 | 0.987 | 0.987 | 0.388 | 0.967 | `2026-04-25T16-03-59Z_nvidia_nim-deepseek-v4-flash_goldenv1_abl-baseline` |

The strongest clean live model is `meta/llama-3.3-70b-instruct`, matching the fake backend pass rate while retaining near-perfect citation accuracy. A later verification sweep also produced new run artifacts under `2026-04-25T16-21-16Z_*` through `2026-04-25T16-29-52Z_*`; those rows are intentionally treated as hosted-provider robustness evidence because the later window produced many refusal-shaped or rate-limited responses.

### 6.1a Public 100-example baseline

The expanded public benchmark run is:

| Backend / dataset | n_examples | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | refusal_calibration | run_id |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| fake / `goldenv2-public100` | 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `2026-04-25T18-24-28Z_fake_goldenv2-public100_abl-baseline` |

### 6.2 NIM catalog refresh

The original plan named three NIM models:

- `meta/llama-3.3-70b-instruct`
- `deepseek-ai/deepseek-v3.1`
- `nvidia/llama-3.1-nemotron-70b-instruct`

Live probing showed that model availability had changed. The active catalog now uses:

- `meta/llama-3.3-70b-instruct`
- `qwen/qwen3-next-80b-a3b-instruct`
- `moonshotai/kimi-k2-instruct-0905`

DeepSeek V4 Flash and V4 Pro were reachable in the later health probe, while DeepSeek V3.2 and Kimi timed out at the 10s health-check threshold. DeepSeek V4 Flash completed full baseline runs but did not beat Meta on the clean headline run. This is an important operational lesson: a compliance-grade system cannot treat model IDs as permanent infrastructure. ANVIL’s `nim-check` path exists to catch catalog drift before evaluation or release.

### 6.3 Ablation results

Ablations were run against the deterministic fake backend on the 30-example golden set.

| Ablation | pass_rate | calculation_correctness | citation_accuracy | faithfulness | retrieval_recall_at_k | refusal_calibration |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.967 | 1.000 | 1.000 | 1.000 | 0.987 | 1.000 |
| bm25-only | 0.967 | 1.000 | 1.000 | 1.000 | 0.987 | 1.000 |
| vector-only | 0.900 | 1.000 | 0.980 | 1.000 | 0.987 | 1.000 |
| no-graph | 0.967 | 1.000 | 1.000 | 1.000 | 0.987 | 1.000 |
| no-pinned | 0.567 | 0.000 | 1.000 | 1.000 | 0.987 | 1.000 |
| no-refusal | 0.833 | 1.000 | 1.000 | 1.000 | 0.987 | 0.867 |
| no-citation-enforcer | 0.967 | 1.000 | 1.000 | 1.000 | 0.987 | 1.000 |

The most important row is `no-pinned`. Disabling pinned data drops calculation correctness from 1.000 to 0.000 and pass rate from 0.967 to 0.567. This quantitatively supports the design decision that material values and joint efficiencies should not be free-form LLM extractions.

The second load-bearing row is `no-refusal`. Disabling the pre-generation refusal gate drops refusal calibration from 1.000 to 0.867 and pass rate from 0.967 to 0.833. This supports the design decision that refusal must happen before generation rather than after a potentially hallucinated answer is produced.

BM25-only matching baseline is not evidence that vector retrieval is useless in general. It is evidence that SPES-1 is small and lexically controlled. Vector retrieval is retained for larger, messier corpora where users may paraphrase references or use non-canonical terminology.

Graph expansion also shows no aggregate degradation when disabled on this corpus. This is a limitation of the small benchmark: on SPES-1, the key supporting paragraphs often already rank in the top 10. The graph path remains architecturally justified for larger standards where formulas, tables, and applicability clauses may be lexically distant.

### 6.4 Trust-boundary calibration

The relevance threshold was swept across:

`0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30`

Results:

| threshold | refusal_precision | refusal_recall | non_refusal_coverage | TP | FP | TN | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 1.000 | 1.000 | 1.000 | 5 | 0 | 25 | 0 |
| 0.075 | 1.000 | 1.000 | 1.000 | 5 | 0 | 25 | 0 |
| 0.100 | 0.833 | 1.000 | 0.960 | 5 | 1 | 24 | 0 |
| 0.125 | 0.714 | 1.000 | 0.920 | 5 | 2 | 23 | 0 |
| 0.150 | 0.625 | 1.000 | 0.880 | 5 | 3 | 22 | 0 |
| 0.175 | 0.556 | 1.000 | 0.840 | 5 | 4 | 21 | 0 |
| 0.200 | 0.500 | 1.000 | 0.800 | 5 | 5 | 20 | 0 |
| 0.250 | 0.455 | 1.000 | 0.760 | 5 | 6 | 19 | 0 |
| 0.300 | 0.312 | 1.000 | 0.560 | 5 | 11 | 14 | 0 |

The defended operating point is 0.05. It achieves perfect precision, recall, and in-domain coverage on the current golden set. The threshold is deliberately low because the lexical-overlap cap already suppresses many unrelated queries. Raising the threshold causes false refusals on legitimate in-domain queries without improving recall on the current out-of-domain cohort.

### 6.5 Parser benchmark

The parser benchmark compares local structural parsing against a naive text-extraction baseline and keeps hosted parsers key-gated.

#### SPES-1 synthetic PDF

| System | Table F1 | Formula fidelity | Paragraph recall | Section recall | Latency ms/page | Cost |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| pymupdf4llm | 0.957 | 1.000 | 1.000 | 0.882 | 144.1 | free |
| reducto | 0.731 | 0.000 | 1.000 | 0.765 | 6325.7 | ~$0.01/page |
| naive_pdfminer | 0.000 | 0.000 | 0.000 | 0.000 | 23.2 | free |

#### NASA SP-8007 public-domain PDF

| System | Table F1 | Formula fidelity | Paragraph recall | Section recall | Latency ms/page | Cost |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| pymupdf4llm | 1.000 | 1.000 | 1.000 | 1.000 | 1346.6 | free |
| reducto | 0.000 | 1.000 | 1.000 | 0.704 | 721.5 | ~$0.01/page |
| naive_pdfminer | 0.000 | 1.000 | 1.000 | 0.000 | 15.7 | free |

The NASA ground truth is approximate because it is derived from the best available parser rather than full manual annotation. The controlled SPES-1 row is therefore the more defensible parser benchmark. It shows that structural PDF-to-Markdown parsing is essential: naive text extraction can recover words but not the hierarchy, formulas, and tables required for compliance-grade retrieval. Reducto is now connected and measured. It recovers paragraph references well and is faster than pymupdf4llm on NASA under this benchmark, but it trails the local parser on SPES table recovery and loses formula fidelity after Markdown normalization.

Parser-benchmark-driven fixes improved SPES-1 extraction:

- bold-wrapped headings now preserve paragraph references;
- formula comparison normalizes whitespace;
- table repair handles stacked headers and continuation rows;
- SPES-1 table F1 improved from 0.483 to 0.957.

### 6.6 Agentic tool-calling loop

ANVIL includes an agentic loop that exposes core operations as tools:

- retrieval;
- graph lookup;
- pinned material lookup;
- pinned joint-efficiency lookup;
- deterministic calculation;
- finalization/refusal.

The loop is bounded by a step budget and tool-error budget. It records every tool call in an agent transcript. The evaluation runner scores agent outputs with the same metrics as the fixed pipeline.

The current report treats the agent loop as implemented infrastructure rather than a headline improvement. A live 3-example smoke run passed all metrics after hardening. Full 30-example comparisons were then attempted twice and produced committed `summary.json`, `per_example.json`, `report.md`, and `agent_transcripts.json` artifacts, but the agent rows were dominated by NIM `429 Too Many Requests` decider errors. After adding a 10-second inter-example delay, a full agent-only run completed with `pass_rate = 0.533`, `calculation_correctness = 0.917`, `citation_accuracy = 1.000`, `avg_tool_calls = 2.20`, and `finalize_rate = 0.400` under run `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent`. It still hit 429s late in the run, so it is not a clean agent win, but it is the best current full agent artifact and shows the calculation tool path working on live NIM.

This distinction matters. In a compliance-grade project, “the feature exists,” “the feature improves metrics,” and “the hosted provider allowed a fair measurement” are different claims. ANVIL currently supports the first claim, has partial evidence for the hardened calculation path, and records the provider-instability boundary rather than hiding it.

---

## 7. Discussion

### 7.1 Why pinned data is load-bearing

The strongest result in the project is the pinned-data ablation. It shows that if the deterministic calculation path is removed, the system loses every calculation example’s numeric correctness.

This does not mean every LLM would always extract every value incorrectly. Instead, it means the system’s safety contract depends on typed calculation steps that are produced by trusted code. If the LLM is left to extract and compute values, the result may still look plausible, but it is no longer auditable in the same way.

For regulated calculations, auditability is part of correctness. A response must show:

- the formula;
- every input value;
- the source of each input;
- the computed result;
- the cited paragraph or table.

Pinned data makes that contract enforceable.

### 7.2 Why refusal is a first-class response

Many RAG systems treat refusal as a prompt instruction. ANVIL treats refusal as a pipeline decision. The ablation demonstrates why: if the refusal gate is disabled, out-of-domain queries reach the LLM and refusal calibration drops.

The important design point is timing. Refusal before generation prevents the LLM from inventing irrelevant but plausible content. Refusal after generation is already too late for some failure modes, especially fabricated citations or hallucinated material values.

### 7.3 Why citation enforcement must validate quotes

A citation is not merely a paragraph label. A model can cite the right paragraph while quoting text that does not exist. ANVIL’s citation enforcer checks the quote itself against source content. This also applies to pinned-data citations that reference canonical tables not retrieved in the current top-k set.

The audit fix that removed the canonical-reference escape hatch is a central trust improvement. Without it, a malicious or mistaken LLM could cite `Table M-1` with fabricated text and pass validation.

### 7.4 Why the graph result is currently inconclusive

The graph-expansion ablation shows no measurable aggregate loss on the current corpus. This should not be oversold. SPES-1 is small, controlled, and designed for reproducibility. It does not fully stress cross-document or long-distance reference retrieval.

The graph’s value is architectural and future-facing: when the corpus grows, formulas and required tables may not share terms with the user query. A graph makes those dependencies explicit. The current result says only that the current SPES-1 benchmark is not large enough to quantify that benefit cleanly.

### 7.5 Model selection under live-provider drift

The NIM catalog refresh illustrates a practical problem in real LLM systems: hosted model availability changes. DeepSeek and Nemotron choices from the original plan were not all available or reliable at audit time. ANVIL’s answer is to make model selection observable:

- model IDs live in one catalog;
- `ANVIL_NIM_MODELS` can override the catalog for bake-offs;
- `anvil nim-check` probes availability;
- run manifests capture the model used for each result.

This is more defensible than pretending the originally planned model list remained valid.

---

## 8. Limitations

ANVIL has several important limitations.

### 8.1 Synthetic standard

SPES-1 is synthetic. This is legally and reproducibly useful, but it means the corpus is cleaner than real ASME text. Real standards contain layout artifacts, exceptions, errata, multi-document references, and dense tables that may challenge both parsing and retrieval.

### 8.2 Public benchmark scale

The public golden dataset now has 100 examples. That is enough to catch major architectural failures, such as disabling pinned data or refusal, but still not enough to precisely estimate rare failure modes in real ASME deployments. Some components, such as graph expansion on long-distance references, remain below the current benchmark’s noise floor.

### 8.3 Single-document setting

The current retrieval setting is mostly single-standard. Real engineering workflows may require multi-standard, multi-edition, or code-plus-company-spec retrieval.

### 8.4 Limited real-PDF annotation

The SPES-1 PDF benchmark has reliable ground truth because it comes from the project’s source Markdown. The NASA benchmark is useful but only approximately annotated. A stronger parser study would manually label sections, tables, formulas, and references for several public-domain engineering PDFs.

### 8.5 Agent metrics are provider-limited

The agentic loop is implemented and test-covered, and live comparison artifacts now exist. However, the full agent rows are not a clean headline metric because repeated NIM 429s caused decider failures before tool calls could execute. The project should not claim agentic improvement until a full run completes without provider-side saturation.

### 8.6 No human expert validation

The metrics are deterministic and domain-informed, but there is no licensed pressure-vessel engineer signing off on the synthetic calculations. The project is a software artifact demonstrating evaluation and provenance design, not an engineering design authority.

---

## 9. Future Work

1. **Add private license-holder ASME validation.** Run `scripts/run_private_asme_eval.py` against licensed local inputs and publish sanitized aggregate metrics only.

2. **Add manually annotated real PDFs.** Replace pseudo-ground-truth parser rows with hand-labeled sections, tables, formulas, and cross-references.

3. **Repeat live agent-vs-fixed comparison after quota reset with cooldown.** Use `--sleep-between-examples 10` or higher and keep the existing delayed full-agent artifact as the rate-limit baseline.

4. **Improve Reducto normalization if hosted parsing remains important.** The API now connects, but current Markdown output does not beat the local default on SPES-1.

5. **Scale to multi-document retrieval.** Add edition-aware and source-priority logic for cases where multiple standards or addenda apply.

6. **Add LLM-judge-assisted calibration.** Use Evalugator-style approaches to expand refusal and faithfulness calibration beyond the public 100-example set, with human spot checks.

7. **Improve graph-specific benchmark coverage.** Create examples where graph expansion is required to retrieve table/formula dependencies not lexically visible in the query.

8. **Introduce cost and latency dashboards.** Current run artifacts capture enough metadata to add per-model cost, latency, and failure-rate summaries.

---

## 10. Reproducibility Appendix

### 10.1 Key commands

- Install: `uv sync --extra dev`
- Unit/integration/evaluation tests: `uv run pytest tests/ -q`
- Lint: `uv run ruff check src/ tests/ scripts/`
- Type-check: `uv run mypy src/`
- Ingest: `uv run python scripts/ingest.py`
- Fake evaluation: `uv run python scripts/evaluate.py`
- NIM health check: `uv run anvil nim-check`
- NIM headline run: `uv run python scripts/run_nim_headlines.py --include-fake`
- Ablations: `uv run python scripts/run_ablations.py --backend fake`
- Calibration: `uv run python scripts/run_calibration.py`
- Parser benchmark: `uv run python scripts/run_parser_benchmark.py`
- Agent evaluation, key-gated: `uv run python scripts/run_agent_eval.py --model meta/llama-3.3-70b-instruct --max-steps 8`

### 10.2 Active NIM catalog

The active catalog at the time of this report is:

- `meta/llama-3.3-70b-instruct`
- `qwen/qwen3-next-80b-a3b-instruct`
- `moonshotai/kimi-k2-instruct-0905`

Override with `ANVIL_NIM_MODELS` for live model bake-offs.

### 10.3 Important run IDs

| Purpose | run_id |
| :--- | :--- |
| Fake baseline | `2026-04-25T14-41-21Z_fake_goldenv1_abl-baseline` |
| NIM Meta baseline | `2026-04-25T14-34-17Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline` |
| NIM Qwen baseline | `2026-04-25T14-18-20Z_nvidia_nim-qwen3-next-80b-a3b-instruct_goldenv1_abl-baseline` |
| NIM Kimi baseline | `2026-04-25T14-23-10Z_nvidia_nim-kimi-k2-instruct-0905_goldenv1_abl-baseline` |
| NIM DeepSeek V4 Flash baseline | `2026-04-25T16-03-59Z_nvidia_nim-deepseek-v4-flash_goldenv1_abl-baseline` |
| Agent full retry, fixed | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| Agent full retry, rate-limited agent | `2026-04-25T16-41-40Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| Agent partial fallback, fixed | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__fixed` |
| Agent partial fallback, rate-limited agent | `2026-04-25T16-49-16Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |
| Agent delayed full run | `2026-04-25T17-36-34Z_nvidia_nim-llama-3.3-70b-instruct_goldenv1_abl-baseline__agent` |

### 10.4 Configuration defaults

| Setting | Default / current behavior |
| :--- | :--- |
| LLM backend | fake unless configured |
| Production LLM path | NVIDIA NIM through OpenAI-compatible backend |
| Embedder | deterministic hash by default; sentence-transformer for production runs |
| Parser | `pymupdf4llm` local structural parser |
| Refusal threshold | 0.05 |
| Pinned material data | enabled |
| Citation enforcer | enabled |
| Graph expansion | enabled in baseline |
| Raw request logs | produced locally, gitignored |
| Summary artifacts | committed under `data/runs/` |

---

## 11. Conclusion

ANVIL demonstrates that a regulated-domain RAG system should be evaluated as an engineered pipeline, not as a single prompt. The project’s strongest evidence comes from component ablations: pinned data is essential for calculation correctness, the refusal gate is essential for out-of-domain safety, and citation validation closes a subtle but dangerous provenance gap. Live NVIDIA NIM results show that the pipeline can run against real hosted models while preserving the same typed response contract and evaluation harness used in offline CI.

The project remains intentionally honest about its limits: SPES-1 is synthetic, the golden set is small, graph expansion is not yet strongly differentiated, Reducto does not yet beat the local default after normalization, and the agentic loop needs a quota-stable full run before it becomes a headline claim. Even with those limitations, ANVIL is a defensible portfolio artifact because its claims are tied to run IDs, manifests, metrics, and design decisions rather than anecdotal examples. That is the central lesson: compliance-grade AI systems need reproducible evidence trails as much as they need good answers.