# Design Decisions

ADR-style record of every "why" in ANVIL. Each entry has the format
**Decision · Context · Options Considered · Chosen Option · Rationale**.
New decisions go at the bottom; old ones are amended (not deleted) when
revisited.

---

## ADR-001 · Synthetic standard (SPES-1) instead of ASME BPVC

**Context.** The real ASME BPVC is copyrighted. Distributing the source
text or reproducing tables verbatim is not viable for a public open-source
demo, but the *structure* (section hierarchy, cross-references, formula
applicability conditions, temperature-dependent stress tables) is exactly
what we need to demonstrate.

**Options considered.**

1. Use a real public-domain standard (e.g. an old EU PED annex). Rejected
   because the structure differs from ASME and the audience targets ASME.
2. Use ASME at "fair-use" length — short excerpts. Rejected because
   citation provenance breaks down on excerpts.
3. Build a faithful synthetic analog (SPES-1).

**Chosen.** Option 3. SPES-1 mirrors ASME paragraph naming (`A-27` ↔
`UG-27`, `B-12` ↔ `UW-12`), uses an `SM-` material prefix to distinguish
from real `SA-` specs, and ships with a 16-row Table M-1, a 6×3 joint
efficiency matrix, and 10 hand-verified design examples.

**Rationale.** The synthetic standard is simultaneously the test fixture,
the parsing input, and the citation target. Every metric in the evaluation
suite is computed against it; every citation in a generated response is
traced back to one of its DocumentElements.

---

## ADR-002 · Pinned data preferred over RAG for material lookups

**Context.** A hallucinated allowable stress is the worst failure mode in
this system: it produces an arithmetically valid but physically wrong
thickness, and there is no way for an LLM to know it was wrong. Generic
RAG, which retrieves *text* and lets the LLM extract values, is exactly
the wrong shape.

**Options considered.**

1. Trust the LLM to extract values from retrieved table rows.
2. Pin material properties as Python data structures separate from the
   RAG index.
3. Both — retrieve for citation context but pin for values.

**Chosen.** Option 3. `src/anvil/pinned/materials.py` loads
`data/synthetic/table_1a_materials.json` into typed records. The retrieval
layer still indexes Table M-1 (so generation has provenance) but the
calculation engine sources every numeric value from pinned data via
`get_allowable_stress(spec_grade, temp_c)`.

**Rationale.** Decouples "where does this number come from?" (pinned, with
linear interpolation, refusing to extrapolate) from "where can I cite this
number from?" (Table M-1 in the parsed standard). A regression test
(`test_markdown_table_m1_matches_pinned_data`) ensures the two sources
never drift.

---

## ADR-003 · LLM never performs arithmetic

**Context.** LLMs can output numerically wrong answers with high
confidence. Allowing the LLM to compute the final thickness is the same
class of failure as allowing it to hallucinate the input stress.

**Options considered.**

1. Let the LLM both select the formula AND compute the result, validate
   post-hoc.
2. Have the LLM select the formula and identify inputs, but compute
   deterministically in Python.

**Chosen.** Option 2. `generation/calculation_engine.py` is the only
producer of numerical results. Every formula is in `pinned/formulas.py`
implemented with `decimal.Decimal` (28-digit precision, half-up rounding).

**Rationale.** Removes a whole class of failure mode. The LLM's job
becomes orchestration and explanation, which is what it is good at. The
seven `CalculationStep`s emitted by the engine are typed by `StepKey` enum
so downstream consumers can never misidentify a step by its description
string.

---

## ADR-004 · `result_key` is an enum, not a description-substring match

**Context.** An earlier prototype identified calculation steps by parsing
their `description` strings (e.g. `"compute" in step.description.lower()`).
Renaming a description silently broke evaluation.

**Options considered.**

1. Match by description substring (status quo at the time).
2. Add a typed enum `StepKey` keyed off the semantic role of each step.

**Chosen.** Option 2. `schemas/generation.py` exports `StepKey`, the
calculation engine sets it on every step, and metrics + the fake LLM
backend index by it.

**Rationale.** Test
`test_calculation_step_carries_result_key` enforces the enum's coverage
and would catch any future regression. Description strings are now free to
be reworded for human readers without breaking machine consumers.

---

## ADR-005 · Refusal gate runs *before* the LLM

**Context.** A "refuse to answer" decision made *after* generation is
already a generated response — and a generated response is a citation
liability. The cleanest way to reject an out-of-domain query is to never
involve the LLM at all.

**Options considered.**

1. Generate first, then validate, then refuse if validation fails.
2. Refuse pre-generation when retrieval evidence is insufficient.

**Chosen.** Option 2 (with a small post-generation citation validator
that downgrades confidence rather than refusing outright).

**Rationale.** Pre-generation refusal eliminates the LLM as a source of
hallucinated content for OOD queries. The post-generation enforcer is a
belt-and-braces measure for in-domain queries where a real backend might
introduce subtle drift.

---

## ADR-006 · `bm25s` and `sqlite-vec` are fail-loud, not silently degraded

**Context.** Production retrieval quality depends on which backend is
actually loaded. A subtle "I tried `sqlite-vec`, it failed, I silently
fell back to NumPy" path means deployed quality drifts away from CI
quality without telling anyone.

**Options considered.**

1. Always silently fall back to the in-repo backend.
2. Silently fall back only on `ImportError`; raise on every other error.

**Chosen.** Option 2. `vector_store.py` and `hybrid_retriever.py` log a
loud WARNING and use the fallback when the production package is *not
installed*; **every other error path raises `RetrievalError`**. The
fallback BM25 implementation is documented as slower and tokenizes
differently from `bm25s`, so substituting one for the other silently
would produce subtly different retrieval results.

**Rationale.** Operators see the WARNING immediately at startup; broken
environments fail loudly close to the configuration site rather than
producing degraded retrieval far from it.

---

## ADR-007 · CitationBuilder refuses to fabricate citations

**Context.** The pinned-data calculation path needs to produce citations
back into the parsed standard (e.g. citing Table M-1 for the SM-516 row).
An earlier implementation hard-coded the `quoted_text` for these
citations. When the standard text was reworded, the citations silently
referenced text that no longer existed.

**Options considered.**

1. Hard-code citation strings (status quo).
2. Resolve every citation against the parsed elements; if the row/section
   isn't found, raise.

**Chosen.** Option 2. `CitationBuilder.from_elements(elements)` builds an
index of paragraph_ref → element. `for_material(spec_grade)` finds the
row whose first cell is the spec_no, fails loud if missing.
`for_paragraph(ref)` walks up sub-paragraphs (`A-27(c)(1)` → `A-27`) and
raises if no parent matches. `for_joint_efficiency(joint_type)` requires
the row to exist.

**Rationale.** Test `test_citation_builder_uses_real_document_content`
proves that every quote is a substring of the standard. Reworking the
markdown automatically updates citations; deleting a row triggers a clean
failure during ingestion rather than a silent runtime mismatch.

---

## ADR-008 · Citation enforcer no longer trusts `quoted_text` on the canonical-ref branch (Audit A1)

**Context.** When a citation referenced a canonical SPES-1 paragraph
(e.g. `Table M-1`) but its `source_element_id` was not in the retrieved
chunks (legitimate for pinned-data lookups), the enforcer used to accept
the citation **without validating `quoted_text`**. A test even
demonstrated this — a citation with `quoted_text="SM-516 Gr 70 at 350°C"`
passed unchecked.

**Options considered.**

1. Keep current behavior (silently trust canonical-ref quotes).
2. Always require the cited element to be in retrieved chunks (would
   break legitimate pinned-data citations).
3. Thread a parsed-element index into `validate_citations` and validate
   the quote against the resolved canonical element. Fail closed if no
   index is available.

**Chosen.** Option 3. The generator and API now pass `element_index`
through to `validate_citations`. `_resolve_canonical_ref` walks
sub-paragraphs the same way `CitationBuilder._resolve_ref` does. With no
index, the canonical branch flags every such citation — silent
permissive fallback is exactly the failure mode this system exists to
prevent.

**Rationale.** Tests
`test_canonical_ref_branch_rejects_fabricated_quote` and
`test_canonical_ref_branch_fails_closed_without_element_index` lock in
the new behavior; existing pinned-data calculation paths still pass via
the legitimate quote-validation path.

---

## ADR-009 · `applicability_conditions` is parsed from the standard, not hardcoded (Audit A2)

**Context.** `ParsedFormula.applicability_conditions` was always set to
`[]` despite the standard explicitly carrying sentences like
*"This formula applies when: t ≤ R/2 AND P ≤ 0.385 × S × E."* after every
formula. A schema field meant to drive compliance reasoning was a dead
field.

**Options considered.**

1. Hardcode the conditions per-formula in the engine.
2. Parse them from the surrounding text in `formula_extractor`.

**Chosen.** Option 2. `_extract_applicability_conditions` runs over the
text region between this formula's code fence and the next, extracts
`applies when ...` / `applicable when ...` clauses, and splits AND-joined
sub-conditions into separate list entries.

**Rationale.** Reflects what the standard actually says, automatically
updates if the standard text is reworded, and gives downstream code
structural access to per-formula constraints. Locked in by three
regression tests including one that runs against the live SPES-1 markdown.

---

## ADR-010 · `--strict` mypy passes everywhere (Audit B1)

**Context.** mypy under `strict = true` was reporting 13 errors across 10
files: missing return types, `Any` leakage from untyped third-party
libs, name shadowing, and a list-comprehension element type that drifted
across the union boundary.

**Options considered.**

1. Loosen the mypy config (e.g. drop `--strict`).
2. Fix each error at its source.

**Chosen.** Option 2. Each fix was minimal: explicit `bool(...)` /
typed-local annotations for `Any`-returning third-party calls, a typed
constant declaration for `_REQUIRED_REFS_FOR_CALC`, a renamed
`calc_citations` local to remove shadowing, and a typed
`outgoing: list[dict[str, object]]` to satisfy the API response model.

**Rationale.** `--strict` is a load-bearing safety net for a
compliance-grade codebase. Loosening it would have hidden the very
class of bugs the project exists to surface.

---

## ADR-011 · NIM catalog refresh 2026-04-25

**Context.** When the live NIM key was wired in, `anvil nim-check`
revealed stale and weak model choices in the original NIM catalog:

* `deepseek-ai/deepseek-v3.1` — `410 Gone` (EOL'd on 2026-04-15 per
  the response body).
* `nvidia/llama-3.1-nemotron-70b-instruct` — `404 Not Found` (the
  function id rotated out of the hosted free-tier catalog).
* `openai/gpt-oss-120b` — reachable but weak for ANVIL: barely met the
  golden threshold and produced citation / JSON reliability failures.
* `nvidia/llama-3.3-nemotron-super-49b-v1.5` — reachable but slower and
  weaker than the best live NIM rows for this task.

**Options considered.**

1. Keep the stale entries and document them as "expected to fail" —
   keeps git history clean, but lies to every reviewer running
   `anvil nim-check`.
2. Refresh the catalog using live probes and real golden-dataset
   evaluations, picking currently hosted models that are reachable and
   strong on structured compliance answers.

**Chosen.** Option 2. The default catalog now contains:

* `meta/llama-3.3-70b-instruct` (kept) — Meta family, strong general
  instruction-follower. Golden pass rate 0.967 with sentence-transformer
  retrieval.
* `qwen/qwen3-next-80b-a3b-instruct` — fast reachable replacement
  candidate. Golden pass rate 0.800.
* `moonshotai/kimi-k2-instruct-0905` — fast reachable cross-family
  candidate. Golden pass rate 0.900.

DeepSeek V4 was considered because the live catalog lists
`deepseek-ai/deepseek-v4-flash` and `deepseek-ai/deepseek-v4-pro`, but
it was not made active: v4-flash returned HTTP 502 after about 104 s,
and v4-pro timed out after 180 s on 2026-04-25. These remain explicit
`ANVIL_NIM_MODELS` override candidates for future bake-offs.

**Rationale.** The drift-detection wiring (`anvil nim-check --list`)
caught both stale entries within minutes of the first real probe —
exactly the failure mode that motivated building the drift detector
in the first place. The refresh is recorded here, in the catalog
docstring, in `.env.example`, in `docs/nim_integration.md`, and in
the locked test in `tests/unit/test_nim_health.py` so a reviewer can
trace the swap end-to-end. The live probe + drift report is the
recommended check before each ablation matrix.
