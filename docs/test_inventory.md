# Test Inventory

**Hand-maintained.** Every test added during this build-out is recorded
here with its anchor: which milestone, ADR, metric, or audit finding
the test exists to defend. The goal is that a reviewer scanning the
test directory can immediately see *which tests are load-bearing* and
which are scaffolding.

When a test is removed or weakened the corresponding row here MUST be
deleted/updated in the same commit.

| Test (`tests/...`) | Anchor | What it asserts |
| :--- | :--- | :--- |

## Audit-phase regressions (committed before the build-out)

| Test | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_citation_enforcer.py::test_canonical_ref_branch_rejects_fabricated_quote` | Audit A1 / ADR-008 | A canonical-ref citation whose `quoted_text` is not in the resolved element MUST be rejected. |
| `unit/test_citation_enforcer.py::test_canonical_ref_branch_accepts_real_quote_from_standard` | Audit A1 / ADR-008 | A canonical-ref citation whose `quoted_text` IS present in the resolved element passes. |
| `unit/test_citation_enforcer.py::test_canonical_ref_branch_fails_closed_without_element_index` | Audit A1 / ADR-008 | Without an `element_index`, canonical-ref citations fail closed. |
| `unit/test_citation_enforcer.py::test_canonical_ref_branch_walks_up_subparagraphs` | Audit A1 / ADR-008 | `A-27(c)(1)` resolves to `A-27` when no sub-paragraph element exists. |
| `unit/test_audit_regressions.py::test_formula_extractor_populates_applicability_conditions` | Audit A2 / ADR-009 | Parser populates `applicability_conditions` from "applies when …" sentences. |
| `unit/test_audit_regressions.py::test_formula_extractor_attaches_conditions_per_formula` | Audit A2 / ADR-009 | Multiple formulas in one paragraph get per-fence conditions. |
| `unit/test_audit_regressions.py::test_synthetic_standard_a27_formulas_have_conditions` | Audit A2 / ADR-009 | Live SPES-1 markdown produces non-empty `applicability_conditions` for every A-27 formula. |

## M1 — NIM connection check

| Test | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_nim_health.py::test_nim_catalog_has_three_locked_models` | Plan §0 (locked NIM catalog) | Exactly 3 models, exactly the 3 model IDs from the plan, every entry has `label`/`purpose`/`supports_reasoning`. |
| `unit/test_nim_health.py::test_check_nim_health_ok_path` | M1 acceptance | 200 OK + valid usage block → `reachable=True`, latency recorded, request_id surfaced. |
| `unit/test_nim_health.py::test_check_nim_health_missing_key_returns_actionable_error` | M1 acceptance | No `NVIDIA_API_KEY` → actionable error message; never raises. |
| `unit/test_nim_health.py::test_check_nim_health_http_401_redacts_api_key` | M1 + plan §11 (redaction) | A 401 echoing the key in its body MUST NOT leak the key into `error`. |
| `unit/test_nim_health.py::test_check_nim_health_http_429_is_not_an_exception` | M1 acceptance | Rate-limit responses surface as structured errors, not exceptions. |
| `unit/test_nim_health.py::test_check_nim_health_transport_error_caught` | M1 acceptance | DNS / TLS / timeout errors surface as `error`, never raised. |
| `unit/test_nim_health.py::test_check_all_nim_models_probes_every_model_once` | M1 acceptance | Every model probed exactly once, in catalog order. |
| `unit/test_nim_health.py::test_check_all_nim_models_partial_failure_continues` | M1 acceptance | One model failing must not abort probes of the others. |
| `unit/test_nim_health.py::test_cli_nim_check_json_mode_returns_zero_on_any_reachable` | M1 acceptance | `anvil nim-check --json` exits 0 when ≥ 1 model reachable, emits a parseable JSON array on stdout. |
| `unit/test_nim_health.py::test_cli_nim_check_json_mode_returns_one_on_total_failure` | M1 acceptance | All models down → exit 1 (CI-blocking). |
| `unit/test_nim_health.py::test_cli_nim_check_no_key_exits_with_actionable_message` | M1 acceptance | Missing key → exit 1 with a message naming `NVIDIA_API_KEY` and `build.nvidia.com`. |
| `unit/test_nim_health.py::test_list_nim_catalog_returns_model_ids` | NIM integration | `/v1/models` happy path returns the model id list. |
| `unit/test_nim_health.py::test_list_nim_catalog_returns_empty_without_key` | NIM integration | No key → empty list, never raises. |
| `unit/test_nim_health.py::test_list_nim_catalog_returns_empty_on_http_error` | NIM integration | 500 on `/v1/models` → empty list, never raises. |
| `unit/test_nim_health.py::test_cli_nim_check_list_reports_drift` | NIM integration | `--list` surfaces locked-vs-live catalog drift in `catalog_drift` block. |

## M2 — Run-logger + manifest + report-writer

| Test | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_run_logger.py::test_redact_key_is_idempotent_and_short` | Plan §11 (redaction) | `redact_key` is deterministic per-key, never echoes the secret, ≤ 22 chars. |
| `unit/test_run_logger.py::test_redact_key_handles_empty` | Plan §11 | Empty key → `<unset>`. |
| `unit/test_run_logger.py::test_make_run_id_is_filesystem_safe_and_deterministic` | Plan §3.1 | Slugs contain no `/` or spaces and are reproducible from the same inputs. |
| `unit/test_run_logger.py::test_dataset_hash_changes_when_dataset_changes` | Plan §11 | A silent edit to the golden dataset flips the hash. |
| `unit/test_run_logger.py::test_build_manifest_redacts_env_secrets` | Plan §11 (redaction) | NVIDIA_API_KEY in env never lands verbatim in the manifest. |
| `unit/test_run_logger.py::test_build_manifest_attaches_dataset_hash` | M2 acceptance | `dataset_hash` + `n_examples` populated when examples are passed. |
| `unit/test_run_logger.py::test_render_report_includes_pass_rate_and_metric_table` | M2 acceptance | report.md surfaces pass rate, all metrics, and the worst failing example. |
| `unit/test_run_logger.py::test_render_report_handles_all_passing_summary` | M2 acceptance | All-passing run → "every example passed" branch is rendered. |
| `unit/test_run_logger.py::test_run_logger_writes_all_committed_artifacts` | M2 acceptance | All four committed files exist and parse correctly. |
| `unit/test_run_logger.py::test_run_logger_records_raw_responses_when_called` | M2 acceptance | One JSONL line per `record_example` call. |
| `unit/test_run_logger.py::test_run_logger_does_not_create_jsonl_files_if_unused` | Plan §3.1 (lazy fh) | Empty .jsonl files never persisted when no record_*() calls happen. |
| `unit/test_run_logger.py::test_run_logger_marks_incomplete_runs` | M2 acceptance | A crashed run (no `write_summary`) lands with `status=incomplete`. |
| `unit/test_run_logger.py::test_run_logger_redacts_secrets_in_manifest` | Plan §11 (redaction) | End-to-end: API key never leaks into manifest.json or summary.json on disk. |
| `unit/test_run_logger.py::test_run_logger_carries_dataset_hash_when_examples_attached` | M2 acceptance | `attach_examples` populates `dataset_hash` and `n_examples` in the persisted manifest. |

## M3 — Pipeline ablations

| Test | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_ablations.py::test_ablation_catalog_locks_seven_named_configs` | Plan §4.1 | Exactly the 7 named ablations exist; an 8th must be a deliberate edit. |
| `unit/test_ablations.py::test_get_ablation_raises_on_unknown_name` | Fail-loud config | Unknown ablation names raise at lookup, not at runtime. |
| `unit/test_ablations.py::test_baseline_pipeline_uses_hybrid_retriever_mode` | Defaults | Baseline pipeline is the `hybrid` retrieval mode. |
| `unit/test_ablations.py::test_pipeline_threads_retrieval_mode[*]` | Plumbing | Each retrieval-mode ablation actually selects the right `HybridRetriever.mode`. |
| `unit/test_ablations.py::test_unknown_retriever_mode_raises_at_construction` | Fail-loud config | Typos in the mode string fail at construction, never silently as baseline. |
| `unit/test_ablations.py::test_baseline_generator_has_all_gates_on` | Defaults | Baseline generator has all three gates ON. |
| `unit/test_ablations.py::test_generator_threads_each_gate_flag[*]` | Plumbing | Each generator gate is actually flipped by its named ablation. |
| `unit/test_ablations.py::test_bm25_only_results_are_pure_bm25_signal` | Behavioral | bm25-only ablation surfaces zero vector contribution per chunk. |
| `unit/test_ablations.py::test_vector_only_results_are_pure_vector_signal` | Behavioral | vector-only ablation surfaces zero BM25 contribution per chunk. |
| `unit/test_ablations.py::test_no_graph_mode_skips_graph_expanded_chunks` | Behavioral | hybrid_no_graph mode never returns graph-source chunks. |
| `unit/test_ablations.py::test_no_pinned_data_drops_calculation_correctness` | Plan §4.4 acceptance | Disabling pinned data must drop calculation_correctness on calc queries. Locks ADR-002. |
| `unit/test_ablations.py::test_no_refusal_gate_lets_ood_query_reach_llm` | Plan §4.1 A6 | OOD query under no-refusal must NOT carry the gate's refusal_reason. Locks ADR-005. |

## M4 — Trust-boundary calibration

| Test | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_calibration.py::test_relevance_threshold_locked_at_documented_value` | docs/trust_calibration.md operating point | RELEVANCE_THRESHOLD literally equals 0.05; changing it requires coordinated edits. |
| `unit/test_calibration.py::test_calibration_recall_holds_at_chosen_threshold` | M4 acceptance | Refusal recall = 1.0 at the chosen threshold (every OOD example refused). |
| `unit/test_calibration.py::test_calibration_precision_holds_at_chosen_threshold` | M4 acceptance | Refusal precision = 1.0 at the chosen threshold (no in-domain false-refusal). |
| `unit/test_calibration.py::test_calibration_higher_threshold_increases_false_refusals` | Sweep sanity | At threshold=0.30 at least one in-domain FP appears — sweep direction is informative. |

## M5 — Real NIM baseline rows + headline artifacts

| Test / artifact | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_nim_health.py::test_nim_catalog_has_three_locked_models` | M5 / ADR-011 | The live NIM catalog remains intentionally small and reviewable after the catalog refresh. |
| `unit/test_nim_health.py::test_cli_nim_check_list_reports_drift` | M5 / catalog drift | `anvil nim-check --list` reports locked-vs-live drift before expensive eval runs. |
| `docs/headline_results.md` | M5 deliverable | Committed model comparison table links each headline number to a stamped run directory. |
| `docs/nim_integration.md` | M5 deliverable | Documents the active catalog, key-gated behavior, and bake-off override path. |

## M6 — Agentic tool-calling loop

| Test / artifact | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_agent.py::*` | M6 / agent loop invariants | Scripted agent backend, budget exhaustion, tool-error handling, decision invariants, and tool adapter behavior are locked without a live LLM. |
| `integration/test_agent_eval.py::*` | M6 / evaluation parity | Agent runs are scored through the same metric surface as fixed-pipeline runs; budget exhaustion returns refusal-shaped responses. |
| `docs/agent_loop.md` | M6 design artifact | Documents tool surface, loop guarantees, transcript persistence, and known limitations. |
| `docs/pipeline_vs_agent.md` | M6 honesty gate | Explicitly states that live agent-vs-fixed NIM metrics are pending until committed run artifacts exist. |

## M7 — Real-PDF parser benchmark

| Test / artifact | Anchor | What it asserts |
| :--- | :--- | :--- |
| `unit/test_parser_metrics.py::*` | M7 metric definitions | Table F1, formula fidelity, paragraph-ref recall, and section recall behave deterministically. |
| `unit/test_parser_benchmark.py::TestSchemaDriftGuard` | M7 cached-output guard | Cached parser benchmark outputs continue to match the expected schema. |
| `docs/parser_benchmark.md` | M7 deliverable | Records SPES-1 and NASA parser results, parser fixes, remaining parser gaps, and the defended default parser. |
| `src/anvil/parsing/hybrid_parser.py` | M7 prototype / ADR-014 | Feature-flagged parser switching normalizes providers back to `DocumentElement` while keeping `pymupdf4llm` as the defended default. |

## M8 — Workshop-paper-grade report

| Test / artifact | Anchor | What it asserts |
| :--- | :--- | :--- |
| `docs/report.md` | M8 deliverable | Consolidates system overview, related work, evaluation methodology, NIM results, ablations, calibration, parser benchmark, limitations, and reproducibility appendix. |
| `docs/cost_budget.md` | M8 / reviewer transparency | Documents live NIM request budgeting, quota guardrails, and recommended experiment sequencing. |

## M9 — CI, CLI, and deployment polish

| Test / artifact | Anchor | What it asserts |
| :--- | :--- | :--- |
| `.github/workflows/ci.yml` | M9 CI gate | CI runs lint, type-check, the 285-test suite, the offline pipeline regression eval, and an optional NIM health check. |
| `src/anvil/cli.py` | M9 CLI surface / ADR-012 | Installed `anvil` command exposes `nim-check`, `ingest`, `query`, `calculate`, `eval`, and `compare`. |
| `Dockerfile` | M9 deployment | Builds a deterministic read-only FastAPI demo image with no required secrets by default. |
| `fly.toml` | M9 deployment / ADR-013 | Provides a Fly.io deployment template with offline defaults and explicit NIM secret instructions. |

---

## Conventions

- One row per test function. Group by milestone.
- The "Anchor" column links a test to a single load-bearing reason. If a test guards more than one thing, prefer the strictest invariant.
- Removing a test requires deleting its row in the same commit.
- Renaming a test requires updating the row in the same commit. CI grep-checks the row count vs. the file count to catch drift (added once M2 lands).
