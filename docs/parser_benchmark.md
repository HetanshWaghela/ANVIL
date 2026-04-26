# Parser Benchmark — M7

> **Goal**: Compare PDF parsing systems for ANVIL's compliance-grade RAG pipeline and defend the choice of default parser.

---

## 1. Methodology

### Corpus

| PDF | Source | License | Pages | Tables | Formulas | Paragraph Refs |
|-----|--------|---------|-------|--------|----------|---------------|
| `spes1_synthetic.pdf` | Rendered from `data/synthetic/standard.md` via `pandoc --pdf-engine=xelatex` | MIT (project-internal) | 3 | 2 (B-12, M-1) | 4 (A-27(c)(1), A-27(c)(2), A-27(d), A-32) | 17 section headings |
| `nasa_sp8007.pdf` | [NASA SP-8007: Buckling of Thin-Walled Circular Cylinders](https://ntrs.nasa.gov/citations/19690013955) (1968, rev. 2020) | US Government public domain | 50 | Multiple figures/tables | Engineering formulas | Section-based references |
| `nasa_std_8719_17d.pdf` | [NASA-STD-8719.17D: Ground-Based Pressure Vessels and Pressurized Systems](https://standards.nasa.gov/standard/NASA/NASA-STD-871917) | Publicly accessible NASA standard | optional download | Real pressure-system requirements | Applicability/certification references | NASA section numbering |
| `nasa_std_8719_26.pdf` | [NASA-STD-8719.26: Ground Based Non-Code Metallic Pressure Vessels](https://standards.nasa.gov/standard/NASA/NASA-STD-871926) | Publicly accessible NASA standard | optional download | Pressure-vessel qualification requirements | ASME-adjacent references | NASA section numbering |

**SPES-1 (Controlled Baseline):** The SPES-1 markdown is the authoritative source for ground truth. We parse the source markdown via `parse_markdown_standard()` to produce exact `GroundTruthAnnotation` objects (tables, formulas, paragraph refs, section headings). The PDF is a pandoc-rendered copy — any loss between markdown→PDF→parser→markdown is a genuine measure of parser fidelity.

**NASA SP-8007 (Real-World PDF):** A public-domain US government technical report on thin-walled cylinder buckling. Contains engineering formulas, data tables, and multi-column figures. Ground truth for this PDF is approximate — derived from the best available parser (pymupdf4llm) since manual annotation was not feasible.

**NASA pressure-system supplements:** NASA-STD-8719.17D and NASA-STD-8719.26 are tracked in `data/parser_benchmark/public_pressure_sources.json` and can be downloaded with `scripts/download_public_pressure_docs.py`. They are not part of SPES-1 calculation correctness; they are public real-world stress tests for parser structure, retrieval/citation behavior, and refusal boundaries.

### Evaluation Metrics

All metrics produce a score in [0, 1] where higher is better.

| Metric | Definition | Formula |
|--------|-----------|---------|
| **Table Recovery F1** | Cell-level exact-match F1 across all ground-truth tables | `F1 = 2·P·R / (P+R)` where P = precision, R = recall on `{(row, col, normalized_text)}` tuples |
| **Formula Fidelity** | Fraction of GT formulas where `plain_text` matches (after stripping all whitespace) | `|matched| / |GT_formulas|` |
| **Paragraph Ref Recall** | Fraction of expected paragraph refs (e.g., `A-27`, `B-12`) found in any element's `paragraph_ref` field | `|found| / |GT_refs|` |
| **Section Recall** | Fraction of expected section headings found in any element's `title` field (case-insensitive substring match) | `|found| / |GT_headings|` |
| **Latency (ms/page)** | Wall-clock parsing time divided by page count | Direct measurement |
| **Cost ($/page)** | API cost per page for hosted parsers | Estimated from pricing tiers |

### Systems Compared

| System | Type | Dependency | Cost |
|--------|------|-----------|------|
| `pymupdf4llm` | Local | `pymupdf4llm>=0.0.17` | Free |
| `naive_pdfminer` | Local | `pdfplumber>=0.11` | Free |
| `reducto` | Hosted API | `REDUCTO_API_KEY` | ~$0.01/page |
| `azure_di` | Hosted API, opt-in only | `AZURE_DOCUMENT_INTELLIGENCE_KEY` + `_ENDPOINT` | ~$0.01/page |

---

## 2. Results

Latest command used in the 2026-04-26 validation pass:

```bash
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer,reducto
```

Reducto used cached provider outputs during this pass and did not require
adapter changes.

### SPES-1 Synthetic Standard (Controlled Baseline)

| System | Table F1 | Formula Fid | Para Recall | Section Recall | Latency ms/pg | $/page |
|--------|----------|-------------|-------------|----------------|---------------|--------|
| **pymupdf4llm** | **0.957** | **1.000** | **1.000** | **0.882** | 144.1 | free |
| reducto | 0.731 | 0.000 | **1.000** | 0.765 | 6325.7 | ~$0.01 |
| **naive_pdfminer** | 0.000 | 0.000 | 0.000 | 0.000 | 23.2 | free |

> **Note**: Reducto now runs with the current API schema (`upload.file_id` → parse `input`, table format `md`). It is useful evidence because it is a real hosted parser row, but it does not beat the local default on the controlled SPES-1 benchmark. Azure DI is no longer part of the default benchmark; it can still be requested explicitly with `--systems ...azure_di`.

### NASA SP-8007 (Real-World PDF)

| System | Table F1 | Formula Fid | Para Recall | Section Recall | Latency ms/pg | $/page |
|--------|----------|-------------|-------------|----------------|---------------|--------|
| **pymupdf4llm** | 1.000 | 1.000 | 1.000 | 1.000 | 1346.6 | free |
| reducto | 0.000 | 1.000 | 1.000 | 0.704 | 721.5 | ~$0.01 |
| **naive_pdfminer** | 0.000 | 1.000 | 1.000 | 0.000 | 15.7 | free |

> **Note**: NASA SP-8007 ground truth is approximate (derived from pymupdf4llm output), so pymupdf4llm scores 1.0 by construction. The value here is in comparing *relative* performance: naive_pdfminer correctly extracts text (paragraphs, refs) but produces zero structural understanding (no tables, no section hierarchy).

### Parser Fixes Applied (M7 → post-fix)

The initial benchmark run exposed three parsing defects. All three were fixed during M7:

| Defect | Metric | Before Fix | After Fix | Fix Location |
|--------|--------|-----------|-----------|-------------|
| Bold-wrapped headings (`**A-27**`) | para_recall | 0.000 | **1.000** | `markdown_parser.py`: `_strip_md_formatting()` |
| Formula whitespace stripping | formula_fid | 0.000 | **1.000** | `parser_metrics.py`: `_normalize_formula()` |
| Section heading bold markers | section_recall | 0.765 | **0.882** | `markdown_parser.py`: `_strip_md_formatting()` |
| Wrapped wide table rows | table_f1 | 0.483 | **0.957** | `table_extractor.py`: stacked-header + continuation-row repair |

---

## 3. Failure-Mode Catalog

### 3.1 Heading Bold-Wrapping ~~Breaks~~ (FIXED) Paragraph Reference Extraction

**Affected**: `pymupdf4llm` on all PDFs rendered from markdown

**Root Cause**: `pymupdf4llm` converts all PDF headings to Markdown with bold wrapping:

```diff
- ### A-27 Thickness of Shells Under Internal Pressure
+ ## **A-27 Thickness of Shells Under Internal Pressure**
```

**Status**: ✅ **FIXED**. Added `_strip_md_formatting()` to `markdown_parser.py` which strips `**` and `*` markers before applying the paragraph-ref regex. Result: `paragraph_ref_recall = 1.000`.

### 3.2 Formula Whitespace Stripping ~~Breaks~~ (FIXED) Exact Match

**Affected**: `pymupdf4llm` on PDFs containing fenced code blocks

**Root Cause**: When pandoc renders markdown formulas to PDF and pymupdf4llm extracts them back:

```
Source markdown:  t = (P × R) / (S × E − 0.6 × P)
pymupdf4llm:     t=(P×R)/(S×E−0.6×P)
```

**Status**: ✅ **FIXED**. Updated `_normalize_formula()` in `parser_metrics.py` to strip ALL whitespace before comparison, since the mathematical content is semantically identical. Result: `formula_fidelity = 1.000`.

### 3.3 Table Header Corruption in Multi-Column Tables ~~OPEN~~ (MOSTLY FIXED)

**Affected**: `pymupdf4llm` on wide tables (10+ columns)

**Root Cause**: The M-1 materials table has 13 columns (Spec, Grade, Product Form, plus 10 temperature columns). When pymupdf4llm extracts this table, header alignment can drift:

```
Ground truth headers: ["Spec", "Grade", "Product Form", "40°C", "100°C", ...]
pymupdf4llm headers:  ["", "", "Product", "Product", ...]
```

The header row gets merged or misaligned, causing cell-level F1 to drop.

**Status**: Mostly fixed. `table_extractor.py` now promotes sparse stacked headers, merges wrapped continuation rows, and splits unambiguous fused text/numeric cells such as `Forging138`.

**Impact**: `table_recovery_f1 = 0.957` on SPES-1. The residual gap is from PDF extraction artifacts that lose cell text before the markdown parser sees the table.

### 3.4 Naive Text Extraction: Zero Structural Understanding

**Affected**: `naive_pdfminer` (pdfplumber) on all PDFs

**Root Cause**: `pdfplumber.extract_text()` produces raw text with no heading detection, table parsing, or formula extraction. All content is emitted as flat `PARAGRAPH` elements with no structural metadata.

**Impact**: All structural metrics = 0.000. This establishes the floor baseline and confirms that structural parsing (pymupdf4llm's markdown conversion) is essential for compliance-grade retrieval.

### 3.5 Heading Level Flattening (OPEN)

**Affected**: `pymupdf4llm` on hierarchical documents

**Root Cause**: pymupdf4llm promotes all headings to `##` level regardless of their original depth (`###`, `####`). The SPES-1 standard uses `##` for Parts, `###` for Sections, and heading text for sub-sections. After conversion, the hierarchy is lost:

```
Source: ## Part A → ### A-23 → ### A-27 → **A-27(c)(1)** (sub-heading)
pymupdf4llm: ## Part A → ## A-23 → ## A-27 → ## A-27(c)(1) (all same level)
```

**Impact**: Parent-child relationships between sections cannot be reconstructed. `section_recall` = 0.882 (2 of 17 headings missed — the title and appendix which lack paragraph refs in their bold-wrapped form).

### 3.6 Cross-Reference Paragraph Markers After PDF Round-Trip (OPEN)

**Affected**: `pymupdf4llm` on SPES-1

**Root Cause**: Inline paragraph markers like `A-27(c)(1)` survive the PDF round-trip but the sub-paragraph notation `(c)(1)` can be separated from the base reference by formatting artifacts. The cross-reference detection regex `XREF_PATTERN` expects contiguous text like `A-27(c)(1)` and may miss refs split across lines.

**Impact**: Partial loss of sub-paragraph-level cross-references.

---

## 4. Defended Choice

**ANVIL ships with `pymupdf4llm` as the default parser**, with the following rationale:

After applying the benchmark-driven fixes, pymupdf4llm scores **0.957 table F1**, **1.000 formula fidelity**, **1.000 paragraph ref recall**, and **0.882 section recall** on the controlled SPES-1 baseline. Reducto successfully runs and recovers paragraph references, but its current Markdown output loses SPES formula fidelity and trails pymupdf4llm on controlled table recovery. pymupdf4llm remains the correct default for three reasons. First, **it is free and local**, requiring no API keys, no network access, and no per-page costs — critical for a compliance-grade system processing hundreds of pages of engineering standards. Second, **it produces structural Markdown** — tables as pipe-tables, headings as `##` markers, code fences preserved — which the `parse_markdown_standard()` pipeline can consume directly. Third, the remaining table gap is now small and attributable to upstream PDF extraction loss, not simple header wrapping. Reducto remains a plausible fallback candidate for OCR-heavy PDFs, but it is not the defended default on the current benchmark; Azure DI is opt-in only.

---

## 5. Remaining Work

1. **Improve Reducto normalization** — inspect `data/parser_benchmark/reducto/*/output.json` and add provider-specific Markdown normalization if a hosted fallback is still desired.

2. **Manual NASA ground truth** — replace the current best-effort NASA annotation with a small manually pinned subset so parser improvements are not judged against self-generated pseudo-ground-truth.

3. **Heading hierarchy recovery** — use font-size heuristics from pymupdf4llm's metadata to restore `###` vs `##` heading levels.

---

## 6. Reproducibility

```bash
# Generate SPES-1 PDF from source markdown
pandoc data/synthetic/standard.md -o data/parser_benchmark/pdfs/spes1_synthetic.pdf \
    --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt -V mainfont="Helvetica"

# Run the default benchmark (local parsers + Reducto when REDUCTO_API_KEY is set)
uv run python scripts/run_parser_benchmark.py

# Optional public NASA pressure-system PDFs
uv run python scripts/download_public_pressure_docs.py
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer

# Explicit provider sweep, if you decide to compare Azure later
uv run python scripts/run_parser_benchmark.py --systems pymupdf4llm,naive_pdfminer,reducto,azure_di

# Results are cached in data/parser_benchmark/<system>/<pdf_stem>/output.json
# Aggregate metrics in data/parser_benchmark/results.json
```

All cached outputs are validated by `tests/unit/test_parser_benchmark.py::TestSchemaDriftGuard`.
Metric functions are independently tested in `tests/unit/test_parser_metrics.py` (25 tests).
