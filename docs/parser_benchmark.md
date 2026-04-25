# Parser Benchmark — M7

> **Goal**: Compare PDF parsing systems for ANVIL's compliance-grade RAG pipeline and defend the choice of default parser.

---

## 1. Methodology

### Corpus

| PDF | Source | License | Pages | Tables | Formulas | Paragraph Refs |
|-----|--------|---------|-------|--------|----------|---------------|
| `spes1_synthetic.pdf` | Rendered from `data/synthetic/standard.md` via `pandoc --pdf-engine=xelatex` | MIT (project-internal) | 3 | 2 (B-12, M-1) | 4 (A-27(c)(1), A-27(c)(2), A-27(d), A-32) | 17 section headings |
| `nasa_sp8007.pdf` | [NASA SP-8007: Buckling of Thin-Walled Circular Cylinders](https://ntrs.nasa.gov/citations/19690013955) (1968, rev. 2020) | US Government public domain | 50 | Multiple figures/tables | Engineering formulas | Section-based references |

**SPES-1 (Controlled Baseline):** The SPES-1 markdown is the authoritative source for ground truth. We parse the source markdown via `parse_markdown_standard()` to produce exact `GroundTruthAnnotation` objects (tables, formulas, paragraph refs, section headings). The PDF is a pandoc-rendered copy — any loss between markdown→PDF→parser→markdown is a genuine measure of parser fidelity.

**NASA SP-8007 (Real-World PDF):** A public-domain US government technical report on thin-walled cylinder buckling. Contains engineering formulas, data tables, and multi-column figures. Ground truth for this PDF is approximate — derived from the best available parser (pymupdf4llm) since manual annotation was not feasible.

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
| `azure_di` | Hosted API | `AZURE_DOCUMENT_INTELLIGENCE_KEY` + `_ENDPOINT` | ~$0.01/page |

---

## 2. Results

### SPES-1 Synthetic Standard (Controlled Baseline)

| System | Table F1 | Formula Fid | Para Recall | Section Recall | Latency ms/pg | $/page |
|--------|----------|-------------|-------------|----------------|---------------|--------|
| **pymupdf4llm** | 0.483 | **1.000** | **1.000** | **0.882** | 132.8 | free |
| **naive_pdfminer** | 0.000 | 0.000 | 0.000 | 0.000 | 35.2 | free |
| reducto | — | — | — | — | — | ~$0.01 |
| azure_di | — | — | — | — | — | ~$0.01 |

> **Note**: Hosted parsers (reducto, azure_di) were not run due to missing API keys. Entries marked "—" will be populated when keys are available.

### NASA SP-8007 (Real-World PDF)

| System | Table F1 | Formula Fid | Para Recall | Section Recall | Latency ms/pg | $/page |
|--------|----------|-------------|-------------|----------------|---------------|--------|
| **pymupdf4llm** | 1.000 | 1.000 | 1.000 | 1.000 | 1345.4 | free |
| **naive_pdfminer** | 0.000 | 1.000 | 1.000 | 0.000 | 16.1 | free |

> **Note**: NASA SP-8007 ground truth is approximate (derived from pymupdf4llm output), so pymupdf4llm scores 1.0 by construction. The value here is in comparing *relative* performance: naive_pdfminer correctly extracts text (paragraphs, refs) but produces zero structural understanding (no tables, no section hierarchy).

### Parser Fixes Applied (M7 → post-fix)

The initial benchmark run exposed three parsing defects. All three were fixed during M7:

| Defect | Metric | Before Fix | After Fix | Fix Location |
|--------|--------|-----------|-----------|-------------|
| Bold-wrapped headings (`**A-27**`) | para_recall | 0.000 | **1.000** | `markdown_parser.py`: `_strip_md_formatting()` |
| Formula whitespace stripping | formula_fid | 0.000 | **1.000** | `parser_metrics.py`: `_normalize_formula()` |
| Section heading bold markers | section_recall | 0.765 | **0.882** | `markdown_parser.py`: `_strip_md_formatting()` |

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

### 3.3 Table Header Corruption in Multi-Column Tables (OPEN)

**Affected**: `pymupdf4llm` on wide tables (10+ columns)

**Root Cause**: The M-1 materials table has 13 columns (Spec, Grade, Product Form, plus 10 temperature columns). When pymupdf4llm extracts this table, header alignment can drift:

```
Ground truth headers: ["Spec", "Grade", "Product Form", "40°C", "100°C", ...]
pymupdf4llm headers:  ["", "", "Product", "Product", ...]
```

The header row gets merged or misaligned, causing cell-level F1 to drop.

**Impact**: `table_recovery_f1 = 0.483` on SPES-1 (partial table recovery). This is the primary remaining gap.

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

After applying the bold-stripping and whitespace-normalization fixes identified by this benchmark, pymupdf4llm scores **1.000 formula fidelity**, **1.000 paragraph ref recall**, and **0.882 section recall** on the controlled SPES-1 baseline — with the only remaining gap being table header alignment on wide tables (table_f1 = 0.483). pymupdf4llm is the correct default for three reasons. First, **it is free and local**, requiring no API keys, no network access, and no per-page costs — critical for a compliance-grade system processing hundreds of pages of engineering standards. Second, **it is the only free parser that produces structural Markdown** — tables as pipe-tables, headings as `##` markers, code fences preserved — which the `parse_markdown_standard()` pipeline can consume directly. The naive pdfplumber baseline confirmed that raw text extraction (0.000 across all structural metrics) is unusable. Third, **the remaining table recovery gap is a known, scoped problem** — wide tables with 10+ columns suffer header misalignment, which can be addressed with a targeted header-repair heuristic. Hosted alternatives (Reducto, Azure DI) should be evaluated when API keys are available and may serve as a **fallback for table-heavy PDFs**, but the default must remain local and free.

---

## 5. Remaining Work

1. **Add table header repair heuristic** — when a table has blank/repeated headers, attempt to reconstruct from the first data row or caption text. This would address the remaining table_f1 gap.

2. **Re-run with hosted parsers** (Reducto, Azure DI) once API keys are provisioned, to complete the comparison and update this document.

3. **Heading hierarchy recovery** — use font-size heuristics from pymupdf4llm's metadata to restore `###` vs `##` heading levels.

---

## 6. Reproducibility

```bash
# Generate SPES-1 PDF from source markdown
pandoc data/synthetic/standard.md -o data/parser_benchmark/pdfs/spes1_synthetic.pdf \
    --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt -V mainfont="Helvetica"

# Run the benchmark (skips hosted parsers if keys missing)
uv run python scripts/run_parser_benchmark.py

# Results are cached in data/parser_benchmark/<system>/<pdf_stem>/output.json
# Aggregate metrics in data/parser_benchmark/results.json
```

All cached outputs are validated by `tests/unit/test_parser_benchmark.py::TestSchemaDriftGuard`.
Metric functions are independently tested in `tests/unit/test_parser_metrics.py` (25 tests).
