#!/usr/bin/env python3
"""Parser benchmark runner — compare PDF parsing systems on a controlled corpus.

Usage:
    uv run python scripts/run_parser_benchmark.py

Runs all available parsers against PDFs in data/parser_benchmark/pdfs/,
computes metrics against ground truth, and writes results to
data/parser_benchmark/results.json.

Reducto is included in the default run and skipped if `REDUCTO_API_KEY`
is missing. Azure DI remains available but is opt-in via `--systems`.
Cached outputs are reused if the PDF mtime matches.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import structlog

from anvil.evaluation.parser_metrics import (
    score_formula_fidelity,
    score_paragraph_ref_recall,
    score_section_recall,
    score_table_recovery_f1,
)
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.schemas.document import DocumentElement, ParsedFormula, ParsedTable
from anvil.schemas.parser_benchmark import (
    GroundTruthAnnotation,
    ParserMetricResult,
    ParserOutput,
)

logger = structlog.get_logger()

BENCHMARK_DIR = Path("data/parser_benchmark")
PDF_DIR = BENCHMARK_DIR / "pdfs"
GT_DIR = BENCHMARK_DIR / "ground_truth"
RESULTS_PATH = BENCHMARK_DIR / "results.json"
CACHE_DEPENDENCIES = (
    Path(__file__),
    Path("src/anvil/parsing/markdown_parser.py"),
    Path("src/anvil/parsing/table_extractor.py"),
    Path("src/anvil/parsing/formula_extractor.py"),
    Path("src/anvil/evaluation/parser_metrics.py"),
)

# Mapping of system name → env vars required (empty for local parsers)
SYSTEM_REQUIREMENTS: dict[str, list[str]] = {
    "pymupdf4llm": [],
    "naive_pdfminer": [],
    "reducto": ["REDUCTO_API_KEY"],
    "azure_di": ["AZURE_DOCUMENT_INTELLIGENCE_KEY", "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
}
DEFAULT_SYSTEMS: tuple[str, ...] = ("pymupdf4llm", "naive_pdfminer", "reducto")


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------


def generate_spes1_ground_truth(pdf_path: Path) -> GroundTruthAnnotation:
    """Generate ground truth for the SPES-1 synthetic PDF.

    The ground truth is derived from parsing the source markdown directly,
    since the markdown is the authoritative source.
    """
    source_md = Path("data/synthetic/standard.md")
    if not source_md.exists():
        raise FileNotFoundError(f"Synthetic standard markdown not found: {source_md}")

    elements = parse_markdown_standard(source_md)

    tables: list[ParsedTable] = []
    formulas: list[ParsedFormula] = []
    paragraph_refs: list[str] = []
    section_headings: list[str] = []

    for el in elements:
        if el.table is not None:
            tables.append(el.table)
        if el.formula is not None:
            formulas.append(el.formula)
        if el.paragraph_ref and el.paragraph_ref not in paragraph_refs:
            paragraph_refs.append(el.paragraph_ref)
        if el.title and el.title not in section_headings:
            section_headings.append(el.title)

    gt = GroundTruthAnnotation(
        pdf_path=str(pdf_path),
        tables=tables,
        formulas=formulas,
        paragraph_refs=paragraph_refs,
        section_headings=section_headings,
    )

    # Write ground truth to disk
    gt_path = GT_DIR / f"{pdf_path.stem}.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(gt.model_dump_json(indent=2), encoding="utf-8")
    logger.info("ground_truth_generated", pdf=str(pdf_path), gt_path=str(gt_path))

    return gt


# ---------------------------------------------------------------------------
# Parser implementations
# ---------------------------------------------------------------------------


def _get_page_count(pdf_path: Path) -> int:
    """Get the page count of a PDF using pymupdf."""
    try:
        import pymupdf  # type: ignore[import-untyped]

        doc = pymupdf.open(str(pdf_path))
        count = doc.page_count
        doc.close()
        return count
    except ImportError:
        # Fallback: attempt to count from pdfplumber
        try:
            import pdfplumber  # type: ignore[import-untyped]

            with pdfplumber.open(str(pdf_path)) as pdf:
                return len(pdf.pages)
        except ImportError:
            warnings.warn(
                "Neither pymupdf nor pdfplumber available for page counting",
                stacklevel=2,
            )
            return 1


def run_pymupdf4llm(pdf_path: Path) -> ParserOutput:
    """Parse PDF using pymupdf4llm → markdown → parse_markdown_standard."""
    import pymupdf4llm  # type: ignore[import-untyped]

    page_count = _get_page_count(pdf_path)

    t0 = time.perf_counter()
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    elements = parse_markdown_standard(md_text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return ParserOutput(
        system="pymupdf4llm",
        pdf_path=str(pdf_path),
        page_count=page_count,
        elements=elements,
        latency_ms=latency_ms,
        cost_usd=None,
        pdf_mtime=pdf_path.stat().st_mtime,
    )


def run_naive_pdfminer(pdf_path: Path) -> ParserOutput:
    """Parse PDF using pdfplumber — raw text extraction with no structural understanding.

    This is the floor baseline: extracts text but makes no attempt to parse
    tables, formulas, or structural elements. All text goes into a single
    PARAGRAPH element per page.
    """
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pdfplumber is required for the naive_pdfminer baseline. "
            "Install with `uv add pdfplumber`."
        ) from None

    page_count = _get_page_count(pdf_path)
    elements: list[DocumentElement] = []

    t0 = time.perf_counter()
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            elements.append(
                DocumentElement(
                    element_id=f"naive-page-{page_num}",
                    element_type="paragraph",
                    paragraph_ref=None,
                    title=None,
                    content=text,
                    page_number=page_num,
                )
            )
    latency_ms = (time.perf_counter() - t0) * 1000

    return ParserOutput(
        system="naive_pdfminer",
        pdf_path=str(pdf_path),
        page_count=page_count,
        elements=elements,
        latency_ms=latency_ms,
        cost_usd=None,
        pdf_mtime=pdf_path.stat().st_mtime,
    )


def run_reducto(pdf_path: Path) -> ParserOutput:
    """Parse PDF using Reducto hosted API.

    Requires REDUCTO_API_KEY env var. Converts Reducto's output
    to DocumentElements via the markdown adapter.
    """
    import httpx

    api_key = os.environ["REDUCTO_API_KEY"]
    page_count = _get_page_count(pdf_path)

    t0 = time.perf_counter()

    # Upload and parse via Reducto API
    with httpx.Client(timeout=120.0) as client:
        # Step 1: Upload the file
        upload_resp = client.post(
            "https://platform.reducto.ai/upload",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        )
        upload_resp.raise_for_status()
        upload_payload = upload_resp.json()
        file_ref = (
            upload_payload.get("file_id")
            or upload_payload.get("document_url")
            or upload_payload.get("url")
            or ""
        )
        if not file_ref:
            raise RuntimeError("Reducto upload response did not contain file_id/document_url/url")

        # Step 2: Parse the uploaded document
        parse_resp = client.post(
            "https://platform.reducto.ai/parse",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": file_ref,
                "formatting": {"table_output_format": "md"},
            },
        )
        if parse_resp.status_code >= 400:
            raise RuntimeError(
                "Reducto parse failed: "
                f"HTTP {parse_resp.status_code}: {parse_resp.text[:500]}"
            )
        result = parse_resp.json()

    # Extract markdown content from Reducto response
    chunks = result.get("result", {}).get("chunks", [])
    md_parts: list[str] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if content:
            md_parts.append(content)

    md_text = "\n\n".join(md_parts)
    elements = parse_markdown_standard(md_text) if md_text.strip() else []

    latency_ms = (time.perf_counter() - t0) * 1000

    # Estimate cost: Reducto free tier has limits, paid is ~$0.01/page
    cost_per_page = 0.01
    cost_usd = cost_per_page * page_count

    return ParserOutput(
        system="reducto",
        pdf_path=str(pdf_path),
        page_count=page_count,
        elements=elements,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        pdf_mtime=pdf_path.stat().st_mtime,
    )


def run_azure_di(pdf_path: Path) -> ParserOutput:
    """Parse PDF using Azure Document Intelligence.

    Requires AZURE_DOCUMENT_INTELLIGENCE_KEY and
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env vars.
    """
    import httpx

    api_key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
    endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"].rstrip("/")
    page_count = _get_page_count(pdf_path)

    t0 = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        # Submit analysis request
        analyze_url = (
            f"{endpoint}/formrecognizer/documentModels/prebuilt-layout:analyze"
            "?api-version=2023-07-31"
        )
        resp = client.post(
            analyze_url,
            headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "Content-Type": "application/pdf",
            },
            content=pdf_path.read_bytes(),
        )
        resp.raise_for_status()

        # Poll for result
        result_url = resp.headers.get("Operation-Location", "")
        if not result_url:
            raise RuntimeError("Azure DI did not return Operation-Location header")

        for _ in range(60):  # max 60 polls × 2s = 120s
            time.sleep(2)
            poll_resp = client.get(
                result_url,
                headers={"Ocp-Apim-Subscription-Key": api_key},
            )
            poll_resp.raise_for_status()
            status = poll_resp.json().get("status", "")
            if status == "succeeded":
                break
            if status == "failed":
                raise RuntimeError(f"Azure DI analysis failed: {poll_resp.json()}")
        else:
            raise RuntimeError("Azure DI analysis timed out")

        result = poll_resp.json()

    # Convert Azure DI output to markdown-like text, then parse
    md_parts: list[str] = []
    analyze_result = result.get("analyzeResult", {})

    # Extract paragraphs
    for para in analyze_result.get("paragraphs", []):
        role = para.get("role", "")
        content = para.get("content", "")
        if role in ("title", "sectionHeading"):
            md_parts.append(f"## {content}")
        else:
            md_parts.append(content)

    # Extract tables as markdown
    for table in analyze_result.get("tables", []):
        rows_data: dict[int, dict[int, str]] = {}
        col_count = table.get("columnCount", 0)
        for cell in table.get("cells", []):
            r = cell.get("rowIndex", 0)
            c = cell.get("columnIndex", 0)
            rows_data.setdefault(r, {})[c] = cell.get("content", "")

        if rows_data:
            sorted_rows = sorted(rows_data.keys())
            for r_idx, r in enumerate(sorted_rows):
                row_cells = [rows_data[r].get(c, "") for c in range(col_count)]
                md_parts.append("| " + " | ".join(row_cells) + " |")
                if r_idx == 0:
                    md_parts.append("|" + "|".join(["---"] * col_count) + "|")

    md_text = "\n\n".join(md_parts)
    elements = parse_markdown_standard(md_text) if md_text.strip() else []

    latency_ms = (time.perf_counter() - t0) * 1000

    # Azure DI pricing: ~$0.01/page for Read model
    cost_per_page = 0.01
    cost_usd = cost_per_page * page_count

    return ParserOutput(
        system="azure_di",
        pdf_path=str(pdf_path),
        page_count=page_count,
        elements=elements,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        pdf_mtime=pdf_path.stat().st_mtime,
    )


PARSER_FUNCS: dict[str, type[object] | object] = {
    "pymupdf4llm": run_pymupdf4llm,
    "naive_pdfminer": run_naive_pdfminer,
    "reducto": run_reducto,
    "azure_di": run_azure_di,
}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_path(system: str, pdf_stem: str) -> Path:
    return BENCHMARK_DIR / system / pdf_stem / "output.json"


def _load_cached(system: str, pdf_path: Path) -> ParserOutput | None:
    """Load cached output if it exists and PDF mtime matches."""
    cache = _cache_path(system, pdf_path.stem)
    if not cache.exists():
        return None

    try:
        cached = ParserOutput.model_validate_json(cache.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("cache_parse_failed", system=system, cache=str(cache))
        return None

    current_mtime = pdf_path.stat().st_mtime
    if abs(cached.pdf_mtime - current_mtime) > 0.01:
        logger.info("cache_stale", system=system, pdf=str(pdf_path))
        return None
    cache_mtime = cache.stat().st_mtime
    newer_dependency = next(
        (
            dep
            for dep in CACHE_DEPENDENCIES
            if dep.exists() and dep.stat().st_mtime > cache_mtime
        ),
        None,
    )
    if newer_dependency is not None:
        logger.info(
            "cache_stale",
            system=system,
            pdf=str(pdf_path),
            dependency=str(newer_dependency),
        )
        return None

    logger.info("cache_hit", system=system, pdf=str(pdf_path))
    return cached


def _save_cache(output: ParserOutput) -> None:
    """Save parser output to cache."""
    cache = _cache_path(output.system, Path(output.pdf_path).stem)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    logger.info("cache_saved", system=output.system, cache=str(cache))


def _is_stale(path: Path, dependencies: tuple[Path, ...]) -> Path | None:
    if not path.exists():
        return path
    path_mtime = path.stat().st_mtime
    return next(
        (
            dep
            for dep in dependencies
            if dep.exists() and dep.stat().st_mtime > path_mtime
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _check_requirements(system: str) -> bool:
    """Check if required env vars are set for a parser system."""
    reqs = SYSTEM_REQUIREMENTS.get(system, [])
    return all(os.environ.get(var) for var in reqs)


def _parse_systems(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYSTEMS)
    systems = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEM_REQUIREMENTS]
    if unknown:
        supported = ", ".join(sorted(SYSTEM_REQUIREMENTS))
        raise ValueError(f"Unknown parser system(s): {unknown}. Supported: {supported}")
    return systems


def run_benchmark(systems: list[str] | None = None) -> list[ParserMetricResult]:
    """Run the full parser benchmark.

    Returns a list of ParserMetricResult for each (system, PDF) pair.
    """
    selected_systems = systems or list(DEFAULT_SYSTEMS)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning("no_pdfs_found", dir=str(PDF_DIR))
        return []

    all_results: list[ParserMetricResult] = []
    all_outputs: list[ParserOutput] = []

    for pdf_path in pdfs:
        logger.info("benchmark_pdf", pdf=str(pdf_path))

        # Load or generate ground truth
        gt_path = GT_DIR / f"{pdf_path.stem}.json"
        gt_dependencies = CACHE_DEPENDENCIES + (pdf_path,)
        gt_stale_dep = _is_stale(gt_path, gt_dependencies)
        if gt_path.exists() and gt_stale_dep is None:
            gt = GroundTruthAnnotation.model_validate_json(
                gt_path.read_text(encoding="utf-8")
            )
        elif pdf_path.stem == "spes1_synthetic":
            gt = generate_spes1_ground_truth(pdf_path)
        else:
            # For non-SPES PDFs without pre-built GT, create a minimal
            # annotation from pymupdf4llm output (best available)
            logger.info(
                "generating_best_effort_gt",
                pdf=str(pdf_path),
                note="Using pymupdf4llm output as approximate ground truth",
                stale_dependency=str(gt_stale_dep) if gt_stale_dep is not None else None,
            )
            pymupdf_output = _load_cached("pymupdf4llm", pdf_path)
            if pymupdf_output is None:
                pymupdf_output = run_pymupdf4llm(pdf_path)
                _save_cache(pymupdf_output)

            gt_tables = [e.table for e in pymupdf_output.elements if e.table is not None]
            gt_formulas = [e.formula for e in pymupdf_output.elements if e.formula is not None]
            gt_refs = list(
                {
                    e.paragraph_ref
                    for e in pymupdf_output.elements
                    if e.paragraph_ref is not None
                }
            )
            gt_headings = list(
                {e.title for e in pymupdf_output.elements if e.title is not None}
            )
            gt = GroundTruthAnnotation(
                pdf_path=str(pdf_path),
                tables=gt_tables,
                formulas=gt_formulas,
                paragraph_refs=gt_refs,
                section_headings=gt_headings,
            )
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(gt.model_dump_json(indent=2), encoding="utf-8")

        # Run each selected parser
        for system in selected_systems:
            if not _check_requirements(system):
                logger.warning(
                    "parser_skipped",
                    system=system,
                    reason="Missing required env vars",
                    required=SYSTEM_REQUIREMENTS[system],
                )
                continue

            # Check cache
            output = _load_cached(system, pdf_path)
            if output is None:
                logger.info("running_parser", system=system, pdf=str(pdf_path))
                try:
                    parser_func = PARSER_FUNCS[system]
                    assert callable(parser_func)
                    output = parser_func(pdf_path)
                    _save_cache(output)
                except Exception as e:
                    logger.error(
                        "parser_failed", system=system, pdf=str(pdf_path), error=str(e)
                    )
                    continue

            all_outputs.append(output)

            # Compute metrics
            pred_tables = [e.table for e in output.elements if e.table is not None]
            pred_formulas = [e.formula for e in output.elements if e.formula is not None]

            table_f1 = score_table_recovery_f1(pred_tables, gt.tables)
            formula_fid = score_formula_fidelity(pred_formulas, gt.formulas)
            para_recall = score_paragraph_ref_recall(output.elements, gt.paragraph_refs)
            sect_recall = score_section_recall(output.elements, gt.section_headings)

            latency_per_page = output.latency_ms / max(output.page_count, 1)
            cost_per_page = (
                output.cost_usd / max(output.page_count, 1)
                if output.cost_usd is not None
                else None
            )

            result = ParserMetricResult(
                system=system,
                pdf_path=str(pdf_path),
                table_recovery_f1=round(table_f1, 4),
                formula_fidelity=round(formula_fid, 4),
                paragraph_ref_recall=round(para_recall, 4),
                section_recall=round(sect_recall, 4),
                latency_ms_per_page=round(latency_per_page, 2),
                cost_usd_per_page=round(cost_per_page, 6) if cost_per_page is not None else None,
            )
            all_results.append(result)

            logger.info(
                "metrics_computed",
                system=system,
                pdf=pdf_path.name,
                table_f1=result.table_recovery_f1,
                formula_fid=result.formula_fidelity,
                para_recall=result.paragraph_ref_recall,
                sect_recall=result.section_recall,
                latency_ms_per_page=result.latency_ms_per_page,
            )

    # Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_json = [r.model_dump() for r in all_results]
    RESULTS_PATH.write_text(json.dumps(results_json, indent=2), encoding="utf-8")
    logger.info("results_written", path=str(RESULTS_PATH), count=len(all_results))

    return all_results


def print_results_table(results: list[ParserMetricResult]) -> None:
    """Print results as a Rich table to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fallback to plain text
        print("\n=== Parser Benchmark Results ===\n")
        for r in results:
            print(
                f"{r.system:20s} | {Path(r.pdf_path).name:25s} | "
                f"tbl_f1={r.table_recovery_f1:.3f} | "
                f"fml_fid={r.formula_fidelity:.3f} | "
                f"para_rec={r.paragraph_ref_recall:.3f} | "
                f"sect_rec={r.section_recall:.3f} | "
                f"lat_ms/pg={r.latency_ms_per_page:.1f} | "
                f"$/pg={r.cost_usd_per_page or 'free'}"
            )
        return

    console = Console()
    table = Table(title="Parser Benchmark Results", show_lines=True)

    table.add_column("System", style="cyan bold")
    table.add_column("PDF", style="dim")
    table.add_column("Table F1", justify="right", style="green")
    table.add_column("Formula Fid", justify="right", style="green")
    table.add_column("Para Recall", justify="right", style="green")
    table.add_column("Section Recall", justify="right", style="green")
    table.add_column("Latency ms/pg", justify="right", style="yellow")
    table.add_column("$/page", justify="right", style="red")

    for r in results:
        cost_str = f"${r.cost_usd_per_page:.4f}" if r.cost_usd_per_page is not None else "free"
        table.add_row(
            r.system,
            Path(r.pdf_path).name,
            f"{r.table_recovery_f1:.3f}",
            f"{r.formula_fidelity:.3f}",
            f"{r.paragraph_ref_recall:.3f}",
            f"{r.section_recall:.3f}",
            f"{r.latency_ms_per_page:.1f}",
            cost_str,
        )

    console.print(table)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default=os.environ.get("ANVIL_PARSER_BENCHMARK_SYSTEMS"),
        help=(
            "Comma-separated parser systems to run. Defaults to "
            "pymupdf4llm,naive_pdfminer,reducto. Azure DI is available only "
            "when explicitly requested."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_benchmark(_parse_systems(args.systems))
    if results:
        print_results_table(results)
    else:
        print("No results generated. Check that PDFs exist in data/parser_benchmark/pdfs/")
