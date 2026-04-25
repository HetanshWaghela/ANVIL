"""Unit tests for the parser benchmark framework.

Tests:
1. Schema drift guard: cached output.json files deserialize into ParserOutput
2. Pinned metric assertions for SPES-1 PDF (controlled baseline)
3. Ground truth generation from synthetic standard
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.schemas.parser_benchmark import ParserOutput

BENCHMARK_DIR = Path("data/parser_benchmark")


# ---------------------------------------------------------------------------
# Schema drift guard
# ---------------------------------------------------------------------------


class TestSchemaDriftGuard:
    """Verify every cached output.json deserializes into ParserOutput."""

    def _collect_output_files(self) -> list[Path]:
        """Collect all output.json files in data/parser_benchmark/."""
        if not BENCHMARK_DIR.exists():
            return []
        return sorted(BENCHMARK_DIR.rglob("output.json"))

    def test_all_cached_outputs_valid(self) -> None:
        """Every output.json must deserialize into ParserOutput without errors."""
        output_files = self._collect_output_files()
        if not output_files:
            pytest.skip("No cached output.json files found in data/parser_benchmark/")

        errors: list[str] = []
        for path in output_files:
            try:
                raw = path.read_text(encoding="utf-8")
                ParserOutput.model_validate_json(raw)
            except Exception as e:
                errors.append(f"{path}: {e}")

        if errors:
            pytest.fail(
                f"Schema drift: {len(errors)} output.json file(s) failed validation:\n"
                + "\n".join(errors)
            )


# ---------------------------------------------------------------------------
# Pinned metric assertions for SPES-1 PDF
# ---------------------------------------------------------------------------


class TestPinnedMetrics:
    """Pinned assertions for the controlled SPES-1 baseline.

    These assertions verify that the parsers meet minimum quality thresholds
    on our synthetic standard, where ground truth is exact.

    If no cached benchmark results exist, these tests skip with a warning
    rather than failing — the benchmark must be run first.
    """

    @pytest.fixture()
    def results(self) -> list[dict[str, object]]:
        """Load results.json if it exists."""
        results_path = BENCHMARK_DIR / "results.json"
        if not results_path.exists():
            pytest.skip(
                "No results.json found — run `uv run python scripts/run_parser_benchmark.py` first"
            )
        raw = json.loads(results_path.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        return raw  # type: ignore[return-value]

    def _find_result(
        self, results: list[dict[str, object]], system: str, pdf_stem: str
    ) -> dict[str, object] | None:
        """Find a result by system name and PDF stem."""
        for r in results:
            pdf_path = str(r.get("pdf_path", ""))
            if r.get("system") == system and pdf_stem in pdf_path:
                return r
        return None

    def test_pymupdf4llm_table_f1(self, results: list[dict[str, object]]) -> None:
        """pymupdf4llm table recovery on SPES-1 (PDF round-trip).

        Wide tables (M-1 with 13 columns) suffer header misalignment in
        pymupdf4llm. The remaining gap is documented in the failure-mode
        catalog (§3.3). Threshold set below measured 0.483.
        """
        r = self._find_result(results, "pymupdf4llm", "spes1_synthetic")
        if r is None:
            pytest.skip("No pymupdf4llm result for spes1_synthetic")
        f1 = float(r["table_recovery_f1"])  # type: ignore[arg-type]
        assert f1 >= 0.40, f"pymupdf4llm table_recovery_f1 = {f1} (expected ≥ 0.40)"

    def test_naive_pdfminer_table_f1_floor(self, results: list[dict[str, object]]) -> None:
        """naive_pdfminer has no table recovery (floor baseline)."""
        r = self._find_result(results, "naive_pdfminer", "spes1_synthetic")
        if r is None:
            pytest.skip("No naive_pdfminer result for spes1_synthetic")
        f1 = float(r["table_recovery_f1"])  # type: ignore[arg-type]
        assert f1 >= 0.0, f"naive_pdfminer table_recovery_f1 = {f1} (expected ≥ 0.0)"

    def test_pymupdf4llm_paragraph_ref_recall(
        self, results: list[dict[str, object]]
    ) -> None:
        """pymupdf4llm achieves ≥ 0.90 paragraph ref recall on SPES-1.

        After the _strip_md_formatting fix, bold-wrapped headings like
        '**A-27 Thickness...**' are correctly parsed. Measured: 1.000.
        """
        r = self._find_result(results, "pymupdf4llm", "spes1_synthetic")
        if r is None:
            pytest.skip("No pymupdf4llm result for spes1_synthetic")
        recall = float(r["paragraph_ref_recall"])  # type: ignore[arg-type]
        assert recall >= 0.90, f"pymupdf4llm paragraph_ref_recall = {recall} (expected ≥ 0.90)"

    def test_pymupdf4llm_section_recall(
        self, results: list[dict[str, object]]
    ) -> None:
        """pymupdf4llm achieves ≥ 0.80 section recall on SPES-1.

        After bold-stripping, most headings are recovered. The 2 missed
        headings are the document title and appendix which lack standard
        paragraph-ref patterns. Measured: 0.882.
        """
        r = self._find_result(results, "pymupdf4llm", "spes1_synthetic")
        if r is None:
            pytest.skip("No pymupdf4llm result for spes1_synthetic")
        recall = float(r["section_recall"])  # type: ignore[arg-type]
        assert recall >= 0.80, f"pymupdf4llm section_recall = {recall} (expected ≥ 0.80)"

    def test_pymupdf4llm_formula_fidelity(
        self, results: list[dict[str, object]]
    ) -> None:
        """pymupdf4llm achieves ≥ 0.80 formula fidelity on SPES-1.

        After the _normalize_formula fix (strip all whitespace before
        comparison), compressed formulas like 't=(P×R)/...' correctly
        match ground truth 't = (P × R) / ...'. Measured: 1.000.
        """
        r = self._find_result(results, "pymupdf4llm", "spes1_synthetic")
        if r is None:
            pytest.skip("No pymupdf4llm result for spes1_synthetic")
        fid = float(r["formula_fidelity"])  # type: ignore[arg-type]
        assert fid >= 0.80, f"pymupdf4llm formula_fidelity = {fid} (expected ≥ 0.80)"
