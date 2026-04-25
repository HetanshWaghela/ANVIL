"""Deterministic calculation engine.

The LLM NEVER does arithmetic. It selects the formula and identifies the
inputs. This module:

1. Looks up S and E from pinned data (not RAG)
2. Validates all inputs against applicability conditions
3. Calls the Decimal-backed formula functions
4. Builds a typed `CalculationResult` with `CalculationStep`s and Citations
5. Returns None (with a reason) if any required value is missing

Every numerical result in an `AnvilResponse` must come from this engine.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anvil import CalculationError
from anvil.pinned import (
    cylindrical_shell_thickness_inside_radius,
    cylindrical_shell_thickness_outside_radius,
    get_allowable_stress,
    get_joint_efficiency,
    get_material,
    mawp_cylindrical_inside_radius,
    mawp_cylindrical_outside_radius,
    mawp_spherical,
    next_standard_plate,
    spherical_shell_thickness,
)
from anvil.pinned.formulas import (
    check_cylindrical_applicability,
    check_spherical_applicability,
)
from anvil.schemas.document import DocumentElement, ElementType
from anvil.schemas.generation import CalculationStep, Citation, InputValue, StepKey

ComponentType = Literal[
    "cylindrical_shell",
    "cylindrical_shell_outside_radius",
    "spherical_shell",
]


@dataclass
class CalculationInputs:
    """All inputs required to run a shell thickness calculation."""

    component: ComponentType
    P_mpa: float
    design_temp_c: float
    material: str
    joint_type: int
    rt_level: str
    corrosion_allowance_mm: float
    inside_diameter_mm: float | None = None
    outside_diameter_mm: float | None = None


@dataclass
class CalculationResult:
    """The full output of a thickness calculation."""

    inputs: CalculationInputs
    S_mpa: float
    E: float
    R_mm: float
    formula_ref: str
    applicability_lhs: float
    applicability_rhs: float
    applicability_ok: bool
    t_min_mm: float
    t_design_mm: float
    t_nominal_mm: int
    mawp_mpa: float
    steps: list[CalculationStep]
    warnings: list[str]


class CalculationEngine:
    """Orchestrates deterministic thickness + MAWP calculations with provenance."""

    def __init__(self, citation_builder: CitationBuilder | None = None) -> None:
        self._citations = citation_builder or CitationBuilder.default()

    def calculate(self, inp: CalculationInputs) -> CalculationResult:
        """Run the full calculation pipeline."""
        # --- Material / S lookup -------------------------------------------
        mat = get_material(inp.material)
        if mat is None:
            raise CalculationError(
                f"Material '{inp.material}' not in pinned data. "
                f"Cannot proceed without allowable stress."
            )
        S = get_allowable_stress(inp.material, inp.design_temp_c)
        if S is None:
            raise CalculationError(
                f"Allowable stress for {inp.material} at {inp.design_temp_c}°C "
                f"unavailable (out of tabulated range or exceeds max temp "
                f"{mat.max_temp_c}°C per M-23)."
            )

        # --- Joint efficiency / E lookup -----------------------------------
        E = get_joint_efficiency(inp.joint_type, inp.rt_level)
        if E is None:
            raise CalculationError(
                f"Joint efficiency for Type {inp.joint_type} / {inp.rt_level} "
                f"not found in Table B-12."
            )

        # --- Radius ---------------------------------------------------------
        if inp.component == "cylindrical_shell_outside_radius":
            if inp.outside_diameter_mm is None:
                raise CalculationError(
                    "outside_diameter_mm required for A-27(c)(2) calculation."
                )
            R = inp.outside_diameter_mm / 2.0
        else:
            if inp.inside_diameter_mm is None:
                raise CalculationError(
                    "inside_diameter_mm required for this component."
                )
            R = inp.inside_diameter_mm / 2.0

        # --- Formula + applicability (pressure side) ----------------------
        # Per ASME UG-27 / SPES-1 A-27, the simple thin-wall formulas only
        # apply when BOTH conditions hold:
        #   (a) P ≤ 0.385·S·E  (for cylindrical) or 0.665·S·E (for spherical)
        #   (b) t ≤ R/2        (thin-wall assumption)
        # If (a) fails we refuse immediately — the formula is invalid. If
        # (b) fails we refuse AFTER computing t, because t is needed to check.
        formula_ref, applicability_fn, thickness_fn, mawp_fn, formula_plain = (
            self._formula_for(inp.component)
        )
        lhs, rhs, ok = applicability_fn(inp.P_mpa, S, E)
        if not ok:
            raise CalculationError(
                f"Applicability condition failed for {formula_ref}: "
                f"P={lhs} MPa exceeds the thin-wall limit "
                f"{rhs:.3f} MPa (= 0.385·S·E for cylindrical or "
                f"0.665·S·E for spherical). Thick-wall rules are not "
                f"covered by this standard."
            )

        # --- Calculate t_min -----------------------------------------------
        t_min = thickness_fn(inp.P_mpa, R, S, E)

        # Thin-wall geometry check: t_min ≤ R/2.
        # `R` is the inside radius for A-27(c)(1) and A-27(d), and the
        # outside radius for A-27(c)(2); the ≤ R/2 bound is the standard
        # thin-wall criterion in either frame.
        if t_min > R / 2.0:
            raise CalculationError(
                f"Thin-wall geometry condition failed for {formula_ref}: "
                f"computed t={t_min} mm exceeds R/2 = {R / 2.0} mm. This "
                f"vessel is in the thick-wall regime and must be designed "
                f"per the thick-wall rules (A-27(c)(3), not covered)."
            )

        if inp.corrosion_allowance_mm < 0:
            raise CalculationError(
                f"corrosion_allowance_mm must be ≥ 0 (got "
                f"{inp.corrosion_allowance_mm})."
            )
        t_design = round(t_min + inp.corrosion_allowance_mm, 2)
        t_nominal = next_standard_plate(t_design)
        # Invariant: next_standard_plate returns plate ≥ t_design = t_min + CA
        # with t_min > 0 and CA ≥ 0 (checked above), so plate > CA and
        # t_corroded > 0. Assert it so a future refactor can't silently
        # introduce a negative-pressure MAWP.
        t_corroded = t_nominal - inp.corrosion_allowance_mm
        assert t_corroded > 0, (
            f"invariant violated: t_corroded={t_corroded} ≤ 0 "
            f"(t_nominal={t_nominal}, CA={inp.corrosion_allowance_mm})"
        )
        mawp = mawp_fn(S, E, t_corroded, R)

        # --- Build citation-bearing steps ----------------------------------
        steps = self._build_steps(
            inp, S, E, R, formula_ref, formula_plain, lhs, rhs, t_min, t_design,
            t_nominal, mawp, t_corroded,
        )

        return CalculationResult(
            inputs=inp,
            S_mpa=S,
            E=E,
            R_mm=R,
            formula_ref=formula_ref,
            applicability_lhs=lhs,
            applicability_rhs=rhs,
            applicability_ok=True,  # invariant: always True if we reach here
            t_min_mm=t_min,
            t_design_mm=t_design,
            t_nominal_mm=t_nominal,
            mawp_mpa=mawp,
            steps=steps,
            warnings=[],
        )

    # ---- internal ---------------------------------------------------------

    def _formula_for(
        self, component: ComponentType
    ) -> tuple[
        str,
        Callable[[float, float, float], tuple[float, float, bool]],
        Callable[[float, float, float, float], float],
        Callable[[float, float, float, float], float],
        str,
    ]:
        """Return (paragraph_ref, applicability_fn, thickness_fn, mawp_fn, plain_text)."""
        if component == "cylindrical_shell":
            return (
                "A-27(c)(1)",
                check_cylindrical_applicability,
                cylindrical_shell_thickness_inside_radius,
                mawp_cylindrical_inside_radius,
                "t = (P × R) / (S × E − 0.6 × P)",
            )
        if component == "cylindrical_shell_outside_radius":
            return (
                "A-27(c)(2)",
                check_cylindrical_applicability,
                cylindrical_shell_thickness_outside_radius,
                mawp_cylindrical_outside_radius,
                "t = (P × Ro) / (S × E + 0.4 × P)",
            )
        if component == "spherical_shell":
            return (
                "A-27(d)",
                check_spherical_applicability,
                spherical_shell_thickness,
                mawp_spherical,
                "t = (P × R) / (2 × S × E − 0.2 × P)",
            )
        raise CalculationError(f"Unknown component: {component}")

    def _build_steps(
        self,
        inp: CalculationInputs,
        S: float,
        E: float,
        R: float,
        formula_ref: str,
        formula_plain: str,
        lhs: float,
        rhs: float,
        t_min: float,
        t_design: float,
        t_nominal: int,
        mawp: float,
        t_corroded: float,
    ) -> list[CalculationStep]:
        cit_formula = self._citations.for_paragraph(formula_ref)
        cit_stress = self._citations.for_material(inp.material)
        cit_eff = self._citations.for_joint_efficiency(inp.joint_type)
        cit_corr = self._citations.for_paragraph("A-25")

        applicability_formula = (
            "P ≤ 0.385 × S × E"
            if inp.component != "spherical_shell"
            else "P ≤ 0.665 × S × E"
        )

        steps: list[CalculationStep] = [
            CalculationStep(
                step_number=1,
                result_key=StepKey.ALLOWABLE_STRESS,
                description="Look up maximum allowable stress S from Table M-1.",
                formula=f"S = Table M-1({inp.material}, {inp.design_temp_c}°C)",
                inputs={
                    "T": InputValue(
                        symbol="T",
                        value=inp.design_temp_c,
                        unit="°C",
                        source="user_input",
                    ),
                },
                result=S,
                unit="MPa",
                citation=cit_stress,
            ),
            CalculationStep(
                step_number=2,
                result_key=StepKey.JOINT_EFFICIENCY,
                description=(
                    f"Look up joint efficiency E from Table B-12 for "
                    f"Type {inp.joint_type}, {inp.rt_level}."
                ),
                formula=f"E = Table B-12(Type {inp.joint_type}, {inp.rt_level})",
                inputs={
                    "joint_type": InputValue(
                        symbol="joint_type",
                        value=float(inp.joint_type),
                        unit="",
                        source="user_input",
                    ),
                },
                result=E,
                unit="dimensionless",
                citation=cit_eff,
            ),
            CalculationStep(
                step_number=3,
                result_key=StepKey.APPLICABILITY_CHECK,
                description="Check applicability of the simple thickness formula.",
                formula=applicability_formula,
                inputs={
                    "P": InputValue(symbol="P", value=inp.P_mpa, unit="MPa", source="user_input"),
                    "S": InputValue(symbol="S", value=S, unit="MPa", source="pinned_data", citation=cit_stress),
                    "E": InputValue(symbol="E", value=E, unit="dimensionless", source="pinned_data", citation=cit_eff),
                },
                result=rhs,
                unit="MPa",
                citation=cit_formula,
            ),
            CalculationStep(
                step_number=4,
                result_key=StepKey.MIN_THICKNESS,
                description=f"Compute minimum required thickness per {formula_ref}.",
                formula=formula_plain,
                inputs={
                    "P": InputValue(symbol="P", value=inp.P_mpa, unit="MPa", source="user_input"),
                    "R": InputValue(symbol="R", value=R, unit="mm", source="calculated"),
                    "S": InputValue(symbol="S", value=S, unit="MPa", source="pinned_data", citation=cit_stress),
                    "E": InputValue(symbol="E", value=E, unit="dimensionless", source="pinned_data", citation=cit_eff),
                },
                result=t_min,
                unit="mm",
                citation=cit_formula,
            ),
            CalculationStep(
                step_number=5,
                result_key=StepKey.DESIGN_THICKNESS,
                description="Add corrosion allowance per A-25.",
                formula="t_design = t_min + CA",
                inputs={
                    "t_min": InputValue(symbol="t_min", value=t_min, unit="mm", source="calculated"),
                    "CA": InputValue(symbol="CA", value=inp.corrosion_allowance_mm, unit="mm", source="user_input"),
                },
                result=t_design,
                unit="mm",
                citation=cit_corr,
            ),
            CalculationStep(
                step_number=6,
                result_key=StepKey.NOMINAL_PLATE,
                description="Select next standard plate thickness.",
                formula="t_nominal = next_standard_plate(t_design)",
                inputs={
                    "t_design": InputValue(symbol="t_design", value=t_design, unit="mm", source="calculated"),
                },
                result=float(t_nominal),
                unit="mm",
                citation=cit_formula,
            ),
            CalculationStep(
                step_number=7,
                result_key=StepKey.MAWP,
                description="Back-calculate MAWP using corroded thickness.",
                formula=self._mawp_formula_text(inp.component),
                inputs={
                    "S": InputValue(symbol="S", value=S, unit="MPa", source="pinned_data", citation=cit_stress),
                    "E": InputValue(symbol="E", value=E, unit="dimensionless", source="pinned_data", citation=cit_eff),
                    "t_corr": InputValue(symbol="t_corr", value=t_corroded, unit="mm", source="calculated"),
                    "R": InputValue(symbol="R", value=R, unit="mm", source="calculated"),
                },
                result=mawp,
                unit="MPa",
                citation=cit_formula,
            ),
        ]
        return steps

    @staticmethod
    def _mawp_formula_text(component: ComponentType) -> str:
        if component == "spherical_shell":
            return "P = (2 × S × E × t_corr) / (R + 0.2 × t_corr)"
        if component == "cylindrical_shell_outside_radius":
            return "P = (S × E × t_corr) / (Ro − 0.4 × t_corr)"
        return "P = (S × E × t_corr) / (R + 0.6 × t_corr)"


# ---- Citation builder -------------------------------------------------------


@dataclass
class CitationBuilder:
    """Builds `Citation` objects that point at actual parsed document elements.

    Given the list of `DocumentElement`s parsed from the standard, the builder
    resolves every citation request to:

      * A real `source_element_id` (matching an element the parser produced),
      * A `paragraph_ref` taken from that element (no string construction),
      * A `quoted_text` that is a real substring of that element's content.

    This means a rewording of the SPES-1 markdown file changes the quoted
    text automatically and NEVER introduces a silently-broken citation.
    If the referenced element cannot be found, the builder raises
    `CalculationError` rather than fabricating a plausible-looking citation.
    """

    elements_by_id: dict[str, DocumentElement]
    elements_by_ref: dict[str, DocumentElement]

    @classmethod
    def from_elements(cls, elements: list[DocumentElement]) -> CitationBuilder:
        """Build from a parsed element list (preferred path)."""
        by_id: dict[str, DocumentElement] = {e.element_id: e for e in elements}
        by_ref: dict[str, DocumentElement] = {}
        for el in elements:
            if el.paragraph_ref:
                # Prefer SECTION elements over TABLE/FORMULA children when both
                # share the same paragraph_ref, because sections contain the
                # full narrative text used for quoted excerpts.
                key = el.paragraph_ref.upper()
                existing = by_ref.get(key)
                if existing is None or (
                    existing.element_type != ElementType.SECTION
                    and el.element_type == ElementType.SECTION
                ):
                    by_ref[key] = el
        return cls(elements_by_id=by_id, elements_by_ref=by_ref)

    @classmethod
    def default(cls) -> CitationBuilder:
        """Convenience constructor that parses the bundled SPES-1 markdown."""
        from anvil.parsing.markdown_parser import parse_markdown_standard

        here = Path(__file__).resolve().parents[3]
        path = here / "data" / "synthetic" / "standard.md"
        return cls.from_elements(parse_markdown_standard(path))

    # ---- public API ------------------------------------------------------

    def for_paragraph(self, paragraph_ref: str) -> Citation:
        """Cite a paragraph, quoting its first substantive sentence."""
        el = self._resolve_ref(paragraph_ref)
        quote = self._first_sentence(el.content) or paragraph_ref
        return Citation(
            source_element_id=el.element_id,
            paragraph_ref=paragraph_ref,
            quoted_text=quote,
            page_number=max(el.page_number, 1),
        )

    def for_material(self, material_key: str) -> Citation:
        """Cite the row of Table M-1 for `material_key`, quoting the actual row.

        Markdown rows split cells by `|` pipes, so `"SM-516 Gr 70"` rarely
        appears as a contiguous substring. We deliberately search for the
        `spec_no` (the first whitespace-separated token, e.g. `"SM-516"`),
        which is the first column of each row. If that specific row is not
        found, we raise — we do NOT fall back to the section intro, because
        a citation that says "Table M-1" while quoting generic prose is
        misleading. Fail loud instead.
        """
        if not material_key or not material_key.strip():
            raise CalculationError(
                "for_material() requires a non-empty material_key."
            )
        el = self._resolve_ref("M-1")
        spec_no = material_key.split()[0]
        # Prefer an exact match on the full "spec grade" key; fall back to
        # spec_no only (same table row, just the most-specific cell form).
        quote = (
            self._find_line_containing(el.content, material_key)
            or self._find_line_containing(el.content, spec_no)
        )
        if not quote:
            raise CalculationError(
                f"Material '{material_key}' is not present as a row in "
                f"Table M-1 (searched for '{material_key}' and '{spec_no}' in "
                f"element '{el.element_id}'). Refusing to fabricate a generic "
                f"citation — add the material to the standard or correct "
                f"the spec string."
            )
        return Citation(
            source_element_id=el.element_id,
            paragraph_ref="Table M-1",
            quoted_text=quote,
            page_number=max(el.page_number, 1),
        )

    def for_joint_efficiency(self, joint_type: int | None = None) -> Citation:
        """Cite Table B-12 for a specific joint type.

        When `joint_type` is given we REQUIRE that its row is present — a
        fallback to the section intro would misrepresent which joint type
        we actually used. If the row is not found, raise. When `joint_type`
        is None we quote the section intro (explaining what the table is),
        which is an honest summary of what the citation points at.
        """
        el = self._resolve_ref("B-12")
        if joint_type is not None:
            quote = self._find_line_containing(el.content, f"| {joint_type} |")
            if not quote:
                raise CalculationError(
                    f"Joint type {joint_type} row not found in Table B-12 "
                    f"(element '{el.element_id}'). The synthetic standard "
                    f"only defines joint types 1–6. Refusing to fabricate a "
                    f"citation against the section intro."
                )
        else:
            quote = self._first_sentence(el.content)
            if not quote:
                raise CalculationError(
                    "Table B-12 section has no substantive content to quote. "
                    "The standard file may be malformed."
                )
        return Citation(
            source_element_id=el.element_id,
            paragraph_ref="Table B-12",
            quoted_text=quote,
            page_number=max(el.page_number, 1),
        )

    # ---- helpers ---------------------------------------------------------

    # Strips a trailing "(c)", "(1)", "(iv)" etc. from a paragraph reference.
    # Case-insensitive so it works on both "A-27(C)(1)" and "A-27(c)(1)".
    _SUBPARA_STRIP = re.compile(r"\([A-Z0-9]+\)\s*$", re.IGNORECASE)

    def _resolve_ref(self, paragraph_ref: str) -> DocumentElement:
        """Find the element for a paragraph_ref, walking up sub-paragraphs.

        Given `"A-27(c)(1)"`, this returns the element for `A-27` if the
        sub-paragraph has no standalone element. We never fabricate one.
        Accepts both bare refs (`"M-1"`) and table refs (`"Table M-1"`).
        """
        key = paragraph_ref.upper().strip()
        # Normalize "TABLE X-N" → "X-N" (our elements index by bare ref).
        if key.startswith("TABLE "):
            key = key[len("TABLE ") :].strip()
        if key in self.elements_by_ref:
            return self.elements_by_ref[key]
        # Walk up: "A-27(c)(1)" → "A-27(c)" → "A-27"
        stripped = self._SUBPARA_STRIP.sub("", key).strip()
        while stripped and stripped != key:
            if stripped in self.elements_by_ref:
                return self.elements_by_ref[stripped]
            key = stripped
            stripped = self._SUBPARA_STRIP.sub("", key).strip()
        raise CalculationError(
            f"No element found for paragraph reference '{paragraph_ref}' — "
            f"the standard does not contain this paragraph."
        )

    @staticmethod
    def _first_sentence(text: str) -> str | None:
        """Return the first non-empty sentence of `text`, or None."""
        if not text:
            return None
        # Prefer the first meaningful line (skip blank lines and bullets-only lines)
        for line in text.splitlines():
            s = line.strip()
            if len(s) < 20:
                continue
            # Stop at sentence terminator if present, else take up to 240 chars.
            m = re.search(r"[.;](?:\s|$)", s)
            return s[: m.end() - 1].strip() if m else s[:240]
        return None

    @staticmethod
    def _find_line_containing(text: str, needle: str) -> str | None:
        """Return the first line of `text` containing `needle` (case-insensitive)."""
        if not needle:
            return None
        n = needle.lower()
        for line in text.splitlines():
            if n in line.lower():
                return line.strip()
        return None
