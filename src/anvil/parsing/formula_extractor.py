"""Formula extraction — detect equations and their variable metadata."""

from __future__ import annotations

import re

from anvil.schemas.document import FormulaVariable, ParsedFormula

# Matches fenced code blocks that look like formulas (contains an '=')
_CODE_FENCE = re.compile(r"```(?:.*?)?\n(.*?)\n```", re.DOTALL)

# A simple right-hand-side detector: `<var> = <expr>`
# Allows zero or more spaces around '=' to handle pymupdf4llm's
# whitespace-stripped output (e.g. 't=(P×R)/(S×E−0.6×P)').
_EQUATION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$")

# Detect the "applicability sentence" that typically follows each formula
# in SPES-1, e.g.:
#   "This formula applies when: t ≤ R/2 AND P ≤ 0.385 × S × E."
#   "Applicable when P ≤ 0.385 × S × E."
#   "Applicable when t ≤ 0.356 × R AND P ≤ 0.665 × S × E."
# We capture the clause text up to the first sentence terminator. Multiple
# AND-joined conditions are split into separate entries so downstream
# consumers (refusal gate, evaluation) can reason about each one.
_APPLIES_RE = re.compile(
    r"(?:this\s+formula\s+applies\s+when|applies\s+when|applicable\s+when)"
    r"\s*[:\-]?\s*(.+?)(?=\.\s|\.$|\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_AND_SPLIT = re.compile(r"\s+(?:AND|and)\s+")


# Canonical SPES-1 variable metadata — used when a formula obviously
# matches one of the core thickness formulas.
_STD_VARIABLES: dict[str, FormulaVariable] = {
    "P": FormulaVariable(symbol="P", name="design pressure", unit="MPa", source="user_input"),
    "R": FormulaVariable(symbol="R", name="inside radius", unit="mm", source="derived from inside_diameter/2"),
    "Ro": FormulaVariable(symbol="Ro", name="outside radius", unit="mm", source="derived from outside_diameter/2"),
    "S": FormulaVariable(symbol="S", name="maximum allowable stress", unit="MPa", source="Table M-1"),
    "E": FormulaVariable(symbol="E", name="joint efficiency", unit="dimensionless", source="Table B-12"),
    "t": FormulaVariable(symbol="t", name="minimum required thickness", unit="mm", source="calculated"),
    "D": FormulaVariable(symbol="D", name="inside diameter", unit="mm", source="user_input"),
}


def _identify_variables(plain_text: str) -> list[FormulaVariable]:
    """Identify known single-letter variables in the formula text."""
    vars_found: list[FormulaVariable] = []
    seen: set[str] = set()
    # Sort by length descending so "Ro" matches before "R"
    for sym in sorted(_STD_VARIABLES.keys(), key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])")
        if pattern.search(plain_text) and sym not in seen:
            vars_found.append(_STD_VARIABLES[sym])
            seen.add(sym)
    return vars_found


def _to_latex(plain: str) -> str:
    """Best-effort plain-text → LaTeX. We keep this trivial; downstream does
    not depend on exact LaTeX correctness, only that we record a latex field."""
    s = plain.replace("×", r"\times ").replace("−", "-")
    # Wrap a simple `a = b / c` pattern as a fraction if we can
    m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*\(([^)]+)\)\s*/\s*\(([^)]+)\)\s*$", s)
    if m:
        lhs, num, den = m.group(1), m.group(2), m.group(3)
        return rf"{lhs} = \frac{{{num}}}{{{den}}}"
    return s


def _extract_applicability_conditions(region: str) -> list[str]:
    """Pull "applies when ..." / "applicable when ..." clauses from a region.

    AND-joined sub-conditions are emitted as separate list entries so the
    refusal gate, evaluation layer, and any downstream UI can reason
    about each individual constraint (e.g. ``"t ≤ R/2"`` vs.
    ``"P ≤ 0.385 × S × E"``).
    """
    out: list[str] = []
    for m in _APPLIES_RE.finditer(region):
        clause = m.group(1).strip().rstrip(".:")
        if not clause:
            continue
        for part in _AND_SPLIT.split(clause):
            cleaned = part.strip().rstrip(".:")
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def extract_formulas(
    source_paragraph: str,
    text: str,
) -> list[ParsedFormula]:
    """Extract formulas from a block of text (markdown-style).

    Looks for fenced code blocks that contain `<var> = <expr>`. Returns one
    ParsedFormula per equation found. The formula_id is derived from the
    paragraph and an index (e.g. `A-27(c)(1)-f0`).

    Applicability conditions are populated by scanning the region of
    `text` between this fence and the next one for sentences like
    "This formula applies when: t ≤ R/2 AND P ≤ 0.385 × S × E." The
    schema field is critical for compliance reasoning, so a missing
    sentence simply yields an empty list (it is never silently
    fabricated from defaults).
    """
    results: list[ParsedFormula] = []
    fences = list(_CODE_FENCE.finditer(text))
    idx = 0
    for fence_idx, fence in enumerate(fences):
        block = fence.group(1).strip()
        # The text region "owned" by this formula extends from the end of
        # this fence to the start of the next fence (or end of body).
        region_start = fence.end()
        region_end = (
            fences[fence_idx + 1].start()
            if fence_idx + 1 < len(fences)
            else len(text)
        )
        region = text[region_start:region_end]
        conditions = _extract_applicability_conditions(region)
        # A block may contain multiple lines; take the first equation-like line
        for line in block.splitlines():
            line = line.strip()
            m = _EQUATION.match(line)
            if not m:
                continue
            plain = line
            formula_id = f"{source_paragraph}-f{idx}"
            idx += 1
            results.append(
                ParsedFormula(
                    formula_id=formula_id,
                    latex=_to_latex(plain),
                    plain_text=plain,
                    variables=_identify_variables(plain),
                    applicability_conditions=conditions,
                    source_paragraph=source_paragraph,
                )
            )
            break  # one formula per fence is enough for SPES-1
    return results
