"""Formula extraction — detect equations and their variable metadata."""

from __future__ import annotations

import re

from anvil.schemas.document import FormulaVariable, ParsedFormula

# Matches fenced code blocks that look like formulas (contains an '=')
_CODE_FENCE = re.compile(r"```(?:.*?)?\n(.*?)\n```", re.DOTALL)

# A simple right-hand-side detector: `<var> = <expr>`
_EQUATION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$")


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


def extract_formulas(
    source_paragraph: str,
    text: str,
) -> list[ParsedFormula]:
    """Extract formulas from a block of text (markdown-style).

    Looks for fenced code blocks that contain `<var> = <expr>`. Returns one
    ParsedFormula per equation found. The formula_id is derived from the
    paragraph and an index (e.g. `A-27(c)(1)-f0`).
    """
    results: list[ParsedFormula] = []
    idx = 0
    for fence in _CODE_FENCE.finditer(text):
        block = fence.group(1).strip()
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
                    applicability_conditions=[],
                    source_paragraph=source_paragraph,
                )
            )
            break  # one formula per fence is enough for SPES-1
    return results
