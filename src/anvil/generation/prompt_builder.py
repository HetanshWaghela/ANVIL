"""Assemble retrieved context into a structured prompt for the LLM."""

from __future__ import annotations

from anvil.schemas.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are anvil, a retrieval-augmented assistant for the
Synthetic Pressure Equipment Standard (SPES-1). You answer engineering
questions grounded strictly in the provided context. Rules:

1. Every factual claim MUST include a citation to a specific paragraph or
   table (e.g. "A-27(c)(1)" or "Table M-1").
2. If the context does not contain the information needed to answer,
   respond with confidence="insufficient" and explain what is missing.
3. NEVER invent material property values, formulas, or paragraph references.
4. NEVER perform arithmetic yourself — numeric results are computed
   deterministically by the host application and injected into the response.
5. Your response MUST conform to the `AnvilResponse` schema.
"""


def build_context_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    calculation_summary: str | None = None,
    max_chars_per_chunk: int = 1200,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM call.

    The context is grouped by type (formulas, tables, definitions) with each
    chunk carrying its paragraph reference. A pre-computed calculation summary
    can be injected; the LLM's job is to explain and cite, not to recompute.
    """
    by_type: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        by_type.setdefault(c.element_type, []).append(c)

    sections: list[str] = []
    ordering = ["formula", "table", "section", "paragraph", "definition", "note"]
    for et in ordering + [t for t in by_type if t not in ordering]:
        if et not in by_type:
            continue
        sections.append(f"## Retrieved {et.upper()}s")
        for c in by_type[et]:
            ref = f"[{c.paragraph_ref}]" if c.paragraph_ref else f"[{c.element_id}]"
            content = c.content.strip()
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "…"
            sections.append(f"### {ref} (page {c.page_number})\n{content}\n")
    context_block = "\n".join(sections) if sections else "(no relevant context retrieved)"

    calc_block = ""
    if calculation_summary:
        calc_block = (
            "\n## Precomputed Calculation (deterministic — do not modify)\n"
            f"{calculation_summary}\n"
        )

    user_prompt = (
        f"# Query\n{query}\n\n"
        f"# Retrieved Context\n{context_block}\n"
        f"{calc_block}\n"
        "# Task\n"
        "Answer the query using ONLY the context above. Include a citation "
        "(paragraph ref + exact quoted phrase) for every factual claim. If "
        "the context is insufficient, refuse with a specific reason."
    )
    return SYSTEM_PROMPT, user_prompt
