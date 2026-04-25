"""Feature-flagged hybrid PDF parser prototype.

The default ANVIL PDF path is intentionally conservative: local
`pymupdf4llm` conversion followed by the canonical Markdown parser. The
parser benchmark in `docs/parser_benchmark.md` shows why this is the
right zero-key baseline: it is local, reproducible, structurally useful,
and good enough on the controlled SPES-1 PDF after the table-repair fixes.

This module adds a small *prototype* switching layer for future parser
bake-offs without changing downstream ingestion code. It supports:

* `pymupdf4llm` — local default; no network or credentials.
* `hybrid` — try configured hosted parser(s), then fall back to local.
* `reducto` — hosted parser via Reducto API when `REDUCTO_API_KEY` is set.
* `azure_di` — declared but intentionally fail-loud until the Azure adapter
  is implemented and benchmarked.

The surface is deliberately narrow: every provider must return the same
`list[DocumentElement]` produced by `parse_markdown_standard`, so retrieval,
KG construction, citation enforcement, and evaluation remain provider-agnostic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from anvil import ParsingError
from anvil.logging_config import get_logger
from anvil.parsing.markdown_parser import parse_markdown_standard
from anvil.parsing.pdf_parser import parse_pdf
from anvil.schemas.document import DocumentElement

ParserProvider = Literal["pymupdf4llm", "hybrid", "reducto", "azure_di"]

DEFAULT_PROVIDER: ParserProvider = "pymupdf4llm"
ENV_PROVIDER = "ANVIL_PDF_PARSER"
ENV_REDUCTO_API_KEY = "REDUCTO_API_KEY"
ENV_AZURE_DI_KEY = "AZURE_DOCUMENT_INTELLIGENCE_KEY"
ENV_AZURE_DI_ENDPOINT = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"

log = get_logger(__name__)


@dataclass(frozen=True)
class HybridParserConfig:
    """Configuration for the feature-flagged PDF parser prototype.

    `provider` controls the primary path. The default is the local parser.
    `fallback_to_local` applies only to hosted providers and `hybrid`; local
    parsing itself still fails loud if `pymupdf4llm` is unavailable or the file
    cannot be parsed.
    """

    provider: ParserProvider = DEFAULT_PROVIDER
    fallback_to_local: bool = True
    reducto_api_key: str | None = None
    azure_di_key: str | None = None
    azure_di_endpoint: str | None = None
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> HybridParserConfig:
        """Build parser config from environment variables.

        `ANVIL_PDF_PARSER` may be one of:
        `pymupdf4llm`, `hybrid`, `reducto`, or `azure_di`.
        """
        raw_provider = os.environ.get(ENV_PROVIDER, DEFAULT_PROVIDER).strip()
        provider = _parse_provider(raw_provider)
        return cls(
            provider=provider,
            reducto_api_key=os.environ.get(ENV_REDUCTO_API_KEY),
            azure_di_key=os.environ.get(ENV_AZURE_DI_KEY),
            azure_di_endpoint=os.environ.get(ENV_AZURE_DI_ENDPOINT),
        )


def parse_hybrid_pdf(
    pdf_path: str | Path,
    config: HybridParserConfig | None = None,
) -> list[DocumentElement]:
    """Parse a PDF using the configured provider.

    This is a prototype switching layer, not a new downstream representation.
    All successful paths normalize back into `DocumentElement` instances via
    `parse_markdown_standard`.

    Provider behavior:

    * `pymupdf4llm`: local parser only.
    * `reducto`: hosted Reducto parser; optionally falls back to local.
    * `azure_di`: fail-loud placeholder until the adapter is implemented.
    * `hybrid`: Reducto when keyed, then local. Azure DI is not attempted by
      default because it is not part of the benchmarked production path yet.
    """
    cfg = config or HybridParserConfig.from_env()
    path = Path(pdf_path)
    if not path.exists():
        raise ParsingError(f"PDF not found: {path}")

    if cfg.provider == "pymupdf4llm":
        return _parse_local(path)

    if cfg.provider == "reducto":
        return _parse_hosted_with_optional_fallback(
            provider="reducto",
            path=path,
            cfg=cfg,
        )

    if cfg.provider == "azure_di":
        return _parse_hosted_with_optional_fallback(
            provider="azure_di",
            path=path,
            cfg=cfg,
        )

    if cfg.provider == "hybrid":
        return _parse_hybrid(path, cfg)

    raise ParsingError(f"Unsupported parser provider: {cfg.provider}")


def _parse_provider(raw_provider: str) -> ParserProvider:
    normalized = raw_provider.strip().lower().replace("-", "_")
    if normalized in {"pymupdf", "pymupdf4llm", "local"}:
        return "pymupdf4llm"
    if normalized == "hybrid":
        return "hybrid"
    if normalized == "reducto":
        return "reducto"
    if normalized in {"azure", "azure_di", "azure_document_intelligence"}:
        return "azure_di"
    raise ParsingError(
        f"Unknown {ENV_PROVIDER}={raw_provider!r}. "
        "Expected one of: pymupdf4llm, hybrid, reducto, azure_di."
    )


def _parse_local(path: Path) -> list[DocumentElement]:
    log.info("hybrid_parser.local.start", provider="pymupdf4llm", pdf=str(path))
    elements = parse_pdf(path)
    log.info(
        "hybrid_parser.local.done",
        provider="pymupdf4llm",
        pdf=str(path),
        elements=len(elements),
    )
    return elements


def _parse_hybrid(path: Path, cfg: HybridParserConfig) -> list[DocumentElement]:
    """Prototype policy: try Reducto when keyed, otherwise local.

    The parser benchmark currently defends `pymupdf4llm` as the default. Reducto
    is kept behind a key gate until a real benchmark row exists. Azure DI is not
    attempted in hybrid mode yet because it is explicitly opt-in in the plan and
    docs.
    """
    if cfg.reducto_api_key:
        try:
            return _parse_reducto(path, cfg)
        except ParsingError as exc:
            if not cfg.fallback_to_local:
                raise
            log.warning(
                "hybrid_parser.reducto_fallback",
                pdf=str(path),
                error=str(exc),
            )

    log.info(
        "hybrid_parser.hybrid_local_fallback",
        pdf=str(path),
        reason="no hosted parser produced elements",
    )
    return _parse_local(path)


def _parse_hosted_with_optional_fallback(
    *,
    provider: Literal["reducto", "azure_di"],
    path: Path,
    cfg: HybridParserConfig,
) -> list[DocumentElement]:
    try:
        if provider == "reducto":
            return _parse_reducto(path, cfg)
        return _parse_azure_di(path, cfg)
    except ParsingError as exc:
        if not cfg.fallback_to_local:
            raise
        log.warning(
            "hybrid_parser.hosted_fallback",
            provider=provider,
            pdf=str(path),
            error=str(exc),
        )
        return _parse_local(path)


def _parse_reducto(path: Path, cfg: HybridParserConfig) -> list[DocumentElement]:
    """Parse through Reducto and normalize returned Markdown.

    This intentionally uses Reducto's HTTP API directly instead of adding a hard
    dependency on an SDK. If Reducto's response schema changes, this function
    fails loud with a clear `ParsingError` instead of silently returning empty
    elements.
    """
    api_key = cfg.reducto_api_key
    if not api_key:
        raise ParsingError(f"Reducto parser requested but {ENV_REDUCTO_API_KEY} is not set.")

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a project dependency
        raise ParsingError("httpx is required for the Reducto parser.") from exc

    log.info("hybrid_parser.reducto.start", pdf=str(path))
    with httpx.Client(timeout=cfg.timeout_s) as client:
        upload_response = client.post(
            "https://platform.reducto.ai/upload",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (path.name, path.read_bytes(), "application/pdf")},
        )
        if upload_response.status_code >= 400:
            raise ParsingError(
                "Reducto upload failed: "
                f"HTTP {upload_response.status_code}: {upload_response.text[:300]}"
            )

        upload_payload = _json_object(upload_response)
        file_ref = _first_string(upload_payload, ("file_id", "document_url", "url"))
        if not file_ref:
            raise ParsingError(
                "Reducto upload response did not contain `file_id`, `document_url`, or `url`."
            )

        parse_response = client.post(
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
        if parse_response.status_code >= 400:
            raise ParsingError(
                "Reducto parse failed: "
                f"HTTP {parse_response.status_code}: {parse_response.text[:300]}"
            )

    parse_payload = _json_object(parse_response)
    markdown = _extract_reducto_markdown(parse_payload)
    if not markdown.strip():
        raise ParsingError("Reducto parse response contained no Markdown content.")

    elements = parse_markdown_standard(markdown)
    if not elements:
        raise ParsingError("Reducto Markdown normalized to zero DocumentElements.")

    log.info(
        "hybrid_parser.reducto.done",
        pdf=str(path),
        elements=len(elements),
    )
    return elements


def _parse_azure_di(path: Path, cfg: HybridParserConfig) -> list[DocumentElement]:
    """Placeholder for Azure Document Intelligence.

    Azure DI remains opt-in in the benchmark plan and is not the default parser
    path. This function is intentionally fail-loud until a tested adapter exists;
    otherwise setting `ANVIL_PDF_PARSER=azure_di` would create a false sense of
    support and potentially ship unbenchmarked parser output into retrieval.
    """
    if not cfg.azure_di_key or not cfg.azure_di_endpoint:
        raise ParsingError(
            "Azure DI parser requested but "
            f"{ENV_AZURE_DI_KEY} and/or {ENV_AZURE_DI_ENDPOINT} is unset."
        )
    raise ParsingError(
        "Azure DI parser is declared but not implemented yet. "
        "Run `scripts/run_parser_benchmark.py --systems ...azure_di` after "
        "adding the adapter, then update docs/parser_benchmark.md before "
        "enabling this provider in production."
    )


def _json_object(response: object) -> Mapping[str, object]:
    """Return response JSON as a mapping or fail loud.

    Kept small to avoid propagating untyped provider payloads through the rest
    of the parser.
    """
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        raise ParsingError("HTTP response object does not expose json().")
    payload = cast(object, json_method())
    if not isinstance(payload, Mapping):
        raise ParsingError("Provider response JSON was not an object.")
    return cast(Mapping[str, object], payload)


def _first_string(payload: Mapping[str, object], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_reducto_markdown(payload: Mapping[str, object]) -> str:
    """Extract Markdown content from known Reducto response shapes."""
    direct_markdown = _first_string(payload, ("markdown", "content"))
    if direct_markdown:
        return direct_markdown

    result = payload.get("result")
    if isinstance(result, Mapping):
        result_markdown = _first_string(cast(Mapping[str, object], result), ("markdown", "content"))
        if result_markdown:
            return result_markdown

        chunks = result.get("chunks")
        if isinstance(chunks, Sequence) and not isinstance(chunks, str):
            parts: list[str] = []
            for chunk in chunks:
                if isinstance(chunk, Mapping):
                    content = chunk.get("content")
                    if isinstance(content, str) and content.strip():
                        parts.append(content)
            if parts:
                return "\n\n".join(parts)

    chunks = payload.get("chunks")
    if isinstance(chunks, Sequence) and not isinstance(chunks, str):
        parts = []
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                content = chunk.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
        if parts:
            return "\n\n".join(parts)

    return ""
