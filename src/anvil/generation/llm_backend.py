"""Pluggable LLM backends.

The generation layer is built around a `LLMBackend` protocol so we can:

  * Ship a `FakeLLMBackend` for deterministic tests and offline CI (it returns
    structured responses assembled from the retrieved context and the
    calculation engine output — no LLM required).
  * Use `InstructorBackend` (optional dependency) for real LLM calls against
    any provider supported by `instructor.from_provider`.
  * Use `OpenAICompatibleBackend` for any OpenAI-protocol endpoint that is
    NOT reachable via `instructor.from_provider` — specifically NVIDIA NIM
    (`https://integrate.api.nvidia.com/v1`), Together, Fireworks, vLLM,
    Ollama, LM Studio, etc. The only thing that changes is the base URL,
    API-key env var, and model name.

Tests use `FakeLLMBackend` exclusively. Production users select a real
backend via environment variables. `get_default_backend()` logs loudly
which backend was selected so that a prod deploy which forgot to
configure one sees a clear warning rather than a silent fake response.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Protocol

from anvil import GenerationError, RetryableGenerationError
from anvil.logging_config import get_logger
from anvil.schemas.generation import (
    AnvilResponse,
    CalculationStep,
    Citation,
    LLMAnvilResponse,
    ResponseConfidence,
    StepKey,
)
from anvil.schemas.retrieval import RetrievedChunk

log = get_logger(__name__)


class LLMBackend(Protocol):
    """Minimal contract for a generation backend."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list[CalculationStep],
        refusal_reason: str | None = None,
    ) -> AnvilResponse: ...


class FakeLLMBackend:
    """A deterministic, dependency-free backend.

    Does not call any LLM. Given the retrieved context, pre-computed
    calculation steps, and an optional refusal reason, it assembles a valid
    `AnvilResponse`. This is what the test suite uses — the prompt builder,
    citation enforcer, and refusal gate are all exercised end-to-end, but
    no network call is made.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list[CalculationStep],
        refusal_reason: str | None = None,
    ) -> AnvilResponse:
        if refusal_reason is not None:
            return AnvilResponse(
                query=query,
                answer="I cannot answer this query with the available context.",
                citations=[],
                calculation_steps=[],
                confidence=ResponseConfidence.INSUFFICIENT,
                refusal_reason=refusal_reason,
                retrieved_context_ids=[c.element_id for c in retrieved_chunks],
            )

        # If we have calculation steps, build a calculation answer.
        # Index by the typed result_key instead of parsing description strings.
        if calculation_steps:
            by_key: dict[StepKey, CalculationStep] = {
                s.result_key: s for s in calculation_steps
            }
            parts: list[str] = []
            if StepKey.MIN_THICKNESS in by_key:
                parts.append(
                    f"Minimum required thickness t_min = "
                    f"{by_key[StepKey.MIN_THICKNESS].result:.2f} mm"
                )
            if StepKey.DESIGN_THICKNESS in by_key:
                parts.append(
                    f"design thickness (with corrosion allowance) = "
                    f"{by_key[StepKey.DESIGN_THICKNESS].result:.2f} mm"
                )
            if StepKey.NOMINAL_PLATE in by_key:
                parts.append(
                    f"selected nominal plate = "
                    f"{int(by_key[StepKey.NOMINAL_PLATE].result)} mm"
                )
            if StepKey.MAWP in by_key:
                parts.append(
                    f"MAWP (back-calculated) = "
                    f"{by_key[StepKey.MAWP].result:.3f} MPa"
                )
            answer = "; ".join(parts) + "."

            calc_citations: list[Citation] = list({
                (s.citation.source_element_id, s.citation.paragraph_ref): s.citation
                for s in calculation_steps
            }.values())

            return AnvilResponse(
                query=query,
                answer=answer,
                citations=calc_citations,
                calculation_steps=calculation_steps,
                confidence=ResponseConfidence.HIGH,
                retrieved_context_ids=[c.element_id for c in retrieved_chunks],
            )

        # Non-calculation lookup: return a cited summary composed from the
        # top-K retrieved chunks. Producing a citation per distinct
        # paragraph_ref makes downstream structural-completeness scoring
        # honest — we explicitly claim the paragraphs we are using.
        if not retrieved_chunks:
            return AnvilResponse(
                query=query,
                answer="No context retrieved.",
                confidence=ResponseConfidence.INSUFFICIENT,
                refusal_reason="No context retrieved.",
                retrieved_context_ids=[],
            )

        # Take up to 4 distinct paragraph_ref chunks with the highest scores
        seen_refs: set[str] = set()
        picks: list[RetrievedChunk] = []
        for c in retrieved_chunks:
            ref_key = (c.paragraph_ref or c.element_id).upper()
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            picks.append(c)
            if len(picks) >= 4:
                break

        citations: list[Citation] = []
        for c in picks:
            first_line = next(
                (line.strip() for line in c.content.splitlines() if line.strip()),
                "",
            )
            if not first_line:
                continue
            citations.append(
                Citation(
                    source_element_id=c.element_id,
                    paragraph_ref=c.paragraph_ref or "SPES-1",
                    quoted_text=first_line[:240],
                    page_number=max(c.page_number, 1),
                )
            )

        top = next(
            (pick for pick in picks if pick.content.strip()),
            picks[0],
        )
        top_line = next(
            (line.strip() for line in top.content.splitlines() if line.strip()),
            top.element_id,
        )
        return AnvilResponse(
            query=query,
            answer=top_line,
            citations=citations,
            confidence=ResponseConfidence.MEDIUM,
            retrieved_context_ids=[c.element_id for c in retrieved_chunks],
        )


class InstructorBackend:
    """Real LLM backend using `instructor.from_provider`.

    Lazy-imports instructor so tests can run without the dependency. If the
    provider is configured but instructor is missing, we raise immediately
    rather than silently falling back to a fake response.
    """

    def __init__(self, model: str) -> None:
        if not model:
            raise GenerationError(
                "InstructorBackend requires a non-empty `model` argument "
                "(e.g. 'openai/gpt-4o-mini'). Silent defaults removed."
            )
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:  # pragma: no cover - network dependent
        if self._client is not None:
            return self._client
        try:
            import instructor
        except ImportError as exc:
            raise GenerationError(
                "instructor package is required for InstructorBackend. "
                "Install with `uv add instructor`. Refusing to fall back to "
                "a fake response silently."
            ) from exc
        self._client = instructor.from_provider(self.model, async_client=True)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list[CalculationStep],
        refusal_reason: str | None = None,
    ) -> AnvilResponse:  # pragma: no cover - network dependent
        if refusal_reason is not None:
            return AnvilResponse(
                query=query,
                answer="Refused.",
                confidence=ResponseConfidence.INSUFFICIENT,
                refusal_reason=refusal_reason,
                retrieved_context_ids=[c.element_id for c in retrieved_chunks],
            )
        client = self._get_client()
        llm_response: LLMAnvilResponse = await client.create(
            response_model=LLMAnvilResponse,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_retries=2,
        )
        # Always inject the deterministic calc steps — never trust LLM arithmetic.
        return AnvilResponse(
            **llm_response.model_dump(),
            calculation_steps=calculation_steps,
        )


class OpenAICompatibleBackend:
    """Backend for any OpenAI-protocol endpoint (NVIDIA NIM, Together, vLLM…).

    Works with any server that implements OpenAI's Chat Completions API by
    accepting a custom `base_url` and `api_key`. Uses `instructor` in JSON
    mode for structured `AnvilResponse` output — tool-use isn't universally
    supported across NIM/vLLM/Ollama, but JSON-mode is.

    Example — NVIDIA NIM:

        backend = OpenAICompatibleBackend(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
            model="deepseek-ai/deepseek-v4-flash",
        )

    Fail-loud behavior:
      * Missing `openai` or `instructor` → GenerationError at construction.
      * Missing `api_key` → GenerationError at construction.
      * LLM call failure → re-raises as GenerationError with the underlying
        exception chained.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        api_keys: list[str] | None = None,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise GenerationError("OpenAICompatibleBackend requires a base_url.")
        if not model:
            raise GenerationError("OpenAICompatibleBackend requires a model.")
        candidate_keys = _dedupe_api_keys(api_keys or ([api_key] if api_key else []))
        if not candidate_keys:
            raise GenerationError(
                "OpenAICompatibleBackend requires an api_key. Pass one "
                "explicitly or set the appropriate env var (e.g. "
                "NVIDIA_API_KEY for NVIDIA NIM). Refusing to construct a "
                "backend with no credential — network calls would fail "
                "with opaque 401s far from the configuration site."
            )
        self.base_url = base_url
        self._api_keys = candidate_keys
        self._api_key_index = 0
        self.api_key = candidate_keys[self._api_key_index]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.extra_body = extra_body
        self._client: Any = None
        self.requests_per_minute = float(
            os.environ.get("ANVIL_LLM_REQUESTS_PER_MINUTE", "35")
        )
        self.max_retries = int(os.environ.get("ANVIL_LLM_MAX_RETRIES", "1"))
        self.round_robin_keys = os.environ.get(
            "ANVIL_LLM_ROUND_ROBIN_KEYS", "true"
        ).lower() in {"1", "true", "yes"}
        self._rate_lock = asyncio.Lock()
        self._last_request_at_by_key: dict[int, float] = {}

    def _get_client(self) -> Any:  # pragma: no cover - network dependent
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise GenerationError(
                "openai package is required for OpenAICompatibleBackend. "
                "Install with `uv add openai`."
            ) from exc
        try:
            import instructor
        except ImportError as exc:
            raise GenerationError(
                "instructor package is required for OpenAICompatibleBackend. "
                "Install with `uv add instructor`."
            ) from exc
        raw = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
        )
        # JSON mode works across NIM, Together, Fireworks, vLLM, Ollama.
        # Tool-use mode varies per provider; JSON is the lowest common denom.
        self._client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
        return self._client

    def _rotate_api_key(self, *, reason: str) -> bool:
        """Switch to the next configured API key without logging secrets."""
        if len(self._api_keys) <= 1:
            return False
        self._api_key_index = (self._api_key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._api_key_index]
        self._client = None
        log.warning(
            "llm_backend.api_key_rotated",
            model=self.model,
            key_index=self._api_key_index + 1,
            total_keys=len(self._api_keys),
            reason=reason,
        )
        return True

    def _advance_api_key_for_next_request(self) -> None:
        """Round-robin successful requests across configured keys."""
        if not self.round_robin_keys or len(self._api_keys) <= 1:
            return
        self._api_key_index = (self._api_key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._api_key_index]
        self._client = None

    async def throttle(self) -> None:
        """Apply process-local provider pacing before each live request.

        NIM free-tier limits are commonly around 35-40 requests/minute. We
        default to 35 rpm per configured key and make it env-configurable so
        headline runs do not accidentally convert provider throttling into bad
        benchmark answers.
        """
        if self.requests_per_minute <= 0:
            return
        min_interval_s = 60.0 / self.requests_per_minute
        async with self._rate_lock:
            now = time.monotonic()
            key_index = self._api_key_index
            last_request_at = self._last_request_at_by_key.get(key_index)
            if last_request_at is not None:
                elapsed = now - last_request_at
                if elapsed < min_interval_s:
                    sleep_s = min_interval_s - elapsed
                    log.info(
                        "llm_backend.throttle",
                        model=self.model,
                        key_index=key_index + 1,
                        total_keys=len(self._api_keys),
                        requests_per_minute=self.requests_per_minute,
                        sleep_s=round(sleep_s, 3),
                    )
                    await asyncio.sleep(sleep_s)
            self._last_request_at_by_key[key_index] = time.monotonic()

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        msg = repr(exc).lower()
        retryable_markers = (
            "429",
            "too many requests",
            "ratelimit",
            "rate limit",
            "readtimeout",
            "timeout",
            "502",
            "503",
            "504",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "validation errors for llmanvilresponse",
            "field required",
        )
        return any(marker in msg for marker in retryable_markers)

    def _is_fallback_key_candidate(self, exc: Exception) -> bool:
        if len(self._api_keys) <= 1:
            return False
        msg = repr(exc).lower()
        fallback_markers = (
            "401",
            "unauthorized",
            "invalid api",
            "invalid_api_key",
            "api key",
            "429",
            "too many requests",
            "ratelimit",
            "rate limit",
            "readtimeout",
            "timeout",
        )
        return any(marker in msg for marker in fallback_markers)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        calculation_steps: list[CalculationStep],
        refusal_reason: str | None = None,
    ) -> AnvilResponse:  # pragma: no cover - network dependent
        if refusal_reason is not None:
            return AnvilResponse(
                query=query,
                answer="Refused.",
                confidence=ResponseConfidence.INSUFFICIENT,
                refusal_reason=refusal_reason,
                retrieved_context_ids=[c.element_id for c in retrieved_chunks],
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "response_model": LLMAnvilResponse,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        key_attempts = max(1, len(self._api_keys))
        for key_attempt in range(key_attempts):
            await self.throttle()
            client = self._get_client()
            try:
                llm_response: LLMAnvilResponse = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self.timeout_s,
                )
            except TimeoutError as exc:
                message = (
                    "OpenAI-compatible LLM call timed out after "
                    f"{self.timeout_s:.1f}s against {self.base_url} "
                    f"(model={self.model})"
                )
                if key_attempt < key_attempts - 1:
                    self._rotate_api_key(reason="timeout_try_next_key")
                    continue
                raise RetryableGenerationError(message) from exc
            except Exception as exc:
                message = (
                    f"OpenAI-compatible LLM call failed against {self.base_url} "
                    f"(model={self.model}): {exc!r}"
                )
                retry_with_key_pool = self._is_retryable_exception(
                    exc
                ) or self._is_fallback_key_candidate(exc)
                if retry_with_key_pool:
                    if key_attempt < key_attempts - 1:
                        self._rotate_api_key(reason="provider_error_try_next_key")
                        continue
                    raise RetryableGenerationError(message) from exc
                raise GenerationError(message) from exc
            self._advance_api_key_for_next_request()
            return AnvilResponse(
                **llm_response.model_dump(),
                calculation_steps=calculation_steps,
            )
        raise RetryableGenerationError(
            "OpenAI-compatible LLM call exhausted all configured API keys "
            f"against {self.base_url} (model={self.model})"
        )


def _nvidia_nim_backend() -> OpenAICompatibleBackend:
    """Construct an NVIDIA NIM backend from `NVIDIA_API_KEY` + `ANVIL_LLM_MODEL`.

    The default model tracks `nim_health.NIM_MODELS` (the locked catalog)
    so we never hardcode a model id that drifts away from the test
    catalog and out of sync with `anvil nim-check` output.
    """
    # Imported lazily to avoid a circular import: nim_health imports
    # logging from this layer.
    from anvil.generation.nim_health import NIM_MODELS

    default_model = next(iter(NIM_MODELS))
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise GenerationError(
            "ANVIL_LLM_BACKEND=nvidia_nim requires NVIDIA_API_KEY to be set. "
            "Get one at https://build.nvidia.com. Refusing to proceed."
        )
    api_keys = _nvidia_api_keys_from_env(api_key)
    model = os.environ.get("ANVIL_LLM_MODEL", default_model)
    # NIM's reasoning-capable models accept `chat_template_kwargs.thinking`
    # and `reasoning_effort` via extra_body. Harmless for models that ignore it.
    extra_body: dict[str, Any] = {}
    if os.environ.get("ANVIL_NIM_REASONING", "").lower() in {"1", "true", "high"}:
        extra_body["chat_template_kwargs"] = {
            "thinking": True,
            "reasoning_effort": os.environ.get(
                "ANVIL_NIM_REASONING_EFFORT", "high"
            ),
        }
    # Hosted NIM occasionally holds a request open for minutes under load.
    # Eval runs should treat that as provider noise and retry the example
    # later, not block the whole benchmark indefinitely. Operators can still
    # raise this for slow models with ANVIL_LLM_TIMEOUT_S.
    timeout_s = float(os.environ.get("ANVIL_LLM_TIMEOUT_S", "45"))
    max_tokens = int(os.environ.get("ANVIL_LLM_MAX_TOKENS", "4096"))
    return OpenAICompatibleBackend(
        base_url=os.environ.get(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        api_key=api_key,
        api_keys=api_keys,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        extra_body=extra_body or None,
    )


def _openai_compatible_backend() -> OpenAICompatibleBackend:
    """Construct a generic OpenAI-compatible backend from env vars.

    Required env vars:
      * `OPENAI_COMPAT_BASE_URL` — e.g. `https://api.together.xyz/v1`
      * `OPENAI_COMPAT_API_KEY`
      * `ANVIL_LLM_MODEL`
    """
    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL")
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY")
    model = os.environ.get("ANVIL_LLM_MODEL")
    missing = [
        name
        for name, val in [
            ("OPENAI_COMPAT_BASE_URL", base_url),
            ("OPENAI_COMPAT_API_KEY", api_key),
            ("ANVIL_LLM_MODEL", model),
        ]
        if not val
    ]
    if missing:
        raise GenerationError(
            f"ANVIL_LLM_BACKEND=openai_compatible requires: {', '.join(missing)}"
        )
    assert base_url and api_key and model  # for the type-checker
    timeout_s = float(os.environ.get("ANVIL_LLM_TIMEOUT_S", "120"))
    max_tokens = int(os.environ.get("ANVIL_LLM_MAX_TOKENS", "4096"))
    return OpenAICompatibleBackend(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )


def _split_api_key_env(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _dedupe_api_keys(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for key in keys:
        stripped = key.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped


def _nvidia_api_keys_from_env(primary: str) -> list[str]:
    """Read primary + fallback NIM keys without exposing them to logs.

    Supported env vars:
      * `NVIDIA_API_KEY` — primary key, still required.
      * `NVIDIA_API_KEY_FALLBACKS` — comma/newline separated fallback keys.
      * `NVIDIA_API_KEYS` — optional full comma/newline separated pool.
    """
    return _dedupe_api_keys(
        [primary]
        + _split_api_key_env(os.environ.get("NVIDIA_API_KEY_FALLBACKS"))
        + _split_api_key_env(os.environ.get("NVIDIA_API_KEYS"))
    )


def get_default_backend() -> LLMBackend:
    """Return the backend selected by `ANVIL_LLM_BACKEND` env var.

    Recognized values (case-insensitive):

    | Value                  | Backend                                |
    | :--------------------- | :------------------------------------- |
    | `fake` (default)       | `FakeLLMBackend` — **logs a WARNING**  |
    | `nvidia_nim`           | `OpenAICompatibleBackend` → NIM        |
    | `openai_compatible`    | `OpenAICompatibleBackend` → generic    |
    | `instructor`           | `InstructorBackend` (any provider)     |

    The fake backend is permitted for CI / demos but logs a loud WARNING on
    every selection so a prod deploy that forgot to set env vars sees it
    immediately rather than shipping silent fake responses.
    """
    choice = os.environ.get("ANVIL_LLM_BACKEND", "fake").lower().strip()
    if choice == "nvidia_nim":  # pragma: no cover - network dependent
        backend = _nvidia_nim_backend()
        log.info("llm_backend.selected", backend="nvidia_nim", model=backend.model)
        return backend
    if choice == "openai_compatible":  # pragma: no cover - network dependent
        backend = _openai_compatible_backend()
        log.info(
            "llm_backend.selected",
            backend="openai_compatible",
            base_url=backend.base_url,
            model=backend.model,
        )
        return backend
    if choice == "instructor":  # pragma: no cover - network dependent
        model = os.environ.get("ANVIL_LLM_MODEL")
        if not model:
            raise GenerationError(
                "ANVIL_LLM_BACKEND=instructor requires ANVIL_LLM_MODEL "
                "(e.g. 'openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet')."
            )
        log.info("llm_backend.selected", backend="instructor", model=model)
        return InstructorBackend(model=model)
    if choice != "fake":
        raise GenerationError(
            f"Unknown ANVIL_LLM_BACKEND={choice!r}. Supported values: "
            f"fake, nvidia_nim, openai_compatible, instructor."
        )
    log.warning(
        "llm_backend.fake_selected",
        hint=(
            "FakeLLMBackend is in use — this is CI/test mode. For real "
            "model output, set ANVIL_LLM_BACKEND=nvidia_nim (+NVIDIA_API_KEY), "
            "openai_compatible, or instructor."
        ),
    )
    return FakeLLMBackend()
