"""NVIDIA NIM connection check.

`anvil nim-check` is the first thing a reviewer runs after pulling the
repo. Its job is to give a deterministic, structured answer to the
question *"is the NIM endpoint reachable, and which of the three models
in our locked catalog actually respond right now?"* — without depending
on any of the rest of the pipeline.

Design choices:

* `NIMHealthCheck` is a Pydantic model so the result can be JSON-serialized
  (the CLI prints a Rich table; CI uploads the JSON as an artifact).
* `check_nim_health` **never raises** for transport-level errors. It
  catches every `httpx` / `openai` exception and lands the message in
  `error`. This is deliberate: the CLI iterates over models, and one
  model being down (e.g. catalog rotation) must not abort the run.
* The probe is tiny — a 4-token request that asks the model to reply
  ``"OK"``. We don't validate the answer's content, only that *some*
  text came back. NIM models occasionally refuse small probes, so the
  acceptance criterion is `latency_ms is not None and error is None`.
* Per the locked plan, the catalog is fixed at 3 models. Adding a fourth
  is a deliberate decision (ADR), not a flag.
"""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import BaseModel, Field

from anvil.logging_config import get_logger

log = get_logger(__name__)


# Locked NIM model catalog — see plan §0 (Decisions locked from the
# clarifying round). Each entry covers a different design-space corner so
# the eval table reads as a real comparison rather than three flavors of
# the same family.
NIM_MODELS: dict[str, dict[str, Any]] = {
    "meta/llama-3.3-70b-instruct": {
        "label": "llama-3.3-70b",
        "purpose": "strong general instruction-follower (eval baseline)",
        "supports_reasoning": False,
    },
    "deepseek-ai/deepseek-v3.1": {
        "label": "deepseek-v3.1",
        "purpose": "reasoning-mode stress test (chat_template_kwargs.thinking)",
        "supports_reasoning": True,
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "label": "nemotron-70b",
        "purpose": "NVIDIA-tuned instruction-follower (NIM-native baseline)",
        "supports_reasoning": False,
    },
}

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
PROBE_PROMPT = "Reply with the single word: OK"
PROBE_TIMEOUT_S = 10.0


class NIMHealthCheck(BaseModel):
    """One probe result, JSON-serializable for run logs and CI artifacts."""

    model: str
    label: str
    purpose: str
    base_url: str
    reachable: bool = False
    latency_ms: float | None = None
    sample_response: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    request_id: str | None = Field(
        default=None,
        description="NIM x-request-id header — useful for support tickets.",
    )


def _redact_key(api_key: str | None) -> str:
    """Match the redaction format used by the run-log manifests (M2)."""
    if not api_key:
        return "<unset>"
    import hashlib

    return f"<redacted:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}>"


async def check_nim_health(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> NIMHealthCheck:
    """Probe a single NIM model. Never raises for transport-level errors."""
    if model not in NIM_MODELS:
        log.warning(
            "nim_health.unknown_model_in_catalog",
            model=model,
            known=list(NIM_MODELS),
        )
    meta = NIM_MODELS.get(model, {"label": model, "purpose": "(not in locked catalog)"})
    url = base_url or os.environ.get("NVIDIA_NIM_BASE_URL", DEFAULT_NIM_BASE_URL)
    key = api_key or os.environ.get("NVIDIA_API_KEY")

    result = NIMHealthCheck(
        model=model,
        label=meta["label"],
        purpose=meta["purpose"],
        base_url=url,
    )

    if not key:
        result.error = (
            "NVIDIA_API_KEY is unset. Get a free key at https://build.nvidia.com "
            "and export it before running `anvil nim-check`."
        )
        log.warning("nim_health.missing_key", model=model)
        return result

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover — httpx is a hard dep
        result.error = f"httpx import failed: {exc!r}"
        return result

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        result.latency_ms = round(latency_ms, 1)
        result.request_id = resp.headers.get("x-request-id") or resp.headers.get(
            "nvcf-reqid"
        )
        if resp.status_code != 200:
            result.error = (
                f"HTTP {resp.status_code}: {resp.text[:200]!r}".replace(
                    key, _redact_key(key)
                )
            )
            log.warning(
                "nim_health.http_error",
                model=model,
                status=resp.status_code,
                latency_ms=result.latency_ms,
            )
            return result
        body = resp.json()
        choices = body.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
            result.sample_response = (content or "")[:120]
        usage = body.get("usage") or {}
        result.prompt_tokens = usage.get("prompt_tokens")
        result.completion_tokens = usage.get("completion_tokens")
        result.reachable = True
        log.info(
            "nim_health.ok",
            model=model,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        return result
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        result.latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
        # Never let the api_key leak via an exception repr.
        msg = repr(exc).replace(key, _redact_key(key))
        result.error = msg
        log.warning("nim_health.exception", model=model, error=msg)
        return result


async def check_all_nim_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> list[NIMHealthCheck]:
    """Probe every model in `NIM_MODELS`, sequentially.

    Sequential (not concurrent) on purpose: NIM's free tier rate-limits
    aggressively (~40 req/min), and parallel probes from the same key
    occasionally return 429s that look like real outages. Sequential
    probing is slow but unambiguous.
    """
    results: list[NIMHealthCheck] = []
    for model in NIM_MODELS:
        results.append(
            await check_nim_health(
                model, api_key=api_key, base_url=base_url, timeout=timeout
            )
        )
    return results


async def list_nim_catalog(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> list[str]:
    """Return the live list of model ids exposed by the NIM endpoint.

    Calls the OpenAI-compatible `/v1/models` endpoint. Useful when a
    locked model in `NIM_MODELS` rotates out of the catalog — comparing
    `set(NIM_MODELS)` against this live list is the fastest way to
    detect catalog drift.

    Returns `[]` on any error (no key, transport failure, non-200) —
    callers can distinguish "unset key" from "real outage" via
    `check_nim_health` instead.
    """
    url = base_url or os.environ.get("NVIDIA_NIM_BASE_URL", DEFAULT_NIM_BASE_URL)
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        return []
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return []
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url.rstrip('/')}/models", headers=headers)
        if resp.status_code != 200:
            log.warning(
                "nim_health.list_models_http_error",
                status=resp.status_code,
            )
            return []
        body = resp.json()
        data = body.get("data") or []
        return [item.get("id") for item in data if item.get("id")]
    except Exception as exc:  # noqa: BLE001
        log.warning("nim_health.list_models_exception", error=repr(exc))
        return []
