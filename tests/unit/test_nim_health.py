"""M1 — NIM health-check tests.

`anvil nim-check` is the CLI a reviewer runs first, and the only piece
of code that talks to the real NIM endpoint. These tests stub the
endpoint with `respx` so:

  * CI runs without network access and without a NIM API key.
  * The CLI's contract — exit code, JSON shape, fallback messages — is
    locked in by behavioral assertions.
  * Failure modes (missing key, HTTP 401, HTTP 429, transport timeout)
    each have a dedicated test that pins the structured-error story.

Every test added here is also recorded in `docs/test_inventory.md`
(per the plan's logging & documentation conventions) so a reviewer can
trace each test to the milestone it anchors.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from anvil.cli import main as cli_main
from anvil.generation.nim_health import (
    DEFAULT_NIM_BASE_URL,
    NIM_MODELS,
    NIMHealthCheck,
    check_all_nim_models,
    check_nim_health,
    list_nim_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Inject a sentinel NVIDIA_API_KEY for the duration of one test."""
    key = "nvapi-test-1234567890abcdef"
    monkeypatch.setenv("NVIDIA_API_KEY", key)
    return key


def _ok_response(model: str, content: str = "OK") -> httpx.Response:
    """Mimic the shape NIM returns for a successful chat completion."""
    return httpx.Response(
        200,
        json={
            "id": "cmpl-test",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "total_tokens": 13,
            },
        },
        headers={"x-request-id": "test-req-001"},
    )


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


def test_nim_catalog_has_three_locked_models() -> None:
    """The plan locks 3 models at §0. Adding a 4th must be a deliberate
    decision (ADR), not a silent edit. Encode that contract."""
    assert len(NIM_MODELS) == 3
    expected = {
        "meta/llama-3.3-70b-instruct",
        "deepseek-ai/deepseek-v3.1",
        "nvidia/llama-3.1-nemotron-70b-instruct",
    }
    assert set(NIM_MODELS) == expected
    # Every entry must carry the metadata the CLI / health check rely on.
    for model_id, meta in NIM_MODELS.items():
        assert isinstance(meta["label"], str) and meta["label"], model_id
        assert isinstance(meta["purpose"], str) and meta["purpose"], model_id
        assert isinstance(meta["supports_reasoning"], bool), model_id


# ---------------------------------------------------------------------------
# check_nim_health — single-model probe
# ---------------------------------------------------------------------------


@respx.mock
async def test_check_nim_health_ok_path(fake_key: str) -> None:
    """Happy path: 200 OK with a valid usage block. `reachable=True`,
    `latency_ms` set, `error is None`."""
    model = "meta/llama-3.3-70b-instruct"
    route = respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        return_value=_ok_response(model)
    )
    result = await check_nim_health(model)
    assert isinstance(result, NIMHealthCheck)
    assert result.reachable is True
    assert result.error is None
    assert result.latency_ms is not None
    assert result.sample_response is not None and "OK" in result.sample_response
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 1
    assert result.request_id == "test-req-001"
    # The probe must include the model and the auth header.
    assert route.called
    sent = route.calls[0].request
    body = json.loads(sent.content)
    assert body["model"] == model
    assert body["max_tokens"] == 8
    assert sent.headers["authorization"] == f"Bearer {fake_key}"


async def test_check_nim_health_missing_key_returns_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key → `reachable=False`, helpful error pointing at build.nvidia.com.
    MUST NOT raise — the CLI iterates over models and one missing key
    must not abort the run."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    result = await check_nim_health("meta/llama-3.3-70b-instruct")
    assert result.reachable is False
    assert result.error is not None
    assert "NVIDIA_API_KEY" in result.error
    assert "build.nvidia.com" in result.error
    assert result.latency_ms is None  # no network call happened


@respx.mock
async def test_check_nim_health_http_401_redacts_api_key(
    fake_key: str,
) -> None:
    """A 401 with the body echoing the key (some providers do this) MUST
    NOT leak the key into `error`. Redaction matches the manifest format
    used by the run-logger (M2)."""
    model = "meta/llama-3.3-70b-instruct"
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            text=f"invalid api_key={fake_key}",
        )
    )
    result = await check_nim_health(model)
    assert result.reachable is False
    assert result.error is not None
    assert fake_key not in result.error, "raw API key leaked into error message"
    assert "<redacted:" in result.error
    assert "401" in result.error


@respx.mock
async def test_check_nim_health_http_429_is_not_an_exception(
    fake_key: str,
) -> None:
    """Free-tier rate-limit responses must surface as a structured error,
    not a raised exception. The CLI exits 1 if NO model is reachable —
    a single 429 must allow the next model to be tried."""
    model = "deepseek-ai/deepseek-v3.1"
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(429, text="quota exceeded")
    )
    result = await check_nim_health(model)
    assert result.reachable is False
    assert result.error is not None
    assert "429" in result.error


@respx.mock
async def test_check_nim_health_transport_error_caught(
    fake_key: str,
) -> None:
    """Network-level errors (DNS / TLS / timeout) must be caught and
    surfaced via `error`, never raised."""
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    result = await check_nim_health("nvidia/llama-3.1-nemotron-70b-instruct")
    assert result.reachable is False
    assert result.error is not None
    assert "ConnectError" in result.error or "connection refused" in result.error


# ---------------------------------------------------------------------------
# check_all_nim_models — every model probed
# ---------------------------------------------------------------------------


@respx.mock
async def test_check_all_nim_models_probes_every_model_once(
    fake_key: str,
) -> None:
    """`anvil nim-check` must hit each model in NIM_MODELS exactly once,
    in catalog order, and continue past failures."""
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["model"])
        return _ok_response(body["model"])

    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        side_effect=_handler
    )
    results = await check_all_nim_models()
    assert len(results) == len(NIM_MODELS)
    assert seen == list(NIM_MODELS.keys())
    assert all(r.reachable for r in results)


@respx.mock
async def test_check_all_nim_models_partial_failure_continues(
    fake_key: str,
) -> None:
    """If model #1 returns 500, models #2 and #3 must still be probed —
    partial-failure resilience is the whole point of the structured
    error story."""
    call_count = {"n": 0}

    def _flaky(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(500, text="upstream error")
        body = json.loads(request.content)
        return _ok_response(body["model"])

    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        side_effect=_flaky
    )
    results = await check_all_nim_models()
    assert len(results) == 3
    assert results[0].reachable is False
    assert results[1].reachable is True
    assert results[2].reachable is True


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


@respx.mock
def test_cli_nim_check_json_mode_returns_zero_on_any_reachable(
    fake_key: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`anvil nim-check --json` exits 0 when at least one model is OK
    and emits a JSON array on stdout consumable by `jq` / CI artifacts."""
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        side_effect=lambda request: _ok_response(json.loads(request.content)["model"])
    )
    rc = cli_main(["nim-check", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, dict) and "results" in parsed
    assert len(parsed["results"]) == 3
    assert all(item["reachable"] for item in parsed["results"])
    # Without --list, no catalog_drift block is emitted.
    assert "catalog_drift" not in parsed


@respx.mock
def test_cli_nim_check_json_mode_returns_one_on_total_failure(
    fake_key: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All models down → exit code 1. CI needs this to fail fast."""
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(503, text="service unavailable")
    )
    rc = cli_main(["nim-check", "--json"])
    assert rc == 1
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert all(item["reachable"] is False for item in parsed["results"])


# ---------------------------------------------------------------------------
# list_nim_catalog + --list catalog drift
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_nim_catalog_returns_model_ids(fake_key: str) -> None:
    """Happy path: `/v1/models` returns OpenAI-shape `data` list."""
    respx.get(f"{DEFAULT_NIM_BASE_URL}/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "meta/llama-3.3-70b-instruct", "object": "model"},
                    {"id": "deepseek-ai/deepseek-v3.1", "object": "model"},
                    {"id": "some/new-model", "object": "model"},
                ],
            },
        )
    )
    ids = await list_nim_catalog()
    assert "meta/llama-3.3-70b-instruct" in ids
    assert "some/new-model" in ids
    assert len(ids) == 3


async def test_list_nim_catalog_returns_empty_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key → empty list, never raises."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    ids = await list_nim_catalog()
    assert ids == []


@respx.mock
async def test_list_nim_catalog_returns_empty_on_http_error(fake_key: str) -> None:
    respx.get(f"{DEFAULT_NIM_BASE_URL}/models").mock(
        return_value=httpx.Response(500, text="upstream error")
    )
    assert await list_nim_catalog() == []


@respx.mock
def test_cli_nim_check_list_reports_drift(
    fake_key: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--list` adds a `catalog_drift` block surfacing locked-vs-live
    differences. Useful when a NIM model rotates out of the catalog."""
    respx.post(f"{DEFAULT_NIM_BASE_URL}/chat/completions").mock(
        side_effect=lambda request: _ok_response(json.loads(request.content)["model"])
    )
    # The "live" catalog is missing one of our locked models and includes a
    # new one — both the missing-from-live and new-in-live lists must
    # surface the drift.
    respx.get(f"{DEFAULT_NIM_BASE_URL}/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "meta/llama-3.3-70b-instruct"},
                    {"id": "deepseek-ai/deepseek-v3.1"},
                    {"id": "minimaxai/minimax-m2.7"},
                ],
            },
        )
    )
    rc = cli_main(["nim-check", "--json", "--list"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    drift = parsed["catalog_drift"]
    assert drift["live_count"] == 3
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in drift["missing_from_live"]
    assert "minimaxai/minimax-m2.7" in drift["new_in_live"]


def test_cli_nim_check_no_key_exits_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without NVIDIA_API_KEY the CLI must exit 1 and print a message
    that names the env var AND the URL to obtain a key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    rc = cli_main(["nim-check", "--json"])
    assert rc == 1
    out = capsys.readouterr()
    parsed = json.loads(out.out)
    assert all(not item["reachable"] for item in parsed["results"])
    assert all(
        "NVIDIA_API_KEY" in (item["error"] or "") for item in parsed["results"]
    )


