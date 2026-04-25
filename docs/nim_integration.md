# NVIDIA NIM — Integration Guide

NIM (NVIDIA Inference Microservices) is the OpenAI-compatible hosted
inference catalog at [build.nvidia.com](https://build.nvidia.com).
Conceptually it sits in the same niche as OpenRouter or Together — one
key, one base URL, dozens of models — but the optimization layer
underneath is NVIDIA's TensorRT-LLM / vLLM / SGLang containers running
on real H200/B200 hardware. ANVIL targets NIM as the default real
backend because the free tier is generous and the schema is the boring
OpenAI Chat Completions one.

## Endpoint at a glance

| | |
| :--- | :--- |
| **Base URL** | `https://integrate.api.nvidia.com/v1` |
| **Auth** | `Authorization: Bearer <NVIDIA_API_KEY>` (key prefix `nvapi-`) |
| **Schema** | OpenAI Chat Completions — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`. |
| **Free tier** | 40 req/min RPM, ~1k credits on signup, request up to 5k. |
| **Model id format** | `<provider>/<model-name>`, e.g. `meta/llama-3.3-70b-instruct`. |
| **Get a key** | [build.nvidia.com](https://build.nvidia.com) → join the NVIDIA Developer Program → "Get API Key". |

## Locked model catalog

ANVIL ships with a 3-model locked catalog in
`src/anvil/generation/nim_health.py::NIM_MODELS`. Adding a 4th is a
deliberate edit (a test in `tests/unit/test_nim_health.py` enforces
the count). The three are deliberately picked from different
design-space corners:

| ID | Why we picked it | Reasoning mode |
| :--- | :--- | :---: |
| `meta/llama-3.3-70b-instruct` | Meta-family general instruction-follower; eval baseline. Default for `_nvidia_nim_backend()`. Live-probe latency ~0.3 s. | — |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | NVIDIA-RLHF tuned with thinking-mode; stress-tests our `chat_template_kwargs.thinking` passthrough. Live-probe latency ~0.6 s. | ✓ |
| `openai/gpt-oss-120b` | OpenAI's open-weight 120b on NIM; ensures the table compares across **three different model families** (Meta / NVIDIA / OpenAI), not three flavors of one. Live-probe latency ~0.3 s. | — |

The catalog rotates over time. If a model is deprecated by NIM, the
locked test still passes (the catalog is an in-repo dict) but
`anvil nim-check` will flag it as `reachable=False`. Use
`anvil nim-check --list` to compare the locked catalog to the live
`/v1/models` response and propose a swap.

## Drop-in OpenAI client

NIM works with any OpenAI-compatible client. Two examples that come up
frequently:

### Python

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
resp = await client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Reply with the single word: OK"}],
    temperature=0.0,
    max_tokens=8,
)
print(resp.choices[0].message.content)
```

### `instructor` (structured-output mode)

ANVIL's `OpenAICompatibleBackend` uses this exact pattern. JSON mode
works across NIM, Together, Fireworks, vLLM, and Ollama; tool-use mode
is provider-specific.

```python
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

raw = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
client = instructor.from_openai(raw, mode=instructor.Mode.JSON)

class Answer(BaseModel):
    text: str
    confidence: float

resp = await client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    response_model=Answer,
    messages=[{"role": "user", "content": "..."}],
)
```

## Reasoning-mode passthrough

Some NIM models accept a `chat_template_kwargs.thinking` flag in
`extra_body` to enable chain-of-thought generation. ANVIL exposes this
via two env vars:

```bash
export ANVIL_NIM_REASONING=1                  # enables the passthrough
export ANVIL_NIM_REASONING_EFFORT=high        # 'high' | 'medium' | 'low'
```

When set, `_nvidia_nim_backend()` adds:

```python
extra_body={
    "chat_template_kwargs": {
        "thinking": True,
        "reasoning_effort": "high",
    }
}
```

This is harmless on models that don't honor the kwarg (Llama 3.3, the
default, ignores it). Useful primarily for the DeepSeek catalog row.

## ANVIL's NIM-specific code surface

| Surface | File | Purpose |
| :--- | :--- | :--- |
| `NIM_MODELS` | `src/anvil/generation/nim_health.py` | Locked catalog. |
| `check_nim_health(model)` | `src/anvil/generation/nim_health.py` | Probe a single model; never raises; redacts the API key on failure. |
| `check_all_nim_models()` | `src/anvil/generation/nim_health.py` | Sequential probe over the catalog (free tier doesn't tolerate parallel probes). |
| `list_nim_catalog()` | `src/anvil/generation/nim_health.py` | Live `/v1/models` fetch — used for drift detection. |
| `_nvidia_nim_backend()` | `src/anvil/generation/llm_backend.py` | Constructs an `OpenAICompatibleBackend` from env. |
| `anvil nim-check` | `src/anvil/cli.py` | CLI command; supports `--json`, `--list`, `--api-key`, `--base-url`, `--timeout`. |

## Operational notes

* **Sequential probing.** The free tier 429s under parallel requests
  from the same key. `check_all_nim_models` is sequential by design;
  `pytest -n 8` is fine for our suite because the suite itself uses
  `respx` and never touches the network.
* **Cassette-friendly.** Every NIM call goes through `httpx`, so
  `pytest-respx` can stub the entire suite without code changes —
  this is what CI does.
* **Quota planning.** A full ANVIL ablation matrix is 7 ablations × 30
  examples = 210 requests per backend. Three NIM models = 630
  requests. At 40 RPM that's a 16-minute lower bound, well within the
  free-tier daily credits but not parallelizable.
* **Self-hosted NIM.** `NVIDIA_NIM_BASE_URL` accepts any URL; point it
  at a self-hosted NIM container (e.g. `http://nim-llama-3.3:8000/v1`)
  and the rest of ANVIL works identically.
* **Failure mode policy.** `check_nim_health` and `list_nim_catalog`
  both **fail soft** (return structured errors / empty list), but
  `_nvidia_nim_backend()` raises immediately if the API key is missing
  — we refuse to silently fall through to FakeLLMBackend in production.

## Versus OpenRouter / Together / generic OpenAI-compatible

| | NVIDIA NIM | OpenRouter | Together / Fireworks |
| :--- | :--- | :--- | :--- |
| Free tier credits | 1k–5k tokens-equivalent | $1 trial | $1–5 trial |
| Free tier RPM | 40 | varies (~20) | varies (~60) |
| Model namespace | `<provider>/<model>` | `<provider>/<model>` | `<vendor>/<model>` |
| Tool-use support | partial | per-model | per-model |
| ANVIL backend | `nvidia_nim` | `openai_compatible` (set base URL to OpenRouter's) | `openai_compatible` |
| Self-host option | ✓ (NIM containers) | ✗ | partial |

ANVIL treats NIM as the default because (a) the free tier is enough to
populate the headline-results table without spending money, (b) the
schema is the boring OpenAI one so swapping providers is a one-line
edit, and (c) the JD targets a domain (regulated industrial AI) that
favors NVIDIA's hardware story.

## Testing without a key

CI runs without `NVIDIA_API_KEY`. The `respx` cassettes in
`tests/unit/test_nim_health.py` stub every NIM endpoint, so the full
test suite + the `anvil nim-check` smoke test are exercisable on a
machine that has never touched build.nvidia.com.

To exercise the real path locally:

```bash
export NVIDIA_API_KEY=nvapi-...
uv run anvil nim-check                  # 3-row table, all reachable
uv run anvil nim-check --json --list    # adds catalog_drift block
uv run python scripts/run_nim_headlines.py --include-fake
```
