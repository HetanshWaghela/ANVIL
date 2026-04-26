"""Agent decider — picks the next tool call given the running transcript.

Two implementations:

* `ScriptedAgentBackend` — deterministic, takes a pre-baked sequence
  of (ToolCall | FinalAnswer). Used in unit tests so we can assert on
  loop behavior (budget enforcement, error tolerance, transcript
  shape) without hitting any LLM.

* `LLMAgentBackend` — wraps an `LLMBackend` and uses instructor's
  structured-output mode to choose one of `{ToolCall, FinalAnswer}`
  per turn. Lazily imports `instructor` so unit tests don't need it.

Both share the `AgentBackend` Protocol so the agent core (`agent.py`)
is decoupled from the decider implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from anvil import RetryableGenerationError
from anvil.logging_config import get_logger
from anvil.schemas.agent import AgentStep, FinalAnswer, ToolCall

if TYPE_CHECKING:
    from anvil.generation.llm_backend import LLMBackend

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Decision union
# ---------------------------------------------------------------------------


class AgentDecision(BaseModel):
    """One turn's decision: either pick a tool, or finalize.

    Modeled as an exclusive-or so the LLM (via instructor) is forced
    to commit. The validator below enforces the XOR invariant — a
    decision with both fields set is structurally invalid and would
    indicate an instructor bug.
    """

    tool_call: ToolCall | None = Field(
        default=None,
        description=(
            "Next tool to invoke. Set this OR `final` (never both, "
            "never neither)."
        ),
    )
    final: FinalAnswer | None = Field(
        default=None,
        description=(
            "Terminate the loop with this AnvilResponse. Set this OR "
            "`tool_call` (never both, never neither)."
        ),
    )

    def model_post_init(self, _ctx: Any) -> None:  # pragma: no cover - trivial
        if (self.tool_call is None) == (self.final is None):
            raise ValueError(
                "AgentDecision must set EXACTLY ONE of "
                "`tool_call` or `final`."
            )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentBackend(Protocol):
    """Pluggable decider for the agent loop.

    Implementations receive (a) the user query, (b) the running
    transcript so far, (c) the available tool manifest, and must
    return an `AgentDecision`. They must NOT mutate any of their
    inputs.
    """

    async def decide(
        self,
        query: str,
        history: list[AgentStep],
        tool_manifest: dict[str, dict[str, Any]],
    ) -> AgentDecision: ...


# ---------------------------------------------------------------------------
# Scripted backend (deterministic, for tests)
# ---------------------------------------------------------------------------


class ScriptedAgentBackend:
    """Returns decisions from a pre-baked list, in order.

    Pass a list of `ToolCall | FinalAnswer | AgentDecision` — anything
    else raises at construction time. Each `decide()` call pops the
    head of the queue and wraps it in an `AgentDecision` if needed.

    If the script is exhausted, `decide()` raises `IndexError` —
    making test failures loud rather than silently looping forever.
    """

    def __init__(
        self, script: list[ToolCall | FinalAnswer | AgentDecision]
    ) -> None:
        self._script: list[AgentDecision] = []
        for item in script:
            if isinstance(item, AgentDecision):
                self._script.append(item)
            elif isinstance(item, ToolCall):
                self._script.append(AgentDecision(tool_call=item))
            elif isinstance(item, FinalAnswer):
                self._script.append(AgentDecision(final=item))
            else:
                raise TypeError(
                    f"ScriptedAgentBackend script item must be ToolCall, "
                    f"FinalAnswer, or AgentDecision; got {type(item).__name__}."
                )
        self._cursor = 0
        self.calls: list[tuple[str, int]] = []  # (query_preview, step_index) — for assertions

    async def decide(
        self,
        query: str,
        history: list[AgentStep],
        tool_manifest: dict[str, dict[str, Any]],
    ) -> AgentDecision:
        if self._cursor >= len(self._script):
            raise IndexError(
                "ScriptedAgentBackend script exhausted — the test "
                "produced more turns than expected. Either add more "
                "decisions to the script or check the loop's "
                "termination logic."
            )
        decision = self._script[self._cursor]
        self.calls.append((query[:40], self._cursor))
        self._cursor += 1
        return decision


# ---------------------------------------------------------------------------
# LLM-backed backend (real model, via instructor)
# ---------------------------------------------------------------------------


_AGENT_SYSTEM_PROMPT = (
    "You are a tool-using compliance-grade RAG agent for the ASME-like "
    "engineering standard SPES-1. Each turn, decide EXACTLY ONE of:\n"
    "  - call a tool: set `tool_call` with `name` + `arguments`, leave "
    "`final` null.\n"
    "  - finalize an answer: set `final` with a complete AnvilResponse, "
    "leave `tool_call` null.\n"
    "Setting both, or neither, is a schema error.\n"
    "\n"
    "Rules you MUST follow:\n"
    "  1. NEVER guess material properties, allowable stresses, or joint "
    "efficiencies — call `pinned_lookup` to get them. A successful "
    "`pinned_lookup` for a direct table question is auto-finalized by "
    "the host; you do NOT need to call `pinned_lookup` again.\n"
    "  2. For thickness / MAWP / design-pressure questions, call "
    "`calculate` with all required inputs; never compute them yourself. "
    "A successful `calculate` is auto-finalized by the host.\n"
    "  3. For 'where is X defined' / 'which paragraph governs Y' / "
    "cross-reference questions, call `retrieve_context` ONCE with a "
    "specific natural-language query, then finalize with citations to "
    "the retrieved chunks. Use `graph_lookup` only when you need a "
    "1-hop neighborhood from a known paragraph_ref.\n"
    "  4. After retrieving evidence, FINALIZE on the very next turn. "
    "Repeating retrieval with paraphrased queries is wasted budget and "
    "will be auto-terminated by the host.\n"
    "  5. Every claim in the final answer must cite a paragraph_ref "
    "with a verbatim quote — citations come from `retrieve_context` or "
    "`graph_lookup` results.\n"
    "  6. If retrieval scores are very low or required pinned lookups "
    "fail, finalize with a refusal (confidence=insufficient + "
    "refusal_reason). Do NOT fabricate.\n"
    "  7. Be efficient — you have a hard step budget. Most queries need "
    "1–2 tool calls plus a finalize; calculations need a single "
    "`calculate` call (the host wraps it).\n"
)


class LLMAgentBackend:
    """LLM-driven decider using `instructor` structured outputs.

    Lazy import of instructor: pulling it in at module load would
    force every test that imports `agent_backend` to install
    `instructor`, which we don't want.
    """

    def __init__(
        self,
        backend: LLMBackend,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.system_prompt = system_prompt or _AGENT_SYSTEM_PROMPT

    async def decide(
        self,
        query: str,
        history: list[AgentStep],
        tool_manifest: dict[str, dict[str, Any]],
    ) -> AgentDecision:  # pragma: no cover - network dependent
        # We need a structured `AgentDecision` from the LLM, NOT an
        # `AnvilResponse`. So we bypass `backend.generate()` (which
        # is hard-coded to the AnvilResponse schema) and reach into
        # the underlying instructor client.
        #
        # Two client shapes exist in the codebase:
        #   - `InstructorBackend`'s client uses `client.create(...)`
        #     (instructor.from_provider-style)
        #   - `OpenAICompatibleBackend`'s client uses
        #     `client.chat.completions.create(...)` (NIM, vLLM,
        #     Together) — this is what M5 real-NIM runs use.
        # We dispatch on attribute presence rather than isinstance to
        # keep this file independent of those concrete classes.
        model = self.model or self.backend.model  # type: ignore[attr-defined]
        msgs = self._build_messages(query, history, tool_manifest)
        throttle = getattr(self.backend, "throttle", None)
        timeout_s = getattr(self.backend, "timeout_s", 45.0)
        max_retries = getattr(self.backend, "max_retries", 1)
        key_attempts = max(1, len(getattr(self.backend, "_api_keys", [None])))
        rotate_key = getattr(self.backend, "_rotate_api_key", None)
        is_retryable = getattr(self.backend, "_is_retryable_exception", None)
        is_fallback_candidate = getattr(
            self.backend, "_is_fallback_key_candidate", None
        )

        for key_attempt in range(key_attempts):
            if throttle is not None:
                await throttle()
            client = self.backend._get_client()  # type: ignore[attr-defined]
            try:
                if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                    decision: AgentDecision = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=msgs,
                            response_model=AgentDecision,
                            temperature=0.0,
                            max_retries=max_retries,
                        ),
                        timeout=timeout_s,
                    )
                else:
                    decision = await asyncio.wait_for(
                        client.create(
                            response_model=AgentDecision,
                            messages=msgs,
                            temperature=0.0,
                            max_retries=max_retries,
                        ),
                        timeout=timeout_s,
                    )
            except TimeoutError as exc:
                if rotate_key is not None and key_attempt < key_attempts - 1:
                    rotate_key(reason="agent_timeout_try_next_key")
                    continue
                raise RetryableGenerationError(
                    f"Agent decider call timed out after {timeout_s:.1f}s "
                    f"for model={model}"
                ) from exc
            except Exception as exc:
                retryable = bool(is_retryable is not None and is_retryable(exc))
                fallback_candidate = bool(
                    is_fallback_candidate is not None
                    and is_fallback_candidate(exc)
                )
                if retryable or fallback_candidate:
                    if rotate_key is not None and key_attempt < key_attempts - 1:
                        rotate_key(reason="agent_provider_error_try_next_key")
                        continue
                    raise RetryableGenerationError(
                        f"Agent decider call failed for model={model}: {exc!r}"
                    ) from exc
                raise
            advance_key = getattr(self.backend, "_advance_api_key_for_next_request", None)
            if advance_key is not None:
                advance_key()
            return decision
        raise RetryableGenerationError(
            f"Agent decider call exhausted all configured API keys for model={model}"
        )

    def _build_messages(  # pragma: no cover - tested via integration
        self,
        query: str,
        history: list[AgentStep],
        tool_manifest: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        manifest_lines = []
        for name, meta in tool_manifest.items():
            params = ", ".join(meta.get("parameters", {}).keys()) or "—"
            manifest_lines.append(
                f"  - {name}({params}): {meta.get('description', '')}"
            )
        history_lines: list[str] = []
        for step in history:
            args = ", ".join(f"{k}={v!r}" for k, v in step.call.arguments.items())
            history_lines.append(
                f"step {step.step_index}: {step.call.name}({args}) "
                f"→ {('error: ' + step.result.error) if step.result.error else 'ok'}"
            )
        sys = (
            f"{self.system_prompt}\n\n"
            f"Tools available:\n" + "\n".join(manifest_lines)
        )
        user = (
            f"Query: {query}\n\n"
            f"Transcript so far:\n"
            + ("\n".join(history_lines) if history_lines else "  (empty)")
            + "\n\nReturn your next decision."
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]


__all__ = [
    "AgentBackend",
    "AgentDecision",
    "LLMAgentBackend",
    "ScriptedAgentBackend",
]
