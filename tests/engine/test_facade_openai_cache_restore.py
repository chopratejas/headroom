"""Proving test for FIX #1 — OpenAI cache-mode frozen-prefix restore.

The OpenAI golden corpus already had a cache-mode frozen-prefix case
(``openai_chat_cache_mode_frozen_prefix``), but its frozen messages were
small enough that the real compression pipeline never mutated them — so the
*missing* ``_restore_frozen_prefix`` call in the engine's
``_on_request_openai_chat`` went undetected by the corpus.

``_restore_frozen_prefix`` is a **safety net**: it forces the frozen prefix
back to the exact original client bytes after compression + CCR, in case a
transform mutates the frozen region despite the ``frozen_message_count``
hint. Without it the engine would forward compression-mutated prefix bytes
that differ from the prior turn → prompt-cache bust (invariant I2).

These tests model that scenario directly with a stub pipeline that
deliberately mutates a frozen-prefix message, then assert the engine's
behaviour matches the live handler (``handle_openai_chat`` ~L1656):

  * **cache mode** → engine restores the frozen prefix to original bytes;
  * **token mode** → engine does NOT restore (the handler has no restore in
    token mode — stability there comes from the compression cache, not from
    this safety net).

Together they prove the fix fires in cache mode AND does not over-fire in
token mode. The token-mode case also proves the stub genuinely mutates the
prefix, so the cache-mode "restored" result is meaningful (not a no-op).

Running
-------
  .venv/bin/python -m pytest tests/engine/test_facade_openai_cache_restore.py -v
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("fastapi")

# Marker the stub pipeline prepends to the frozen-prefix message's content.
_MUTATION_MARKER = "MUTATED-BY-STUB::"

# Original content of the first (frozen) message — what the restore must
# bring back byte-for-byte in cache mode.
_FROZEN_ORIGINAL = "Frozen prefix message — must survive byte-for-byte"


# ---------------------------------------------------------------------------
# Stub pipeline: models a transform that violates the frozen hint
# ---------------------------------------------------------------------------


class _Result:
    """Minimal stand-in for the pipeline result (engine reads only .messages)."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages


class _FrozenMutatingPipeline:
    """Deliberately mutates the FROZEN prefix message (index 0).

    This is the exact scenario ``_restore_frozen_prefix`` exists to undo: a
    transform that touches the frozen region even though it was told (via
    ``frozen_message_count``) not to.
    """

    def apply(self, *, messages: list[dict[str, Any]], **_kwargs: Any) -> _Result:
        mutated = [dict(m) for m in messages]
        if mutated:
            content = mutated[0].get("content")
            if isinstance(content, str):
                mutated[0]["content"] = _MUTATION_MARKER + content
        return _Result(mutated)


# ---------------------------------------------------------------------------
# Deterministic session-state stubs (mirror the other engine OpenAI tests)
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, frozen_count: int) -> None:
        self._frozen_count = frozen_count

    def compute_session_id(self, ctx: Any, model: str, msgs: Any) -> str:
        return "fix1-cache-restore-session"

    def get_or_create(self, session_id: str, provider: str) -> Any:
        frozen_count = self._frozen_count

        class _T:
            def get_frozen_message_count(self) -> int:
                return frozen_count

            def get_last_original_messages(self) -> list[Any]:
                return []

            def get_last_forwarded_messages(self) -> list[Any]:
                return []

        return _T()

    def get_fresh_cache(self, session_id: str) -> Any:
        class _C:
            def apply_cached(self, msgs: list[Any]) -> list[Any]:
                return list(msgs)

            def compute_frozen_count(self, msgs: list[Any]) -> int:
                # Token mode reassigns frozen_message_count from this; the
                # value is irrelevant to these tests (restore is cache-only),
                # but mirror the real "freeze all but the live turn".
                return max(0, len(msgs) - 1)

            def update_from_result(self, orig: Any, compr: Any) -> None:
                pass

            def mark_stable_from_messages(self, msgs: Any, up_to: int) -> None:
                pass

        return _C()


# ---------------------------------------------------------------------------
# Engine factory — real provider, stub pipeline
# ---------------------------------------------------------------------------


def _build_engine(*, mode: str, frozen_count: int) -> Any:
    """HeadroomEngine wired to OpenAIComponents with the mutating stub pipeline.

    Reuses the real ``openai_provider`` (for ``get_context_limit``) but swaps
    the pipeline for ``_FrozenMutatingPipeline`` so the frozen-region mutation
    is deterministic and independent of real-transform heuristics.
    """
    from headroom.engine.facade import HeadroomEngine, OpenAIComponents
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config = ProxyConfig(
        optimize=True,
        mode=mode,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_inject_system_instructions=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        ccr_proactive_expansion=False,
        image_optimize=False,
    )
    proxy = HeadroomProxy(config)

    store = _Store(frozen_count)
    oc = OpenAIComponents(
        pipeline=_FrozenMutatingPipeline(),
        provider=proxy.openai_provider,
        session_tracker_store=store,
        get_compression_cache=store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )
    return HeadroomEngine(
        pipelines={},
        config=proxy.config,
        usage_reporter=None,
        salt=b"fix1-cache-restore-salt",
        openai_components=oc,
    )


def _make_ctx(messages: list[dict[str, Any]]) -> Any:
    from headroom.engine.contract import Flavor, Provider, RequestContext

    body = {"model": "gpt-4o", "messages": messages}
    return RequestContext(
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers_view={
            "authorization": "Bearer sk-test-openai-key",
            "content-type": "application/json",
        },
        raw_body=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
        session_key="fix1-cache-restore",
        request_id="req-fix1",
    )


# A 3-message conversation ending in a user turn. With frozen_count=2 and
# ``_strict_previous_turn_frozen_count`` (cache mode), indices 0 and 1 are
# frozen; the stub mutates index 0.
def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": _FROZEN_ORIGINAL},
        {"role": "assistant", "content": "Frozen answer"},
        {"role": "user", "content": "Live turn — free to mutate"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engine_restores_frozen_prefix_in_cache_mode() -> None:
    """Cache mode: a transform that mutates the frozen prefix is undone.

    Proves the engine calls ``_restore_frozen_prefix`` (FIX #1). Without the
    fix the engine forwards the stub-mutated index-0 content → cache-bust.
    """
    engine = _build_engine(mode="cache", frozen_count=2)
    decision = engine.on_request(_make_ctx(_messages()))

    out = json.loads(decision.body)
    first = out["messages"][0]["content"]
    assert first == _FROZEN_ORIGINAL, (
        f"cache mode must restore the frozen prefix to original bytes; got {first!r}"
    )
    assert _MUTATION_MARKER not in decision.body.decode(), (
        "the stub's frozen-prefix mutation must not survive in cache mode"
    )


def test_engine_does_not_restore_frozen_prefix_in_token_mode() -> None:
    """Token mode: the engine does NOT restore (matches the handler).

    The handler only calls ``_restore_frozen_prefix`` in cache mode — token
    mode relies on the compression cache for stability. This test also proves
    the stub genuinely mutates index 0 (so the cache-mode assertion above is
    meaningful, not a no-op).
    """
    engine = _build_engine(mode="token", frozen_count=2)
    decision = engine.on_request(_make_ctx(_messages()))

    out = json.loads(decision.body)
    first = out["messages"][0]["content"]
    assert first == _MUTATION_MARKER + _FROZEN_ORIGINAL, (
        "token mode must NOT restore the frozen prefix (no restore in the "
        f"handler's token path); got {first!r}"
    )
