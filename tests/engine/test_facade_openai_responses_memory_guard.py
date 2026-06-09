"""Proving test for FIX #3 — OpenAI /v1/responses memory fail-safe guard.

The live ``handle_openai_responses`` handler injects memory (context into
``body["input"]`` + sticky tools into ``body["tools"]``) BEFORE compression.
The engine's ``_on_request_openai_responses`` does not yet reproduce that, and
the shadow CANNOT catch the gap because the parity corpus runs with memory
off. In a flipped (``on``) deployment that would silently drop the user's
memory context and bust the prompt cache.

The fix makes the engine refuse LOUDLY (``EngineResponsesMemoryUnsupportedError``)
exactly when memory WOULD inject, so the handler's ``on``-mode fallback forwards
the proven legacy bytes and ``shadow`` records an error — never a silent diverge.

These tests assert the guard fires iff the handler would inject, mirroring
``MemoryDecision``:

  * memory handler present + user_id + not bypass + mode!=disabled → RAISE;
  * bypass header → no raise (handler also skips memory under bypass);
  * memory handler None (the golden-corpus shape) → no raise;
  * mode=disabled → no raise.

Running
-------
  .venv/bin/python -m pytest tests/engine/test_facade_openai_responses_memory_guard.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# Deterministic session-state stub (responses path uses it via compression)
# ---------------------------------------------------------------------------


class _FixedStore:
    def compute_session_id(self, ctx: Any, model: str, msgs: Any) -> str:
        return "fix3-responses-guard-session"

    def get_or_create(self, session_id: str, provider: str) -> Any:
        class _T:
            def get_frozen_message_count(self) -> int:
                return 0

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
                return 0

            def update_from_result(self, orig: Any, compr: Any) -> None:
                pass

            def mark_stable_from_messages(self, msgs: Any, up_to: int) -> None:
                pass

        return _C()


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _build_engine(*, with_memory_handler: bool, default_user_id: str = "test-user") -> Any:
    """HeadroomEngine wired to OpenAIComponents + MemoryComponents.

    ``with_memory_handler=False`` models the golden-corpus / memory-off
    deployment, where ``MemoryComponents.memory_handler`` is None (the engine
    builder always constructs MemoryComponents, with a None handler when memory
    is disabled).
    """
    from headroom.engine.facade import HeadroomEngine, MemoryComponents, OpenAIComponents
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config = ProxyConfig(
        optimize=True,
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

    store = _FixedStore()
    oc = OpenAIComponents(
        pipeline=proxy.openai_pipeline,
        provider=proxy.openai_provider,
        session_tracker_store=store,
        get_compression_cache=store.get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    handler: Any | None = None
    if with_memory_handler:
        handler = MagicMock()
        handler.config.inject_context = True
    mc = MemoryComponents(memory_handler=handler, default_user_id=default_user_id)

    return HeadroomEngine(
        pipelines={},
        config=proxy.config,
        usage_reporter=None,
        salt=b"fix3-responses-guard-salt",
        openai_components=oc,
        memory_components=mc,
    )


def _make_ctx(*, headers: dict[str, str] | None = None) -> Any:
    from headroom.engine.contract import Flavor, Provider, RequestContext

    h: dict[str, str] = {
        "authorization": "Bearer sk-test-openai-key",
        "content-type": "application/json",
        "x-headroom-user-id": "test-user",
    }
    if headers:
        h.update(headers)

    body = {"model": "gpt-4o", "input": "Hello from the Responses API"}
    return RequestContext(
        provider=Provider.OPENAI,
        flavor=Flavor.RESPONSES,
        headers_view=h,
        raw_body=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
        session_key="fix3-responses-guard",
        request_id="req-fix3",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_responses_raises_when_memory_would_inject(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory backend present + user_id + not bypass + mode!=disabled → LOUD refuse.

    Without the guard the engine would silently forward memory-less bytes
    (the shadow can't catch it). The handler's `on`-mode fallback relies on
    this raise to route the request to the legacy (memory-correct) path.
    """
    from headroom.engine.facade import EngineResponsesMemoryUnsupportedError

    # Force the default inject mode regardless of ambient env.
    monkeypatch.delenv("HEADROOM_MEMORY_INJECTION_MODE", raising=False)

    engine = _build_engine(with_memory_handler=True)
    with pytest.raises(EngineResponsesMemoryUnsupportedError):
        engine.on_request(_make_ctx())


def test_responses_bypass_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass header → guard skipped (handler also skips memory under bypass)."""
    monkeypatch.delenv("HEADROOM_MEMORY_INJECTION_MODE", raising=False)

    engine = _build_engine(with_memory_handler=True)
    decision = engine.on_request(_make_ctx(headers={"x-headroom-bypass": "true"}))
    # Bypass → raw inbound bytes forwarded unchanged.
    assert decision.body
    assert json.loads(decision.body)["input"] == "Hello from the Responses API"


def test_responses_no_raise_when_memory_handler_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_handler is None (memory-off deployment / golden corpus) → no raise."""
    monkeypatch.delenv("HEADROOM_MEMORY_INJECTION_MODE", raising=False)

    engine = _build_engine(with_memory_handler=False)
    decision = engine.on_request(_make_ctx())
    assert decision.body  # normal responses output, no exception


def test_responses_no_raise_when_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEADROOM_MEMORY_INJECTION_MODE=disabled → MemoryDecision.inject False → no raise."""
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "disabled")

    engine = _build_engine(with_memory_handler=True)
    decision = engine.on_request(_make_ctx())
    assert decision.body  # memory disabled → engine handles it normally
