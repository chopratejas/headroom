"""Structural tests for OpenAI memory injection (Chunk 5.2).

Mirrors ``test_facade_memory.py`` (Anthropic) but for the OpenAI chat path.

Key differences from the Anthropic memory tests:
  - Uses ``append_text_to_latest_user_chat_message`` (OpenAI helper), which
    scans backwards through all messages without frozen-count awareness.
  - Does NOT skip injection in cache mode (the OpenAI handler injects in all
    modes — there is no is_cache_mode gate around memory).
  - The engine uses ``_on_request_openai_chat`` via ``OpenAIComponents``.
  - prefetched_memory_context is supplied via RequestContext (4.3-i pattern).

Running
-------
  .venv/bin/python -m pytest tests/engine/test_facade_openai_memory.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIXED_CONTEXT = (
    "## Relevant memory\n- auth_middleware.py: JWT validation\n- auth_router.py: /login route"
)


def _make_engine(
    *,
    memory_handler: Any | None = None,
    with_memory: bool = True,
    config_overrides: dict[str, Any] | None = None,
    frozen_count: int = 0,
) -> Any:
    """Build a HeadroomEngine with OpenAIComponents + MemoryComponents."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import HeadroomEngine, MemoryComponents, OpenAIComponents
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config_kwargs: dict[str, Any] = {
        "optimize": True,
        "mode": "token",
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_inject_system_instructions": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "ccr_proactive_expansion": False,
        "image_optimize": False,
    }
    if config_overrides:
        config_kwargs.update(config_overrides)

    config = ProxyConfig(**config_kwargs)
    proxy = HeadroomProxy(config)

    class _FixedStore:
        def compute_session_id(self, ctx: Any, model: str, msgs: Any) -> str:
            return "memory-openai-structural-test-session"

        def get_or_create(self, session_id: str, provider: str) -> Any:
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
                    return 0

                def update_from_result(self, orig: Any, compr: Any) -> None:
                    pass

                def mark_stable_from_messages(self, msgs: Any, up_to: int) -> None:
                    pass

            return _C()

    oc = OpenAIComponents(
        pipeline=proxy.openai_pipeline,
        provider=proxy.openai_provider,
        session_tracker_store=_FixedStore(),
        get_compression_cache=_FixedStore().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    mc = None
    if with_memory:
        _handler = memory_handler
        if _handler is None:
            _handler = MagicMock()
            _handler.config.inject_context = True

        mc = MemoryComponents(
            memory_handler=_handler,
            default_user_id="test-user",
        )

    engine = HeadroomEngine(
        pipelines={(Provider.OPENAI, Flavor.CHAT): proxy.openai_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"memory-openai-structural-test-salt",
        openai_components=oc,
        memory_components=mc,
    )
    return engine


def _make_ctx(
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    prefetched_memory_context: str | None = None,
) -> Any:
    """Build a RequestContext for OpenAI structural tests."""
    from headroom.engine.contract import Flavor, Provider, RequestContext

    h: dict[str, str] = {
        "authorization": "Bearer sk-test-openai-key",
        "content-type": "application/json",
    }
    if headers:
        h.update(headers)

    return RequestContext(
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers_view=h,
        raw_body=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
        session_key="memory-openai-structural",
        request_id="req-openai-mem-test",
        prefetched_memory_context=prefetched_memory_context,
    )


# ---------------------------------------------------------------------------
# 1. Memory off → no-op (MemoryComponents=None)
# ---------------------------------------------------------------------------


def test_memory_noop_when_components_none_openai() -> None:
    """Engine without MemoryComponents is byte-identical — OpenAI golden fixtures unaffected."""
    engine = _make_engine(with_memory=False)

    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    assert _FIXED_CONTEXT not in out["messages"][-1]["content"], (
        "Memory context must not appear when MemoryComponents is None"
    )


# ---------------------------------------------------------------------------
# 2. Byte-exact placement: string content
# ---------------------------------------------------------------------------


def test_memory_injection_string_content_openai() -> None:
    """Memory-on: context appended to latest user message (string content).

    Placement rule: ``original_text + "\\n\\n" + context`` — same as
    ``append_text_to_latest_user_chat_message`` in the handler.
    """
    user_text = "How does authentication work?"
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": user_text}],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    engine = _make_engine()
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    assert last_msg["role"] == "user"
    content = last_msg["content"]
    assert isinstance(content, str)
    assert content == user_text + "\n\n" + _FIXED_CONTEXT


def test_memory_injection_list_content_openai() -> None:
    """Memory-on: context appended to first text block (list content)."""
    original_text = "What is the rate limit?"
    body = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": original_text},
                    {"type": "text", "text": "Additional detail."},
                ],
            }
        ],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    engine = _make_engine()
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    blocks = last_msg["content"]
    assert isinstance(blocks, list)
    # First text block gets the injection.
    assert blocks[0]["text"] == original_text + "\n\n" + _FIXED_CONTEXT
    # Second text block unchanged.
    assert blocks[1]["text"] == "Additional detail."


# ---------------------------------------------------------------------------
# 3. Cache mode: memory injection NOT skipped (OpenAI differs from Anthropic)
# ---------------------------------------------------------------------------


def test_memory_injection_not_skipped_in_cache_mode_openai() -> None:
    """OpenAI: memory injection is NOT skipped in cache mode.

    The live ``handle_openai_chat`` handler has no is_cache_mode gate around
    the memory injection block. The Anthropic engine skips injection in cache
    mode; the OpenAI engine must NOT.
    """
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Cache mode test"}],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    engine = _make_engine(config_overrides={"mode": "cache"})
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    last_msg = out["messages"][-1]
    content = last_msg.get("content", "")
    assert _FIXED_CONTEXT in content, "OpenAI memory injection must NOT be skipped in cache mode"


# ---------------------------------------------------------------------------
# 4. Memory injection scans backwards (frozen-count agnostic)
# ---------------------------------------------------------------------------


def test_memory_injection_scans_backwards_to_latest_user_openai() -> None:
    """Memory is injected into the latest user message regardless of frozen count.

    ``append_text_to_latest_user_chat_message`` scans backwards from the end
    of the message array without frozen-count awareness — the OpenAI handler
    does not pass a frozen count to the helper.
    """
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Turn 1 (earlier)"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Turn 2 (latest)"},
        ],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    # frozen_count=2 — but the helper ignores it; latest user turn still receives context.
    engine = _make_engine(frozen_count=2)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    msgs = out["messages"]
    # Latest (last) user turn must have context injected.
    assert msgs[-1]["role"] == "user"
    assert _FIXED_CONTEXT in msgs[-1]["content"]
    # Earlier user turn must be unchanged.
    assert _FIXED_CONTEXT not in msgs[0]["content"]


# ---------------------------------------------------------------------------
# 5. Empty / None prefetched context → no-op
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_on_empty_context_openai() -> None:
    """Empty prefetched_memory_context → no injection."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    ctx = _make_ctx(body, prefetched_memory_context="")
    engine = _make_engine()
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    assert out["messages"][-1]["content"] == "Hello"


def test_memory_injection_skipped_on_none_context_openai() -> None:
    """None prefetched_memory_context → no injection."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    ctx = _make_ctx(body, prefetched_memory_context=None)
    engine = _make_engine()
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    assert out["messages"][-1]["content"] == "Hello"


# ---------------------------------------------------------------------------
# 6. inject_context=False gate
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_when_inject_context_false_openai() -> None:
    """memory_handler.config.inject_context=False → injection skipped."""
    handler_mock = MagicMock()
    handler_mock.config.inject_context = False

    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "No inject_context test"}],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    engine = _make_engine(memory_handler=handler_mock)
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    assert _FIXED_CONTEXT not in out["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 7. Bypass gate
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_on_bypass_openai() -> None:
    """x-headroom-bypass: true → memory injection is skipped."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Bypass test"}],
    }
    ctx = _make_ctx(
        body,
        headers={"x-headroom-bypass": "true"},
        prefetched_memory_context=_FIXED_CONTEXT,
    )
    engine = _make_engine()
    decision = engine.on_request(ctx)

    assert _FIXED_CONTEXT.encode() not in decision.body, (
        "Memory context must not appear in body under bypass"
    )


# ---------------------------------------------------------------------------
# 8. No user_id → MemoryDecision gate fails
# ---------------------------------------------------------------------------


def test_memory_injection_skipped_when_no_user_id_openai() -> None:
    """No user_id → MemoryDecision.inject=False → no injection.

    Supply MemoryComponents with default_user_id="" and no x-headroom-user-id
    header. MemoryDecision.decide gates on memory_user_id being non-empty.
    """
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import HeadroomEngine, MemoryComponents, OpenAIComponents
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
    )
    proxy = HeadroomProxy(config)

    class _S:
        def compute_session_id(self, *a: Any, **kw: Any) -> str:
            return "s"

        def get_or_create(self, *a: Any, **kw: Any) -> Any:
            class _T:
                def get_frozen_message_count(self) -> int:
                    return 0

                def get_last_original_messages(self) -> list:
                    return []

                def get_last_forwarded_messages(self) -> list:
                    return []

            return _T()

        def get_fresh_cache(self, sid: str) -> Any:
            class _C:
                def apply_cached(self, m: list) -> list:
                    return list(m)

                def compute_frozen_count(self, m: list) -> int:
                    return 0

                def update_from_result(self, *a: Any) -> None:
                    pass

                def mark_stable_from_messages(self, *a: Any) -> None:
                    pass

            return _C()

    handler_mock = MagicMock()
    handler_mock.config.inject_context = True

    oc = OpenAIComponents(
        pipeline=proxy.openai_pipeline,
        provider=proxy.openai_provider,
        session_tracker_store=_S(),
        get_compression_cache=_S().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )
    mc = MemoryComponents(
        memory_handler=handler_mock,
        default_user_id="",  # empty → gate fails
    )
    engine = HeadroomEngine(
        pipelines={(Provider.OPENAI, Flavor.CHAT): proxy.openai_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"s",
        openai_components=oc,
        memory_components=mc,
    )

    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "No user_id test"}],
    }
    # No x-headroom-user-id header and default_user_id="" → gate fails.
    from headroom.engine.contract import RequestContext

    ctx = RequestContext(
        provider=Provider.OPENAI,
        flavor=Flavor.CHAT,
        headers_view={"authorization": "Bearer sk-test"},
        raw_body=json.dumps(body, separators=(",", ":")).encode(),
        session_key="s",
        request_id="",
        prefetched_memory_context=_FIXED_CONTEXT,
    )
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    assert _FIXED_CONTEXT not in out["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 9. Multi-turn body: context lands in latest user message (not assistant)
# ---------------------------------------------------------------------------


def test_memory_injection_skips_assistant_tail_openai() -> None:
    """Memory injection finds the last user message even when last message is assistant."""
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Ack"},
        ],
    }
    ctx = _make_ctx(body, prefetched_memory_context=_FIXED_CONTEXT)
    engine = _make_engine()
    decision = engine.on_request(ctx)

    out = json.loads(decision.body)
    msgs = out["messages"]
    # Last message is assistant — the helper scans back and finds the user turn.
    assert msgs[-2]["role"] == "user"
    assert _FIXED_CONTEXT in msgs[-2]["content"]
    # Assistant message is unchanged.
    assert _FIXED_CONTEXT not in msgs[-1]["content"]
