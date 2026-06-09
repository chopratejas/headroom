"""Structural tests for CCR request-side steps (Chunk 4.2b).

These tests verify that the HeadroomEngine's CCR integration (steps 1-4) calls
the correct callables with the correct arguments, using mock/stub objects.
They are structural rather than byte-exact because:
  - Compression tracking (step 3): verifies track_compression is called with
    correct hash/turn/workspace args — the call args are deterministic but
    the outbound body is unchanged (no body mutation from tracking).
  - Proactive expansion (step 4): analyze_query/execute_expansions are ML-
    nondeterministic; the test seeds a deterministic stub and asserts the
    expansion text lands in the correct user turn.

Running
-------
  .venv/bin/python -m pytest tests/engine/test_facade_ccr.py -v
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

_CCR_TEST_HASH = "abcdef123456789012345678"
_CCR_MARKER = f"[100 items compressed to 10. Retrieve more: hash={_CCR_TEST_HASH}]"
_WORKSPACE_KEY = "myproject-abcd1234ef56"
_WORKSPACE_LABEL = "myproject"


def _make_engine(
    *,
    ccr_context_tracker: Any | None = None,
    get_compression_store: Any | None = None,
    session_turn_counters: dict[str, int] | None = None,
    config_overrides: dict[str, Any] | None = None,
    frozen_count: int = 0,
) -> Any:
    """Build a real HeadroomEngine with CCRComponents for structural tests."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, CCRComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config_kwargs: dict[str, Any] = {
        "optimize": True,
        "mode": "token",
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": True,
        "ccr_inject_system_instructions": True,
        "ccr_handle_responses": False,
        "ccr_context_tracking": True,
        "ccr_proactive_expansion": False,  # disabled by default; tests enable per-case
        "image_optimize": False,
    }
    if config_overrides:
        config_kwargs.update(config_overrides)

    config = ProxyConfig(**config_kwargs)
    proxy = HeadroomProxy(config)

    # Minimal session tracker — fixed frozen count, returns fresh cache.
    class _FixedStore:
        def compute_session_id(self, ctx: Any, model: str, msgs: Any) -> str:
            return "ccr-structural-test-session"

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

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=_FixedStore(),
        get_compression_cache=_FixedStore().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )

    store = get_compression_store or (lambda: MagicMock())
    ccr = CCRComponents(
        ccr_context_tracker=ccr_context_tracker,
        get_compression_store=store,
        session_turn_counters=session_turn_counters if session_turn_counters is not None else {},
    )

    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"ccr-structural-test-salt",
        anthropic_components=ac,
        ccr_components=ccr,
    )
    return engine


def _make_ctx(
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    cwd: str | None = None,
) -> Any:
    """Build a RequestContext for structural tests."""
    from headroom.engine.contract import Flavor, Provider, RequestContext

    h: dict[str, str] = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if cwd:
        h["x-headroom-cwd"] = cwd
    if headers:
        h.update(headers)

    return RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view=h,
        raw_body=json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
        session_key="ccr-structural",
        request_id="req-struct-test",
    )


# ---------------------------------------------------------------------------
# Tests — CCR is no-op when ccr_components is None (regression guard)
# ---------------------------------------------------------------------------


def test_ccr_noop_when_components_none() -> None:
    """Engine without CCRComponents is byte-identical to 4.2a behaviour."""
    from headroom.engine.contract import Flavor, Provider
    from headroom.engine.facade import AnthropicComponents, HeadroomEngine
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )
    proxy = HeadroomProxy(config)

    class _TrivialStore:
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

        def get_fresh_cache(self, session_id: str) -> Any:
            class _C:
                def apply_cached(self, msgs: list) -> list:
                    return msgs

                def compute_frozen_count(self, msgs: list) -> int:
                    return 0

                def update_from_result(self, *a: Any) -> None:
                    pass

                def mark_stable_from_messages(self, *a: Any) -> None:
                    pass

            return _C()

    ac = AnthropicComponents(
        pipeline=proxy.anthropic_pipeline,
        provider=proxy.anthropic_provider,
        session_tracker_store=_TrivialStore(),
        get_compression_cache=_TrivialStore().get_fresh_cache,
        config=proxy.config,
        usage_reporter=None,
    )
    engine = HeadroomEngine(
        pipelines={(Provider.ANTHROPIC, Flavor.MESSAGES): proxy.anthropic_pipeline},
        config=proxy.config,
        usage_reporter=None,
        salt=b"s",
        anthropic_components=ac,
        ccr_components=None,  # No CCR
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }
    from headroom.engine.contract import Flavor, Provider, RequestContext

    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"x-api-key": "k", "anthropic-version": "2023-06-01"},
        raw_body=json.dumps(body).encode(),
        session_key="s",
        request_id="",
    )
    decision = engine.on_request(ctx)
    # passthrough (optimize=False) → raw body returned unchanged
    assert decision.body == ctx.raw_body


# ---------------------------------------------------------------------------
# Tests — step 3: compression tracking
# ---------------------------------------------------------------------------


def test_compression_tracking_called_with_correct_args() -> None:
    """Step 3: track_compression is called with the correct hash, turn, and workspace.

    The body contains a pre-formed CCR marker (24-hex hash) in a tool_result.
    The engine should scan for it, resolve the workspace from x-headroom-cwd,
    look up the store entry, and call track_compression with matching args.
    """
    mock_tracker = MagicMock()
    mock_store = MagicMock()

    # Set up a fake store entry for _CCR_TEST_HASH.
    mock_store.get_metadata.return_value = {
        "tool_name": "search_files",
        "original_item_count": 100,
        "compressed_item_count": 10,
        "query_context": "find auth files",
        "compressed_content": "auth_middleware.py\nauth_router.py\n",
    }

    session_turn_counters: dict[str, int] = {}

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        get_compression_store=lambda: mock_store,
        session_turn_counters=session_turn_counters,
        config_overrides={
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": True,
            "ccr_proactive_expansion": False,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_x",
                        "content": f"Found files. {_CCR_MARKER}",
                    }
                ],
            }
        ],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")

    engine.on_request(ctx)

    # session_turn_counters should have a non-zero entry for the test session.
    # The session key is produced by _FixedStore.compute_session_id → "ccr-structural-test-session".
    assert "ccr-structural-test-session" in session_turn_counters, (
        "session_turn_counters must contain the test session key after a compression event"
    )
    assert session_turn_counters["ccr-structural-test-session"] == 1, (
        "Turn counter for this session must be 1 after the first compression event"
    )

    # get_metadata should have been called for the test hash.
    mock_store.get_metadata.assert_called_once_with(_CCR_TEST_HASH)

    # track_compression should have been called with the correct args.
    mock_tracker.track_compression.assert_called_once()
    call_kwargs = mock_tracker.track_compression.call_args[1]
    # Keyword-only args
    assert call_kwargs["hash_key"] == _CCR_TEST_HASH
    assert call_kwargs["turn_number"] == 1
    assert call_kwargs["tool_name"] == "search_files"
    assert call_kwargs["original_count"] == 100
    assert call_kwargs["compressed_count"] == 10
    # workspace_key must be non-empty (resolved from x-headroom-cwd)
    assert call_kwargs["workspace_key"], "workspace_key must not be empty"


def test_compression_tracking_skipped_when_no_workspace() -> None:
    """Step 3 fail-closed: track_compression is NOT called when workspace unresolvable.

    Without x-headroom-cwd / x-headroom-project-id / cwd: in system prompt,
    workspace resolves to ("", None). The engine should log and skip tracking.
    """
    mock_tracker = MagicMock()
    mock_store = MagicMock()
    mock_store.get_metadata.return_value = {
        "tool_name": "t",
        "original_item_count": 5,
        "compressed_item_count": 1,
        "query_context": "",
        "compressed_content": "",
    }
    session_turn_counters: dict[str, int] = {}

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        get_compression_store=lambda: mock_store,
        session_turn_counters=session_turn_counters,
        config_overrides={
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": True,
            "ccr_proactive_expansion": False,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": f"Tool result: {_CCR_MARKER}",
            }
        ],
    }
    # No cwd header, no system prompt cwd: line → workspace unresolvable.
    ctx = _make_ctx(body)

    engine.on_request(ctx)

    # session_turn_counters is NOT incremented (gated on workspace_key).
    assert session_turn_counters.get("ccr-structural-test-session", 0) == 0, (
        "Turn counter must not be incremented when workspace is unresolvable"
    )
    # track_compression is NOT called (fail-closed).
    mock_tracker.track_compression.assert_not_called()


def test_compression_tracking_skipped_when_no_ccr_marker() -> None:
    """Step 3: track_compression is NOT called when no compression marker present."""
    mock_tracker = MagicMock()
    mock_store = MagicMock()
    session_turn_counters: dict[str, int] = {}

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        get_compression_store=lambda: mock_store,
        session_turn_counters=session_turn_counters,
        config_overrides={
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": True,
            "ccr_proactive_expansion": False,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "No compression here."}],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")

    engine.on_request(ctx)

    assert session_turn_counters.get("ccr-structural-test-session", 0) == 0
    mock_tracker.track_compression.assert_not_called()
    mock_store.get_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — step 4: proactive expansion
# ---------------------------------------------------------------------------


def test_proactive_expansion_injects_into_latest_user_turn() -> None:
    """Step 4: expansion text lands in the latest non-frozen user turn.

    analyze_query is seeded to return a recommendation. execute_expansions
    returns a canned expansion. The test asserts the expansion text appears
    in the outbound body's last user message, prepended to the original text.
    """
    expansion_text = "[CCR EXPANSION] Relevant context: auth_middleware.py contents..."

    mock_tracker = MagicMock()
    mock_tracker.analyze_query.return_value = [
        MagicMock(hash_key=_CCR_TEST_HASH, reason="query matches auth context")
    ]
    mock_tracker.execute_expansions.return_value = [
        MagicMock(hash_key=_CCR_TEST_HASH, content="auth_middleware.py contents...")
    ]
    mock_tracker.format_expansions_for_context.return_value = expansion_text

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        config_overrides={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": False,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": False,
            "ccr_proactive_expansion": True,
        },
    )

    user_query = "Where is the auth middleware defined?"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": user_query}],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")

    decision = engine.on_request(ctx)
    out = json.loads(decision.body)

    # analyze_query must have been called (step 4 gate passed).
    mock_tracker.analyze_query.assert_called_once()
    call_args = mock_tracker.analyze_query.call_args
    assert call_args[0][0] == user_query, "analyze_query must receive the user query"
    assert call_args[1]["workspace_key"], "workspace_key must be non-empty"

    # execute_expansions and format_expansions_for_context must have been called.
    mock_tracker.execute_expansions.assert_called_once()
    mock_tracker.format_expansions_for_context.assert_called_once()

    # Expansion text must appear in the last user message.
    last_msg = out["messages"][-1]
    assert last_msg["role"] == "user"
    content = last_msg.get("content", "")
    if isinstance(content, str):
        assert expansion_text in content, (
            f"Expected expansion text in last user message content.\nGot: {content!r}"
        )
    elif isinstance(content, list):
        text_blocks = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert any(expansion_text in t for t in text_blocks), (
            f"Expected expansion text in a text block of last user message.\n"
            f"Got text blocks: {text_blocks!r}"
        )
    else:
        pytest.fail(f"Unexpected content type: {type(content)}")


def test_proactive_expansion_skipped_in_cache_mode() -> None:
    """Step 4: expansion is skipped in cache mode to preserve prefix stability."""
    mock_tracker = MagicMock()
    mock_tracker.analyze_query.return_value = [MagicMock()]
    mock_tracker.execute_expansions.return_value = [MagicMock()]
    mock_tracker.format_expansions_for_context.return_value = "EXPANSION"

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        config_overrides={
            "optimize": True,
            "mode": "cache",
            "ccr_inject_tool": False,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": False,
            "ccr_proactive_expansion": True,
        },
    )

    user_query = "Find the auth module"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": user_query}],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")

    decision = engine.on_request(ctx)
    out = json.loads(decision.body)

    # analyze_query fired (recommendations exist).
    mock_tracker.analyze_query.assert_called_once()
    mock_tracker.execute_expansions.assert_called_once()
    mock_tracker.format_expansions_for_context.assert_called_once()

    # But expansion text must NOT appear in the last user turn (cache mode skip).
    last_msg = out["messages"][-1]
    content = last_msg.get("content", "")
    if isinstance(content, str):
        assert "EXPANSION" not in content, "Expansion must not be injected in cache mode"
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                assert "EXPANSION" not in b.get("text", ""), (
                    "Expansion must not be injected in cache mode"
                )


def test_proactive_expansion_skipped_when_no_workspace() -> None:
    """Step 4: expansion is skipped (fail-closed) when workspace unresolvable."""
    mock_tracker = MagicMock()
    mock_tracker.analyze_query.return_value = [MagicMock()]
    mock_tracker.execute_expansions.return_value = [MagicMock()]
    mock_tracker.format_expansions_for_context.return_value = "EXPANSION"

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        config_overrides={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": False,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": False,
            "ccr_proactive_expansion": True,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Find auth files"}],
    }
    # No cwd header → workspace unresolvable → skip expansion.
    ctx = _make_ctx(body)

    engine.on_request(ctx)

    # analyze_query must NOT be called (workspace gate failed).
    mock_tracker.analyze_query.assert_not_called()


def test_proactive_expansion_skipped_when_no_recommendations() -> None:
    """Step 4: expansion is skipped gracefully when analyze_query returns no recs."""
    mock_tracker = MagicMock()
    mock_tracker.analyze_query.return_value = []  # No recommendations

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        config_overrides={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": False,
            "ccr_inject_system_instructions": False,
            "ccr_context_tracking": False,
            "ccr_proactive_expansion": True,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Something unrelated"}],
    }
    ctx = _make_ctx(body, cwd="/home/user/myproject")

    engine.on_request(ctx)

    mock_tracker.analyze_query.assert_called_once()
    # execute_expansions not called when no recommendations.
    mock_tracker.execute_expansions.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — workspace resolution
# ---------------------------------------------------------------------------


def test_workspace_resolution_from_project_id_header() -> None:
    """Step 1: x-headroom-project-id header resolves workspace."""
    from headroom.engine.facade import _resolve_ccr_workspace

    headers = {"x-headroom-project-id": "myapp-prod"}
    body: dict[str, Any] = {"messages": []}

    key, label = _resolve_ccr_workspace(headers, body)
    assert key, "workspace_key must be non-empty for a valid project-id"
    assert label == "myapp-prod"


def test_workspace_resolution_from_cwd_header() -> None:
    """Step 1: x-headroom-cwd header resolves workspace when project-id absent."""
    from headroom.engine.facade import _resolve_ccr_workspace

    headers = {"x-headroom-cwd": "/home/user/my-project"}
    body: dict[str, Any] = {"messages": []}

    key, label = _resolve_ccr_workspace(headers, body)
    assert key, "workspace_key must be non-empty for a valid cwd"
    assert label == "my-project"


def test_workspace_resolution_from_system_prompt_cwd() -> None:
    """Step 1: cwd: line in system prompt resolves workspace when no headers."""
    from headroom.engine.facade import _resolve_ccr_workspace

    headers: dict[str, str] = {}
    body: dict[str, Any] = {
        "system": "You are a coding assistant.\ncwd: /home/user/backend\nBe helpful.",
        "messages": [],
    }

    key, label = _resolve_ccr_workspace(headers, body)
    assert key, "workspace_key must be non-empty when cwd: is in system prompt"
    assert label == "backend"


def test_workspace_resolution_returns_empty_on_no_signal() -> None:
    """Step 1 fail-closed: returns ('', None) when no workspace signal present."""
    from headroom.engine.facade import _resolve_ccr_workspace

    headers: dict[str, str] = {}
    body: dict[str, Any] = {"system": "You are helpful.", "messages": []}

    key, label = _resolve_ccr_workspace(headers, body)
    assert key == ""
    assert label is None


# ---------------------------------------------------------------------------
# Tests — bypass gate
# ---------------------------------------------------------------------------


def test_ccr_skipped_on_bypass_header() -> None:
    """All CCR steps are skipped when x-headroom-bypass: true is set."""
    mock_tracker = MagicMock()
    mock_store = MagicMock()

    engine = _make_engine(
        ccr_context_tracker=mock_tracker,
        get_compression_store=lambda: mock_store,
        config_overrides={
            "optimize": True,
            "mode": "token",
            "ccr_inject_tool": True,
            "ccr_inject_system_instructions": True,
            "ccr_context_tracking": True,
            "ccr_proactive_expansion": True,
        },
    )

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": f"Tool result with marker: {_CCR_MARKER}",
            }
        ],
    }
    ctx = _make_ctx(
        body,
        headers={"x-headroom-bypass": "true"},
        cwd="/home/user/myproject",
    )

    decision = engine.on_request(ctx)

    # Bypass returns original bytes unchanged.
    assert decision.body == ctx.raw_body
    # CCR tracker was never touched.
    mock_tracker.track_compression.assert_not_called()
    mock_tracker.analyze_query.assert_not_called()
