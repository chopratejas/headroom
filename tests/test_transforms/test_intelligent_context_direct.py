"""Direct helper coverage for intelligent_context."""

from __future__ import annotations

from types import SimpleNamespace

from headroom.config import IntelligentContextConfig
from headroom.transforms.intelligent_context import (
    IntelligentContextManager,
    _create_message_signature,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))


def test_create_message_signature_tracks_roles_tools_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.transforms.error_detection.content_has_error_indicators",
        lambda text: "error" in text.lower(),
    )

    signature = _create_message_signature(
        [
            {"role": "system", "content": "normal"},
            {"role": "tool", "content": "fatal error"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool_1"}],
                "tool_calls": [{"id": "call_1"}],
            },
        ]
    )

    assert signature is not None
    assert signature.field_count == 3
    assert signature.has_nested_objects is True
    assert signature.max_depth == 1


def test_apply_compress_first_handles_missing_router_and_successful_tool_compression() -> None:
    manager = IntelligentContextManager()
    messages = [{"role": "tool", "content": "x " * 120}]

    manager._content_router = None
    manager._get_content_router = lambda: None  # type: ignore[method-assign]
    unchanged, transforms, saved = manager._apply_compress_first(messages, _Tokenizer(), set())
    assert unchanged == messages
    assert transforms == []
    assert saved == 0

    manager = IntelligentContextManager()
    manager._get_content_router = lambda: SimpleNamespace(  # type: ignore[method-assign]
        compress=lambda content, context=None: SimpleNamespace(
            compressed="short result",
            compression_ratio=0.5,
            strategy_used=SimpleNamespace(value="json"),
        )
    )

    compressed, transforms, saved = manager._apply_compress_first(messages, _Tokenizer(), set())

    assert compressed[0]["content"] == "short result"
    assert transforms == ["compress_first:json:0"]
    assert saved > 0


def test_compress_content_blocks_covers_non_dict_short_and_exception_paths() -> None:
    manager = IntelligentContextManager()
    router = SimpleNamespace(
        compress=lambda content, context="": (_ for _ in ()).throw(RuntimeError("boom"))
    )
    blocks = [
        "raw",
        {"type": "tool_result", "content": "short"},
        {"type": "tool_result", "content": "long " * 100},
        {"type": "text", "text": "keep"},
    ]

    compressed, transforms, saved = manager._compress_content_blocks(blocks, router, _Tokenizer())

    assert compressed == blocks
    assert transforms == []
    assert saved == 0


def test_compress_content_blocks_compresses_large_tool_results() -> None:
    manager = IntelligentContextManager()
    router = SimpleNamespace(
        compress=lambda content, context="": SimpleNamespace(
            compressed="shrunk content",
            compression_ratio=0.5,
            strategy_used=SimpleNamespace(value="json"),
        )
    )
    blocks = [{"type": "tool_result", "content": "long " * 120}]

    compressed, transforms, saved = manager._compress_content_blocks(blocks, router, _Tokenizer())

    assert compressed == [{"type": "tool_result", "content": "shrunk content"}]
    assert transforms == ["compress_first:block:json"]
    assert saved > 0


def test_apply_summarize_and_store_dropped_in_ccr_cover_success_and_failures() -> None:
    manager = IntelligentContextManager()
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]

    manager._get_progressive_summarizer = lambda: None  # type: ignore[method-assign]
    unchanged, transforms, saved = manager._apply_summarize(messages, _Tokenizer(), {1}, 10)
    assert unchanged == messages
    assert transforms == []
    assert saved == 0

    manager._get_progressive_summarizer = lambda: SimpleNamespace(  # type: ignore[method-assign]
        summarize_messages=lambda **kwargs: SimpleNamespace(
            messages=[{"role": "system", "content": "summary"}],
            transforms_applied=["summarize:1"],
            tokens_before=20,
            tokens_after=8,
            summaries_created=["s1"],
        )
    )
    summarized, transforms, saved = manager._apply_summarize(messages, _Tokenizer(), {1}, 10)
    assert summarized == [{"role": "system", "content": "summary"}]
    assert transforms == ["summarize:1"]
    assert saved == 12

    manager._get_progressive_summarizer = lambda: SimpleNamespace(  # type: ignore[method-assign]
        summarize_messages=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    unchanged, transforms, saved = manager._apply_summarize(messages, _Tokenizer(), {1}, 10)
    assert unchanged == messages
    assert transforms == []
    assert saved == 0

    manager._get_compression_store = lambda: None  # type: ignore[method-assign]
    assert manager._store_dropped_in_ccr(messages, {0}) is None

    stored = {}
    manager._get_compression_store = lambda: SimpleNamespace(  # type: ignore[method-assign]
        store=lambda **kwargs: stored.setdefault("payload", kwargs) or "ref-1"
    )
    ref = manager._store_dropped_in_ccr(messages, {0, 1})
    assert ref == stored["payload"] or ref == "ref-1"
    assert "Dropped 2 messages" in stored["payload"]["compressed"]

    manager._get_compression_store = lambda: SimpleNamespace(  # type: ignore[method-assign]
        store=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("store fail"))
    )
    assert manager._store_dropped_in_ccr(messages, {0}) is None


def test_record_drops_to_toin_handles_none_empty_success_and_failure(monkeypatch) -> None:
    manager = IntelligentContextManager()
    manager._record_drops_to_toin([{"role": "user", "content": "hello"}], {0}, 10)

    manager = IntelligentContextManager(toin=SimpleNamespace())
    manager._record_drops_to_toin([{"role": "user", "content": "hello"}], set(), 10)

    manager = IntelligentContextManager(
        toin=SimpleNamespace(record_compression=lambda **kwargs: kwargs)
    )
    monkeypatch.setattr(
        "headroom.transforms.intelligent_context._create_message_signature",
        lambda messages: None,
    )
    manager._record_drops_to_toin([{"role": "user", "content": "hello"}], {0}, 10)

    recorded = {}
    manager = IntelligentContextManager(
        toin=SimpleNamespace(record_compression=lambda **kwargs: recorded.update(kwargs))
    )
    monkeypatch.setattr(
        "headroom.transforms.intelligent_context._create_message_signature",
        lambda messages: SimpleNamespace(structure_hash="abc123"),
    )
    manager._record_drops_to_toin(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
        {0, 1},
        42,
    )
    assert recorded["original_count"] == 2
    assert recorded["compressed_tokens"] == 50
    assert recorded["strategy"] == "intelligent_context_drop"

    manager = IntelligentContextManager(
        toin=SimpleNamespace(
            record_compression=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail"))
        )
    )
    manager._record_drops_to_toin([{"role": "user", "content": "hello"}], {0}, 10)


def test_get_protected_indices_tracks_system_recent_turns_and_tool_links() -> None:
    manager = IntelligentContextManager(
        config=IntelligentContextConfig(keep_last_turns=2, keep_system=True)
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older"},
        {
            "role": "assistant",
            "content": "tooling",
            "tool_calls": [{"id": "call-1"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1"}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "result"}],
        },
    ]

    protected = manager._get_protected_indices(messages)

    assert 0 in protected
    assert 2 in protected
    assert 3 in protected
    assert 4 in protected
    assert 5 in protected


def test_build_scored_drop_candidates_groups_tool_units_turns_and_single_messages() -> None:
    manager = IntelligentContextManager()
    messages = [
        {"role": "assistant", "content": "tool caller"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": "orphan"},
    ]
    scores = [
        SimpleNamespace(total_score=0.2),
        SimpleNamespace(total_score=0.3),
        SimpleNamespace(total_score=0.1),
        SimpleNamespace(total_score=0.5),
        SimpleNamespace(total_score=0.4),
    ]

    candidates = manager._build_scored_drop_candidates(messages, scores, set(), [(0, [1])])

    assert candidates == [
        {"type": "tool_unit", "indices": [0, 1], "score": 0.25, "position": 0},
        {"type": "turn", "indices": [2, 3], "score": 0.3, "position": 2},
        {"type": "single", "indices": [4], "score": 0.4, "position": 4},
    ]


def test_position_based_scores_marks_protected_and_tool_unit_members() -> None:
    manager = IntelligentContextManager()
    scores = manager._position_based_scores(
        [{"role": "user"}, {"role": "assistant"}, {"role": "tool"}],
        protected={0},
        tool_unit_indices={1},
    )

    assert scores[0].is_protected is True
    assert scores[1].drop_safe is False
    assert scores[2].total_score == 1.0


def test_get_compression_store_and_progressive_summarizer_cache_instances() -> None:
    manager = IntelligentContextManager(
        config=IntelligentContextConfig(summarization_enabled=True),
        summarize_fn=lambda messages, context: "summary",
    )

    summarizer = manager._get_progressive_summarizer()
    store = manager._get_compression_store()

    assert summarizer is manager._get_progressive_summarizer()
    assert store is manager._get_compression_store()


def test_get_progressive_summarizer_and_compression_store_handle_import_error(monkeypatch) -> None:
    manager = IntelligentContextManager()

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
        if name.endswith("progressive_summarizer") or name.endswith("compression_store"):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert manager._get_progressive_summarizer() is None
    assert manager._get_compression_store() is None
