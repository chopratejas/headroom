from __future__ import annotations

from types import SimpleNamespace

import pytest

import headroom.transforms.content_router as content_router_module
from headroom.transforms.content_detector import ContentType, DetectionResult
from headroom.transforms.content_router import (
    CompressionCache,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
    _create_content_signature,
    _detect_content,
    _extract_json_block,
    is_mixed_content,
    split_into_sections,
)


def test_compression_cache_handles_hits_skips_evictions_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 112.0, 112.0])
    monkeypatch.setattr(content_router_module.time, "time", lambda: next(times))
    monkeypatch.setattr(content_router_module.time, "perf_counter_ns", lambda: 50)

    cache = CompressionCache(ttl_seconds=10)
    cache.put(1, "compressed", 0.4, "text")
    cache.mark_skip(2)

    assert cache.get(1) == ("compressed", 0.4, "text")
    assert cache.is_skipped(2) is True
    assert cache.size == 1
    assert cache.skip_size == 1

    cache.move_to_skip(1)
    assert cache.get(1) is None
    assert cache.is_skipped(1) is True

    # Expire both skip entries
    assert cache.is_skipped(2) is False
    assert cache.is_skipped(1) is False

    assert cache.stats["cache_hits"] == 1
    assert cache.stats["cache_skip_hits"] == 2
    assert cache.stats["cache_misses"] == 1
    assert cache.stats["cache_evictions"] >= 2

    cache.clear()
    assert cache.size == 0
    assert cache.skip_size == 0


def test_router_result_helpers_and_summary() -> None:
    pure = RouterCompressionResult(
        compressed="small",
        original="very large",
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=4,
            )
        ],
    )
    assert pure.total_original_tokens == 10
    assert pure.total_compressed_tokens == 4
    assert pure.compression_ratio == 0.4
    assert pure.tokens_saved == 6
    assert pure.savings_percentage == 60.0
    assert pure.summary() == "Pure text: 10→4 tokens (60% saved)"

    mixed = RouterCompressionResult(
        compressed="joined",
        original="original",
        strategy_used=CompressionStrategy.MIXED,
        sections_processed=2,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=0,
                compressed_tokens=0,
            ),
            RoutingDecision(
                content_type=ContentType.SEARCH_RESULTS,
                strategy=CompressionStrategy.SEARCH,
                original_tokens=8,
                compressed_tokens=2,
            ),
        ],
    )
    assert mixed.routing_log[0].compression_ratio == 1.0
    assert mixed.summary().startswith("Mixed content: 2 sections, routed to ")


def test_content_signature_and_detection_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage-3d (PR5) wired `_detect_content` through the Rust chain
    (`headroom._core.detect_content_type` → magika → unidiff →
    PlainText). The pre-PR5 Python-side `_get_magika_detector`
    fallback path is gone.

    This test asserts the new contract:
    1. The detection helper delegates to the Rust binding.
    2. Whatever `ContentType` the Rust side returns flows back as a
       Python `DetectionResult` with that same `content_type`.
    """
    signature = _create_content_signature("search", "file.py:10:match", language="python")
    assert signature is not None
    assert len(signature.structure_hash) == 24

    # Monkeypatch the Rust binding to return a deterministic fake
    # result; verify _detect_content propagates the content_type
    # tag back as the Python ContentType enum.
    import headroom._core as _core

    fake_rust_result = SimpleNamespace(
        content_type="source_code",
        confidence=1.0,
        metadata={},
    )
    monkeypatch.setattr(_core, "detect_content_type", lambda content: fake_rust_result)

    result = _detect_content("def main(): pass")
    assert result.content_type is ContentType.SOURCE_CODE
    assert result.confidence == 1.0
    assert result.metadata == {}


def test_mixed_content_section_splitting_and_json_extraction() -> None:
    content = "\n".join(
        [
            "Intro paragraph with Several words included for prose detection.",
            "Another line with enough words to read as normal prose today.",
            "Third line adds more prose so the detector sees real text content.",
            "Fourth sentence keeps the count moving higher for prose patterns.",
            "Fifth sentence does the same for mixed content identification.",
            "Sixth sentence seals the prose threshold for the helper.",
            "```python",
            "def main():",
            "    return 1",
            "```",
            '[{"id": 1}]',
            "src/app.py:10:def main():",
            "src/app.py:11:return 1",
        ]
    )
    assert is_mixed_content(content) is True

    sections = split_into_sections(content)
    assert [section.content_type for section in sections] == [
        ContentType.PLAIN_TEXT,
        ContentType.SOURCE_CODE,
        ContentType.JSON_ARRAY,
        ContentType.SEARCH_RESULTS,
    ]
    assert sections[1].language == "python"
    assert sections[1].is_code_fence is True
    assert sections[2].content == '[{"id": 1}]'
    assert sections[3].end_line == 12

    json_block, end_idx = _extract_json_block(["[", '{"id": 1}', "]"], 0)
    assert json_block == '[\n{"id": 1}\n]'
    assert end_idx == 2
    assert _extract_json_block(["{", '"a": 1'], 0) == (None, 0)


def test_content_router_strategy_and_compress_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))

    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.SOURCE_CODE, 1.0, {}),
    )
    assert router._determine_strategy("code") is CompressionStrategy.KOMPRESS
    assert (
        router._strategy_from_detection(DetectionResult(ContentType.SEARCH_RESULTS, 1.0, {}))
        is CompressionStrategy.SEARCH
    )
    assert router._strategy_from_detection_type(ContentType.GIT_DIFF) is CompressionStrategy.DIFF
    assert (
        router._content_type_from_strategy(CompressionStrategy.PASSTHROUGH)
        is ContentType.PLAIN_TEXT
    )

    mixed_result = RouterCompressionResult(
        compressed="mixed",
        original="mixed",
        strategy_used=CompressionStrategy.MIXED,
    )
    pure_result = RouterCompressionResult(
        compressed="pure",
        original="pure",
        strategy_used=CompressionStrategy.TEXT,
    )
    monkeypatch.setattr(router, "_compress_mixed", lambda *args, **kwargs: mixed_result)
    monkeypatch.setattr(router, "_compress_pure", lambda *args, **kwargs: pure_result)

    monkeypatch.setattr(router, "_determine_strategy", lambda content: CompressionStrategy.MIXED)
    assert router.compress("mixed") is mixed_result

    monkeypatch.setattr(router, "_determine_strategy", lambda content: CompressionStrategy.TEXT)
    assert router.compress("pure") is pure_result
    assert router.compress("   ").strategy_used is CompressionStrategy.PASSTHROUGH


def test_content_router_mixed_pure_apply_and_toin(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ContentRouter()
    mixed_content = "\n".join(["before", "```python", "print('x')", "```", "after"])
    monkeypatch.setattr(
        content_router_module,
        "split_into_sections",
        lambda content: [
            SimpleNamespace(
                content="print('x')",
                content_type=ContentType.SOURCE_CODE,
                language="python",
                is_code_fence=True,
            ),
            SimpleNamespace(
                content="after text",
                content_type=ContentType.PLAIN_TEXT,
                language=None,
                is_code_fence=False,
            ),
        ],
    )
    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda content, strategy, context, language=None, question=None, bias=1.0: (
            f"{strategy.value}:{content}",
            len(content.split()) - 1,
        ),
    )
    result = router._compress_mixed(mixed_content, "ctx")
    assert result.strategy_used is CompressionStrategy.MIXED
    assert result.sections_processed == 2
    assert "```python\ncode_aware:print('x')\n```" in result.compressed

    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda content, strategy, context, language=None, question=None, bias=1.0: (
            "shrunk",
            1,
        ),
    )
    pure = router._compress_pure("some plain text", CompressionStrategy.TEXT, "ctx")
    assert pure.routing_log[0].content_type is ContentType.PLAIN_TEXT
    assert pure.total_original_tokens == 3
    assert pure.total_compressed_tokens == 1

    calls: list[dict] = []
    router._toin = SimpleNamespace(record_compression=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(content_router_module, "_create_content_signature", lambda **kwargs: "sig")
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "original content",
        "small",
        original_tokens=10,
        compressed_tokens=4,
        language="python",
        context="question",
    )
    assert calls[0]["tool_signature"] == "sig"
    assert calls[0]["strategy"] == "text"
    assert calls[0]["query_context"] == "question"

    router._record_to_toin(
        CompressionStrategy.SMART_CRUSHER,
        "x",
        "x",
        original_tokens=10,
        compressed_tokens=4,
    )
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "x",
        "x",
        original_tokens=2,
        compressed_tokens=2,
    )
    monkeypatch.setattr(content_router_module, "_create_content_signature", lambda **kwargs: None)
    router._record_to_toin(
        CompressionStrategy.TEXT,
        "x",
        "y",
        original_tokens=5,
        compressed_tokens=1,
    )
    assert len(calls) == 1
