from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import headroom.transforms.content_router as content_router_module
import headroom.transforms.tag_protector as tag_protector_module
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
    route_and_compress,
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
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
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


def test_apply_strategy_to_content_covers_multiple_strategy_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(router, "_record_to_toin", lambda **kwargs: recorded.append(kwargs))

    monkeypatch.setattr(
        router,
        "_try_ml_compressor",
        lambda content, context, question=None: (f"ml:{content}", 1),
    )
    assert router._apply_strategy_to_content(
        "def main(): pass",
        CompressionStrategy.CODE_AWARE,
        "ctx",
        language="python",
    ) == ("ml:def main(): pass", 1)

    monkeypatch.setattr(
        router,
        "_get_smart_crusher",
        lambda: SimpleNamespace(
            crush=lambda content, query, bias: SimpleNamespace(compressed="json")
        ),
    )
    assert router._apply_strategy_to_content(
        '[{"id": 1}]',
        CompressionStrategy.SMART_CRUSHER,
        "ctx",
        bias=0.8,
    ) == ("json", 1)

    monkeypatch.setattr(
        router,
        "_get_search_compressor",
        lambda: SimpleNamespace(
            compress=lambda content, context, bias: SimpleNamespace(compressed="search shrink")
        ),
    )
    assert router._apply_strategy_to_content(
        "src/app.py:10:def main()",
        CompressionStrategy.SEARCH,
        "ctx",
    ) == ("search shrink", 2)

    monkeypatch.setattr(
        router,
        "_get_log_compressor",
        lambda: SimpleNamespace(
            compress=lambda content, bias: SimpleNamespace(
                compressed="log shrink",
                compressed_line_count=3,
            )
        ),
    )
    assert router._apply_strategy_to_content(
        "traceback\nline2",
        CompressionStrategy.LOG,
        "ctx",
    ) == ("log shrink", 3)

    monkeypatch.setattr(
        router,
        "_get_diff_compressor",
        lambda: SimpleNamespace(
            compress=lambda content, context: SimpleNamespace(
                compressed="diff shrink",
                compressed_line_count=4,
            )
        ),
    )
    assert router._apply_strategy_to_content(
        "@@ -1 +1 @@",
        CompressionStrategy.DIFF,
        "ctx",
    ) == ("diff shrink", 4)

    monkeypatch.setattr(
        router,
        "_get_html_extractor",
        lambda: SimpleNamespace(extract=lambda content: SimpleNamespace(extracted="html text")),
    )
    assert router._apply_strategy_to_content(
        "<p>hello</p>",
        CompressionStrategy.HTML,
        "ctx",
    ) == ("html text", 2)

    assert any(call["strategy"] == CompressionStrategy.KOMPRESS for call in recorded)
    assert any(call["strategy"] == CompressionStrategy.SEARCH for call in recorded)
    assert any(call["strategy"] == CompressionStrategy.LOG for call in recorded)
    assert any(call["strategy"] == CompressionStrategy.DIFF for call in recorded)
    assert any(call["strategy"] == CompressionStrategy.HTML for call in recorded)


def test_try_ml_compressor_handles_tag_protection_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    monkeypatch.setattr(
        tag_protector_module,
        "protect_tags",
        lambda content, compress_tagged_content: ("body", {"TAG": "<tool_call>"}),
    )
    monkeypatch.setattr(
        tag_protector_module,
        "restore_tags",
        lambda content, protected: f"{content} {protected['TAG']}",
    )
    monkeypatch.setattr(
        router,
        "_get_kompress",
        lambda: SimpleNamespace(
            compress=lambda text, context, question, target_ratio=None: SimpleNamespace(
                compressed="shrunk",
                compressed_tokens=1,
            )
        ),
    )
    compressed, tokens = router._try_ml_compressor("<tool_call>body</tool_call>", "ctx")
    assert compressed == "shrunk <tool_call>"
    assert tokens == 2

    monkeypatch.setattr(
        tag_protector_module,
        "protect_tags",
        lambda content, compress_tagged_content: ("", {"TAG": "<system-reminder>"}),
    )
    unchanged, tokens = router._try_ml_compressor("<system-reminder>", "ctx")
    assert unchanged == "<system-reminder>"
    assert tokens == 1

    monkeypatch.setattr(
        tag_protector_module,
        "protect_tags",
        lambda content, compress_tagged_content: ("body", {}),
    )
    monkeypatch.setattr(router, "_get_kompress", lambda: None)
    unchanged, tokens = router._try_ml_compressor("plain text body", "ctx")
    assert unchanged == "plain text body"
    assert tokens == 3


def test_router_helper_methods_cover_images_tools_and_analysis_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig(enable_image_optimizer=True))
    messages = [{"role": "user", "content": "hello"}]

    monkeypatch.setattr(router, "_get_image_optimizer", lambda: None)
    assert router.optimize_images_in_messages(messages, tokenizer=SimpleNamespace()) == (
        messages,
        {"images_optimized": 0, "tokens_saved": 0},
    )

    fake_result = SimpleNamespace(
        original_tokens=120,
        compressed_tokens=40,
        technique=SimpleNamespace(value="resize"),
        confidence=0.91,
    )
    fake_optimizer = SimpleNamespace(
        has_images=lambda msgs: True,
        compress=lambda msgs, provider="openai": [{"role": "user", "content": "optimized"}],
        last_result=fake_result,
    )
    monkeypatch.setattr(router, "_get_image_optimizer", lambda: fake_optimizer)
    optimized, metrics = router.optimize_images_in_messages(messages, tokenizer=SimpleNamespace())
    assert optimized == [{"role": "user", "content": "optimized"}]
    assert metrics == {
        "images_optimized": True,
        "tokens_before": 120,
        "tokens_after": 40,
        "tokens_saved": 80,
        "technique": "resize",
        "confidence": 0.91,
    }

    tool_map = router._build_tool_name_map(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "Read"}}],
                "content": [{"type": "tool_use", "id": "call_2", "name": "Glob"}],
            },
            {"role": "user", "content": "ignore me"},
        ]
    )
    assert tool_map == {"call_1": "Read", "call_2": "Glob"}

    assert (
        router._detect_analysis_intent(
            [
                {"role": "assistant", "content": "old"},
                {"role": "user", "content": "Please analyze and explain this bug"},
            ]
        )
        is True
    )
    assert (
        router._detect_analysis_intent(
            [{"role": "user", "content": ["non-string blocks are ignored"]}]
        )
        is False
    )
    assert router.should_apply([], tokenizer=SimpleNamespace()) is True


def test_route_and_compress_uses_router_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ContentRouter,
        "compress",
        lambda self, content, context="": RouterCompressionResult(
            compressed=f"wrapped:{content}:{context}",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
        ),
    )
    assert route_and_compress("payload", context="ctx") == "wrapped:payload:ctx"


def test_eager_load_and_lazy_import_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    detector_module = ModuleType("headroom.compression.detector")
    magika_calls: list[str] = []
    detector_module._magika_available = lambda: True  # type: ignore[attr-defined]
    detector_module._get_magika = lambda: magika_calls.append("magika")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.compression.detector", detector_module)

    parser_calls: list[str] = []
    code_module = ModuleType("headroom.transforms.code_compressor")
    code_module._check_tree_sitter_available = lambda: True  # type: ignore[attr-defined]
    code_module._get_parser = lambda language: parser_calls.append(language)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.code_compressor", code_module)

    class FakeKompressConfig:
        def __init__(self, model_id: str | None = None) -> None:
            self.model_id = model_id

    class FakeKompressCompressor:
        def __init__(self, config: FakeKompressConfig | None = None) -> None:
            self.config = config

    kompress_module = ModuleType("headroom.transforms.kompress_compressor")
    kompress_module.KompressCompressor = FakeKompressCompressor  # type: ignore[attr-defined]
    kompress_module.KompressConfig = FakeKompressConfig  # type: ignore[attr-defined]
    kompress_module.is_kompress_available = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.kompress_compressor", kompress_module)

    class FakeImageCompressor:
        pass

    image_module = ModuleType("headroom.image")
    image_module.ImageCompressor = FakeImageCompressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.image", image_module)

    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    monkeypatch.setattr(router, "_get_kompress", lambda: object())
    monkeypatch.setattr(router, "_get_code_compressor", lambda: object())
    monkeypatch.setattr(router, "_get_smart_crusher", lambda: object())

    status = router.eager_load_compressors()
    assert status["kompress"] == "enabled"
    assert status["magika"] == "enabled"
    assert status["code_aware"] == "enabled"
    assert status["tree_sitter"] == "loaded (8 languages)"
    assert status["smart_crusher"] == "ready"
    assert magika_calls == ["magika"]
    assert parser_calls == [
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "c",
        "cpp",
    ]

    router2 = ContentRouter()
    router2._runtime_kompress_model = "disabled"
    assert router2._get_kompress() is None
    router2._runtime_kompress_model = "custom/model"
    custom = router2._get_kompress()
    assert isinstance(custom, FakeKompressCompressor)
    assert custom.config is not None
    assert custom.config.model_id == "custom/model"
    router2._runtime_kompress_model = None
    default_first = router2._get_kompress()
    default_second = router2._get_kompress()
    assert isinstance(default_first, FakeKompressCompressor)
    assert default_first is default_second

    image_first = router2._get_image_optimizer()
    image_second = router2._get_image_optimizer()
    assert isinstance(image_first, FakeImageCompressor)
    assert image_first is image_second


def test_eager_load_compressors_handles_unavailable_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector_module = ModuleType("headroom.compression.detector")
    detector_module._magika_available = lambda: False  # type: ignore[attr-defined]
    detector_module._get_magika = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.compression.detector", detector_module)

    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    monkeypatch.setattr(router, "_get_kompress", lambda: None)
    monkeypatch.setattr(router, "_get_code_compressor", lambda: None)
    monkeypatch.setattr(router, "_get_smart_crusher", lambda: None)

    status = router.eager_load_compressors()
    assert status["kompress"] == "unavailable"
    assert status["magika"] == "not installed"
    assert status["code_aware"] == "not installed"
    assert "smart_crusher" not in status


def test_router_getter_helpers_toin_and_image_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCodeAwareCompressor:
        pass

    code_module = ModuleType("headroom.transforms.code_compressor")
    code_module.CodeAwareCompressor = FakeCodeAwareCompressor  # type: ignore[attr-defined]
    code_module._check_tree_sitter_available = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.code_compressor", code_module)

    class FakeSmartCrusher:
        def __init__(self, ccr_config) -> None:  # noqa: ANN001
            self.ccr_config = ccr_config

    smart_module = ModuleType("headroom.transforms.smart_crusher")
    smart_module.SmartCrusher = FakeSmartCrusher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.smart_crusher", smart_module)

    class FakeSearchCompressor:
        pass

    search_module = ModuleType("headroom.transforms.search_compressor")
    search_module.SearchCompressor = FakeSearchCompressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.search_compressor", search_module)

    class FakeLogCompressor:
        pass

    log_module = ModuleType("headroom.transforms.log_compressor")
    log_module.LogCompressor = FakeLogCompressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.log_compressor", log_module)

    class FakeDiffCompressor:
        pass

    diff_module = ModuleType("headroom.transforms.diff_compressor")
    diff_module.DiffCompressor = FakeDiffCompressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.diff_compressor", diff_module)

    class FakeHtmlExtractor:
        pass

    html_module = ModuleType("headroom.transforms.html_extractor")
    html_module.HTMLExtractor = FakeHtmlExtractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.html_extractor", html_module)

    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    assert isinstance(router._get_code_compressor(), FakeCodeAwareCompressor)
    assert router._get_code_compressor() is router._code_compressor
    assert isinstance(router._get_smart_crusher(), FakeSmartCrusher)
    assert router._get_smart_crusher() is router._smart_crusher
    assert isinstance(router._get_search_compressor(), FakeSearchCompressor)
    assert isinstance(router._get_log_compressor(), FakeLogCompressor)
    assert isinstance(router._get_diff_compressor(), FakeDiffCompressor)
    assert isinstance(router._get_html_extractor(), FakeHtmlExtractor)

    toin_calls: list[dict[str, object]] = []
    toin_module = ModuleType("headroom.telemetry.toin")
    toin_module.get_toin = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        record_compression=lambda **kwargs: toin_calls.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "headroom.telemetry.toin", toin_module)
    monkeypatch.setattr(content_router_module, "_create_content_signature", lambda **kwargs: "sig")
    router2 = ContentRouter()
    router2._record_to_toin(
        CompressionStrategy.TEXT,
        "original text",
        "short",
        original_tokens=10,
        compressed_tokens=2,
    )
    assert toin_calls[0]["tool_signature"] == "sig"

    router2._toin = SimpleNamespace(
        record_compression=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    router2._record_to_toin(
        CompressionStrategy.TEXT,
        "original text",
        "short",
        original_tokens=10,
        compressed_tokens=2,
    )

    router3 = ContentRouter(ContentRouterConfig(enable_image_optimizer=False))
    messages = [{"role": "user", "content": "hello"}]
    assert router3.optimize_images_in_messages(messages, tokenizer=SimpleNamespace()) == (
        messages,
        {"images_optimized": 0, "tokens_saved": 0},
    )

    router4 = ContentRouter(ContentRouterConfig(enable_image_optimizer=True))
    monkeypatch.setattr(
        router4,
        "_get_image_optimizer",
        lambda: SimpleNamespace(has_images=lambda msgs: False),
    )
    assert router4.optimize_images_in_messages(messages, tokenizer=SimpleNamespace()) == (
        messages,
        {"images_optimized": 0, "tokens_saved": 0},
    )

    fake_optimizer = SimpleNamespace(
        has_images=lambda msgs: True,
        compress=lambda msgs, provider="openai": msgs,
        last_result=None,
    )
    monkeypatch.setattr(router4, "_get_image_optimizer", lambda: fake_optimizer)
    assert router4.optimize_images_in_messages(messages, tokenizer=SimpleNamespace()) == (
        messages,
        {"images_optimized": 0, "tokens_saved": 0},
    )


def test_content_router_code_aware_success_passthrough_and_tool_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(router, "_record_to_toin", lambda **kwargs: recorded.append(kwargs))
    monkeypatch.setattr(
        router,
        "_get_code_compressor",
        lambda: SimpleNamespace(
            compress=lambda content, language, context: SimpleNamespace(
                compressed="code shrink",
                compressed_tokens=2,
            )
        ),
    )

    assert router._apply_strategy_to_content(
        "def main(): pass",
        CompressionStrategy.CODE_AWARE,
        "ctx",
        language="python",
    ) == ("code shrink", 2)
    assert recorded[0]["strategy"] == CompressionStrategy.CODE_AWARE

    monkeypatch.setattr(content_router_module, "split_into_sections", lambda content: [])
    passthrough = router._compress_mixed("mixed content", "ctx")
    assert passthrough.strategy_used is CompressionStrategy.PASSTHROUGH

    user_profile_router = ContentRouter(
        ContentRouterConfig(tool_profiles={"Search": SimpleNamespace(bias=0.55)})
    )
    assert user_profile_router._get_tool_bias("Search") == 0.55
    assert user_profile_router._get_tool_bias("Grep") == 1.5
    assert user_profile_router._get_tool_bias("unknown-tool") == 1.0


def test_process_content_blocks_handles_excluded_pinned_cached_and_small_content() -> None:
    router = ContentRouter()
    recent_read = "recent " * 120
    cached_content = "cached " * 120
    pinned_content = "Retrieve more: hash=abc " + ("pinned " * 100)
    router._cache.put(hash(cached_content), "cached output", 0.4, "text")

    transforms: list[str] = []
    route_counts = {"excluded_tool": 0, "small": 0, "ratio_too_high": 0}
    compressed_details: list[str] = []
    compressor_timing: dict[str, float] = {}
    message = {
        "role": "assistant",
        "content": [
            {"type": "tool_result", "tool_use_id": "read_call", "content": recent_read},
            {"type": "tool_result", "tool_use_id": "pinned_call", "content": pinned_content},
            {"type": "tool_result", "tool_use_id": "cached_call", "content": cached_content},
            {"type": "tool_result", "tool_use_id": "small_call", "content": "tiny"},
            "raw-block",
        ],
    }

    result = router._process_content_blocks(
        message,
        message["content"],
        context="ctx",
        transforms_applied=transforms,
        excluded_tool_ids={"read_call"},
        tool_name_map={"read_call": "Read"},
        route_counts=route_counts,
        compressed_details=compressed_details,
        read_protection_window=4,
        messages_from_end=1,
        compressor_timing=compressor_timing,
    )

    assert result["content"][2]["content"] == "cached output"
    assert "router:excluded:tool" in transforms
    assert "router:tool_result:text" in transforms
    assert compressed_details == ["tool:text:0.40"]
    assert route_counts["excluded_tool"] == 1
    assert route_counts["small"] == 1
    assert route_counts["already_compressed"] == 1
    assert route_counts["cache_hit"] == 1


def test_process_content_blocks_handles_cache_miss_bias_and_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig(tool_profiles={"Search": SimpleNamespace(bias=0.5)}))
    content = "tool output " * 120
    seen_biases: list[float] = []

    def fake_compress(content: str, context: str = "", bias: float = 1.0):  # noqa: ANN001, ANN202
        seen_biases.append(bias)
        return RouterCompressionResult(
            compressed="shrunk output",
            original=content,
            strategy_used=CompressionStrategy.SEARCH,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.SEARCH_RESULTS,
                    strategy=CompressionStrategy.SEARCH,
                    original_tokens=100,
                    compressed_tokens=40,
                )
            ],
        )

    monkeypatch.setattr(router, "compress", fake_compress)

    transforms: list[str] = []
    route_counts = {"excluded_tool": 0, "small": 0, "ratio_too_high": 0}
    compressed_details: list[str] = []
    message = {
        "role": "assistant",
        "content": [{"type": "tool_result", "tool_use_id": "search_call", "content": content}],
    }

    compressed_message = router._process_content_blocks(
        message,
        message["content"],
        context="ctx",
        transforms_applied=transforms,
        excluded_tool_ids=set(),
        tool_name_map={"search_call": "Search"},
        route_counts=route_counts,
        compressed_details=compressed_details,
        min_ratio=0.85,
    )

    assert compressed_message["content"][0]["content"] == "shrunk output"
    assert seen_biases == [0.5]
    assert "router:tool_result:search" in transforms
    assert compressed_details == ["tool:search:0.40"]
    assert route_counts["cache_miss"] == 1

    tightened_counts = {"excluded_tool": 0, "small": 0, "ratio_too_high": 0}
    unchanged_message = router._process_content_blocks(
        message,
        message["content"],
        context="ctx",
        transforms_applied=[],
        excluded_tool_ids=set(),
        tool_name_map={"search_call": "Search"},
        route_counts=tightened_counts,
        compressed_details=[],
        min_ratio=0.3,
    )
    assert unchanged_message == message
    assert tightened_counts["ratio_too_high"] == 1
    assert router._cache.is_skipped(hash(content)) is True


def test_content_router_apply_covers_protection_and_compression_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    router.config.read_lifecycle.enabled = False

    def count_text(text: str) -> int:
        return len(str(text).split())

    tokenizer = SimpleNamespace(count_text=count_text)
    large_text = "compress " * 120
    code_text = "def fn():\n    pass\n" + ("token " * 80)

    monkeypatch.setattr(router, "_detect_analysis_intent", lambda messages: True)
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(
            content_type=ContentType.SOURCE_CODE
            if str(content).startswith("def fn")
            else ContentType.PLAIN_TEXT,
            confidence=1.0,
            metadata={},
        ),
    )
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, context="", bias=1.0: RouterCompressionResult(
            compressed="compressed payload",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.TEXT,
                    original_tokens=100,
                    compressed_tokens=40,
                )
            ],
        ),
    )

    messages = [
        {"role": "user", "content": "user " * 60},
        {"role": "system", "content": "system " * 60},
        {"role": "assistant", "content": {"nested": True}},
        {"role": "assistant", "content": "tiny"},
        {"role": "assistant", "content": code_text},
        {"role": "assistant", "content": "Retrieve more: hash=abc " + ("pinned " * 80)},
        {"role": "assistant", "content": large_text},
    ]

    result = router.apply(
        messages,
        tokenizer=tokenizer,
        context="ctx",
        compress_system_messages=False,
    )

    assert result.messages[0] == messages[0]
    assert result.messages[1] == messages[1]
    assert result.messages[2] == messages[2]
    assert result.messages[3] == messages[3]
    assert result.messages[4] == messages[4]
    assert result.messages[5] == messages[5]
    assert result.messages[6]["content"] == "compressed payload"
    assert "router:protected:user_message" in result.transforms_applied
    assert "router:protected:system_message" in result.transforms_applied
    assert "router:protected:recent_code" in result.transforms_applied
    assert any(transform.startswith("router:text:0.40") for transform in result.transforms_applied)


def test_content_router_apply_handles_read_lifecycle_and_content_block_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_module = ModuleType("headroom.transforms.read_lifecycle")

    class FakeReadLifecycleManager:
        def __init__(self, config, compression_store=None) -> None:  # noqa: ANN001
            self.config = config
            self.compression_store = compression_store

        def apply(self, messages, frozen_message_count=0):  # noqa: ANN001, ANN201
            assert frozen_message_count == 1
            return SimpleNamespace(
                messages=messages,
                transforms_applied=["lifecycle:applied"],
                ccr_hashes=["hash-1"],
            )

    lifecycle_module.ReadLifecycleManager = FakeReadLifecycleManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.transforms.read_lifecycle", lifecycle_module)

    router = ContentRouter()
    router.config.read_lifecycle.enabled = True
    tokenizer = SimpleNamespace(count_text=lambda text: len(str(text).split()))
    first_message = {"role": "assistant", "content": "frozen " * 60}
    block_message = {
        "role": "assistant",
        "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "tool " * 600}],
    }

    monkeypatch.setattr(router, "_build_tool_name_map", lambda messages: {"tool-1": "Search"})
    monkeypatch.setattr(
        router,
        "_process_content_blocks",
        lambda message, content_blocks, context, transforms_applied, excluded_tool_ids, **kwargs: {
            **message,
            "content": "content-blocks-processed",
        },
    )

    result = router.apply(
        [first_message, block_message],
        tokenizer=tokenizer,
        compression_store="store",
        frozen_message_count=1,
        context="ctx",
        model_limit=1000,
    )

    assert result.messages == [
        first_message,
        {"role": "assistant", "content": "content-blocks-processed"},
    ]
    assert result.transforms_applied == ["lifecycle:applied"]
    assert result.markers_inserted == ["hash-1"]


def test_get_magika_detector_caches_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    content_router_module._magika_detector = None
    content_router_module._magika_status = None
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
        if name.endswith("compression.detector"):
            raise ImportError("missing detector")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert content_router_module._get_magika_detector() is None
    assert content_router_module._magika_status is False
    assert content_router_module._get_magika_detector() is None


def test_content_router_apply_compresses_old_excluded_tool_outputs_with_combined_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(
        ContentRouterConfig(
            exclude_tools={"Read"},
            tool_profiles={"Read": SimpleNamespace(bias=0.6)},
            protect_recent_reads_fraction=0.1,
        )
    )
    router.config.read_lifecycle.enabled = False
    tokenizer = SimpleNamespace(count_text=lambda text: len(str(text).split()))
    seen_biases: list[float] = []

    monkeypatch.setattr(router, "_build_tool_name_map", lambda messages: {"read_call": "Read"})
    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda content: DetectionResult(ContentType.PLAIN_TEXT, 1.0, {}),
    )

    def fake_compress(content: str, context: str = "", bias: float = 1.0):  # noqa: ANN001, ANN202
        seen_biases.append(bias)
        return RouterCompressionResult(
            compressed="old tool compressed",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.TEXT,
                    original_tokens=100,
                    compressed_tokens=20,
                )
            ],
        )

    monkeypatch.setattr(router, "compress", fake_compress)

    messages = [
        {"role": "tool", "tool_call_id": "read_call", "content": "tool output " * 100},
        {"role": "assistant", "content": "recent 1"},
        {"role": "user", "content": "recent 2"},
        {"role": "assistant", "content": "recent 3"},
        {"role": "user", "content": "recent 4"},
    ]

    result = router.apply(messages, tokenizer=tokenizer, context="ctx", biases={0: 0.5})

    assert result.messages[0]["content"] == "old tool compressed"
    assert seen_biases == [0.3]
    assert any(transform.startswith("router:text:0.20") for transform in result.transforms_applied)


def test_content_router_apply_moves_tightened_cached_result_to_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter()
    router.config.read_lifecycle.enabled = False
    tokenizer = SimpleNamespace(count_text=lambda text: len(str(text).split()))
    content = "cached candidate " * 80
    router._cache.put(hash(content), "cached output", 0.9, "text")

    monkeypatch.setattr(
        content_router_module,
        "_detect_content",
        lambda value: DetectionResult(ContentType.PLAIN_TEXT, 1.0, {}),
    )
    monkeypatch.setattr(
        router,
        "compress",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache branch should short-circuit")
        ),
    )

    result = router.apply(
        [{"role": "assistant", "content": content}], tokenizer=tokenizer, model_limit=1000
    )

    assert result.messages == [{"role": "assistant", "content": content}]
    assert router._cache.is_skipped(hash(content)) is True
