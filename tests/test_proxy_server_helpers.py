from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import httpx

import headroom.proxy.server as server_module
from headroom.proxy.server import HeadroomProxy, ProxyConfig, _get_code_aware_banner_status
from headroom.transforms import ContentRouter


def _make_proxy() -> HeadroomProxy:
    return HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )


def test_compression_cache_reuses_entries_and_evicts_oldest(monkeypatch) -> None:
    proxy = _make_proxy()
    monkeypatch.setattr(server_module, "MAX_COMPRESSION_CACHE_SESSIONS", 4)

    cache_a = proxy._get_compression_cache("a")
    assert proxy._get_compression_cache("a") is cache_a

    proxy._compression_caches = {"a": object(), "b": object(), "c": object(), "d": object()}
    new_cache = proxy._get_compression_cache("e")
    assert "a" not in proxy._compression_caches
    assert "e" in proxy._compression_caches
    assert new_cache is proxy._compression_caches["e"]


def test_setup_code_aware_extract_tags_and_system_context(monkeypatch) -> None:
    proxy = _make_proxy()

    class FakeCodeCompressorConfig:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class FakeCodeAwareCompressor:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config

    monkeypatch.setattr(server_module, "is_tree_sitter_available", lambda: True)
    monkeypatch.setattr(server_module, "CodeCompressorConfig", FakeCodeCompressorConfig)
    monkeypatch.setattr(server_module, "CodeAwareCompressor", FakeCodeAwareCompressor)

    transforms = ["sentinel"]
    status = proxy._setup_code_aware(ProxyConfig(code_aware_enabled=True), transforms)
    assert status == "enabled"
    assert isinstance(transforms[0], FakeCodeAwareCompressor)
    assert transforms[1] == "sentinel"

    unavailable_status = proxy._setup_code_aware(ProxyConfig(code_aware_enabled=False), ["x"])
    assert unavailable_status == "available"

    monkeypatch.setattr(server_module, "is_tree_sitter_available", lambda: False)
    unavailable_status = proxy._setup_code_aware(ProxyConfig(code_aware_enabled=True), ["x"])
    assert unavailable_status == "unavailable"

    tags = proxy._extract_tags(
        {
            "X-Headroom-User": "alice",
            "x-headroom-session": "abc",
            "Authorization": "Bearer secret",
        }
    )
    assert tags == {"user": "alice", "session": "abc"}

    messages = [{"role": "system", "content": "existing"}, {"role": "user", "content": "hi"}]
    updated = proxy._inject_system_context(messages, "ctx")
    assert updated[0]["content"] == "existing\n\nctx"

    prepended = proxy._inject_system_context([{"role": "user", "content": "hi"}], "ctx")
    assert prepended[0] == {"role": "system", "content": "ctx"}

    body = {"system": "existing"}
    returned = proxy._inject_system_context(messages, "ctx", body=body)
    assert returned == messages
    assert body["system"] == "existing\n\nctx"

    list_body = {"system": [{"type": "text", "text": "existing"}]}
    proxy._inject_system_context(messages, "ctx", body=list_body)
    assert list_body["system"][-1] == {"type": "text", "text": "ctx"}

    empty_body: dict[str, object] = {}
    proxy._inject_system_context(messages, "ctx", body=empty_body)
    assert empty_body["system"] == "ctx"


def test_code_aware_banner_status_and_summary_logging(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "is_tree_sitter_available", lambda: True)
    assert (
        _get_code_aware_banner_status(ProxyConfig(code_aware_enabled=True))
        == "ENABLED  (AST-based)"
    )
    assert (
        _get_code_aware_banner_status(ProxyConfig(code_aware_enabled=False))
        == "DISABLED (remove --no-code-aware to enable)"
    )

    monkeypatch.setattr(server_module, "is_tree_sitter_available", lambda: False)
    assert (
        _get_code_aware_banner_status(ProxyConfig(code_aware_enabled=True))
        == "NOT INSTALLED (pip install headroom-ai[code])"
    )
    assert _get_code_aware_banner_status(ProxyConfig(code_aware_enabled=False)) == "DISABLED"

    proxy = _make_proxy()
    proxy.metrics = SimpleNamespace(
        requests_total=5,
        requests_cached=2,
        requests_rate_limited=1,
        requests_failed=0,
        tokens_input_total=200,
        tokens_output_total=50,
        tokens_saved_total=100,
        latency_count=2,
        latency_sum_ms=40.0,
    )
    logged: list[str] = []
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: logged.append(message % args if args else message),
    )
    proxy._print_summary()
    assert any("HEADROOM PROXY SESSION SUMMARY" in line for line in logged)
    assert any("Total requests:" in line for line in logged)
    assert any("Token savings:" in line for line in logged)
    assert any("Avg latency:" in line for line in logged)


def test_retry_request_handles_streaming_client_errors_and_retries(monkeypatch) -> None:
    proxy = _make_proxy()
    proxy.config.retry_enabled = True
    proxy.config.retry_max_attempts = 3
    proxy.config.retry_base_delay_ms = 10
    proxy.config.retry_max_delay_ms = 50

    request = httpx.Request("POST", "https://example.test")
    server_error = httpx.Response(503, request=request)
    success = httpx.Response(200, request=request)
    client_error = httpx.Response(429, request=request)

    calls: list[str] = []
    responses = iter([server_error, success])

    async def post(url: str, json=None, headers=None):  # noqa: ANN001, ANN201
        calls.append(url)
        return next(responses)

    proxy.http_client = SimpleNamespace(post=post)
    sleep_calls: list[float] = []
    monkeypatch.setattr(server_module, "jitter_delay_ms", lambda base, max_ms, attempt: 25.0)

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(server_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(proxy._retry_request("POST", "https://example.test", {}, {}))
    assert result.status_code == 200
    assert calls == ["https://example.test", "https://example.test"]
    assert sleep_calls == [0.025]

    async def post_4xx(url: str, json=None, headers=None):  # noqa: ANN001, ANN201
        return client_error

    proxy.http_client = SimpleNamespace(post=post_4xx)
    result = asyncio.run(proxy._retry_request("POST", "https://example.test", {}, {}))
    assert result.status_code == 429

    async def post_stream(url: str, json=None, headers=None):  # noqa: ANN001, ANN201
        return success

    proxy.http_client = SimpleNamespace(post=post_stream)
    result = asyncio.run(proxy._retry_request("POST", "https://example.test", {}, {}, stream=True))
    assert result.status_code == 200

    proxy.config.retry_enabled = False

    async def post_fails(url: str, json=None, headers=None):  # noqa: ANN001, ANN201
        raise httpx.ConnectError("boom")

    proxy.http_client = SimpleNamespace(post=post_fails)
    try:
        asyncio.run(proxy._retry_request("POST", "https://example.test", {}, {}))
    except httpx.ConnectError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected retry-disabled connect error to raise")


def test_log_toin_stats_periodically_logs_stats_and_failures(monkeypatch) -> None:
    sleep_calls: list[int] = []
    info_logs: list[tuple[object, ...]] = []
    debug_logs: list[tuple[object, ...]] = []

    async def fake_sleep(seconds: int) -> None:
        sleep_calls.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(server_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: info_logs.append((message, *args)),
    )
    monkeypatch.setattr(
        server_module.logger,
        "debug",
        lambda message, *args: debug_logs.append((message, *args)),
    )

    class _Toin:
        def get_stats(self) -> dict[str, float]:
            return {
                "total_compressions": 3,
                "patterns_tracked": 4,
                "total_retrievals": 5,
                "global_retrieval_rate": 0.5,
            }

    monkeypatch.setattr(server_module, "get_toin", lambda: _Toin())
    try:
        asyncio.run(server_module._log_toin_stats_periodically(interval_seconds=7))
    except asyncio.CancelledError:
        pass

    assert sleep_calls == [7]
    assert info_logs == []
    assert debug_logs == []

    state = {"sleeps": 0}

    async def fake_sleep_then_cancel(seconds: int) -> None:
        state["sleeps"] += 1
        if state["sleeps"] == 1:
            return
        raise asyncio.CancelledError

    monkeypatch.setattr(server_module.asyncio, "sleep", fake_sleep_then_cancel)
    info_logs.clear()
    debug_logs.clear()
    monkeypatch.setattr(server_module, "get_toin", lambda: _Toin())

    try:
        asyncio.run(server_module._log_toin_stats_periodically(interval_seconds=3))
    except asyncio.CancelledError:
        pass

    assert info_logs == [
        ("TOIN: %d patterns, %d compressions, %d retrievals, %.1f%% retrieval rate", 4, 3, 5, 50.0)
    ]

    state["sleeps"] = 0

    def raise_toin():  # noqa: ANN202
        raise RuntimeError("boom")

    monkeypatch.setattr(server_module, "get_toin", raise_toin)
    debug_logs.clear()

    try:
        asyncio.run(server_module._log_toin_stats_periodically(interval_seconds=2))
    except asyncio.CancelledError:
        pass

    assert debug_logs[0][0] == "Failed to log TOIN stats: %s"
    assert str(debug_logs[0][1]) == "boom"


def test_register_memory_components_is_idempotent_and_skips_batch_store_without_stats(
    monkeypatch,
) -> None:
    proxy = _make_proxy()
    proxy.cache = SimpleNamespace(get_memory_stats=lambda: {"kind": "cache"})
    proxy.logger = SimpleNamespace(get_memory_stats=lambda: {"kind": "logger"})

    compression_store = SimpleNamespace(get_memory_stats=lambda: {"kind": "compression"})
    batch_store = SimpleNamespace(get_memory_stats=lambda: {"kind": "batch"})
    fake_batch_module = types.ModuleType("headroom.ccr.batch_store")
    fake_batch_module.get_batch_context_store = lambda: batch_store

    tracker = SimpleNamespace(registered_components=set(), calls=[])

    def register(name: str, callback) -> None:  # noqa: ANN001
        tracker.registered_components.add(name)
        tracker.calls.append((name, callback()))

    tracker.register = register

    monkeypatch.setattr(server_module, "get_compression_store", lambda: compression_store)
    monkeypatch.setitem(sys.modules, "headroom.ccr.batch_store", fake_batch_module)

    server_module._register_memory_components(proxy, tracker)
    server_module._register_memory_components(proxy, tracker)

    assert tracker.calls == [
        ("compression_store", {"kind": "compression"}),
        ("semantic_cache", {"kind": "cache"}),
        ("request_logger", {"kind": "logger"}),
        ("batch_context_store", {"kind": "batch"}),
    ]

    tracker.calls.clear()
    tracker.registered_components.clear()
    no_stats_batch_module = types.ModuleType("headroom.ccr.batch_store")
    no_stats_batch_module.get_batch_context_store = lambda: object()
    monkeypatch.setitem(sys.modules, "headroom.ccr.batch_store", no_stats_batch_module)
    server_module._register_memory_components(proxy, tracker)

    assert tracker.calls == [
        ("compression_store", {"kind": "compression"}),
        ("semantic_cache", {"kind": "cache"}),
        ("request_logger", {"kind": "logger"}),
    ]


def test_create_app_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "FASTAPI_AVAILABLE", False)

    try:
        server_module.create_app()
    except ImportError as exc:
        assert "FastAPI required" in str(exc)
    else:
        raise AssertionError("expected create_app to require FastAPI when unavailable")


def test_proxy_init_token_mode_smart_routing_adjusts_router_config() -> None:
    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            smart_routing=True,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            code_aware_enabled=False,
        )
    )

    router = next(t for t in proxy.anthropic_pipeline.transforms if isinstance(t, ContentRouter))
    assert router.config.protect_recent_reads_fraction == 0.3
    assert proxy._code_aware_status == "disabled"


def test_proxy_init_wires_optional_components_and_project_memory_db(tmp_path, monkeypatch) -> None:
    created = {}

    class FakeMemoryHandler:
        def __init__(self, memory_config, agent_type):  # noqa: ANN001
            created["memory"] = (memory_config, agent_type)

    reporter_module = types.ModuleType("headroom.telemetry.reporter")

    class FakeUsageReporter:
        def __init__(self, license_key, cloud_url, report_interval):  # noqa: ANN001
            created["reporter"] = (license_key, cloud_url, report_interval)

    reporter_module.UsageReporter = FakeUsageReporter

    learner_module = types.ModuleType("headroom.memory.traffic_learner")

    class FakeTrafficLearner:
        def __init__(self, user_id, agent_type):  # noqa: ANN001
            created["learner"] = (user_id, agent_type)

    learner_module.TrafficLearner = FakeTrafficLearner

    graph_module = types.ModuleType("headroom.graph.watcher")

    class FakeCodeGraphWatcher:
        def __init__(self, project_dir):  # noqa: ANN001
            self.project_dir = project_dir
            created["watcher_dir"] = project_dir

        def start(self) -> bool:
            created["watcher_started"] = True
            return True

    graph_module.CodeGraphWatcher = FakeCodeGraphWatcher

    monkeypatch.setattr(server_module, "MemoryHandler", FakeMemoryHandler)
    monkeypatch.setattr(server_module.Path, "cwd", lambda: tmp_path)
    monkeypatch.setenv("HEADROOM_USER_ID", "user-123")
    monkeypatch.setitem(sys.modules, "headroom.telemetry.reporter", reporter_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.traffic_learner", learner_module)
    monkeypatch.setitem(sys.modules, "headroom.graph.watcher", graph_module)

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            memory_enabled=True,
            memory_db_path="",
            license_key="license-abc",
            license_cloud_url="https://licenses.example.test",
            license_report_interval=123,
            traffic_learning_enabled=True,
            traffic_learning_agent_type="codex",
            code_graph_watcher=True,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    memory_config, agent_type = created["memory"]
    assert memory_config.db_path == str(tmp_path / ".headroom" / "memory.db")
    assert agent_type == "codex"
    assert created["reporter"] == ("license-abc", "https://licenses.example.test", 123)
    assert created["learner"] == ("user-123", "codex")
    assert created["watcher_started"] is True
    assert proxy.code_graph_watcher is not None


def test_proxy_init_clears_code_graph_watcher_when_start_fails(monkeypatch, tmp_path) -> None:
    graph_module = types.ModuleType("headroom.graph.watcher")

    class FakeCodeGraphWatcher:
        def __init__(self, project_dir):  # noqa: ANN001
            self.project_dir = project_dir

        def start(self) -> bool:
            return False

    graph_module.CodeGraphWatcher = FakeCodeGraphWatcher

    monkeypatch.setitem(sys.modules, "headroom.graph.watcher", graph_module)
    monkeypatch.setattr(server_module.Path, "cwd", lambda: tmp_path)

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_graph_watcher=True,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    assert proxy.code_graph_watcher is None


def test_inject_system_context_leaves_non_string_system_message_unchanged() -> None:
    proxy = _make_proxy()
    messages = [{"role": "system", "content": [{"type": "text", "text": "structured"}]}]

    updated = proxy._inject_system_context(messages, "ctx")

    assert updated == messages


def test_next_request_id_increments_monotonically() -> None:
    proxy = _make_proxy()

    first = asyncio.run(proxy._next_request_id())
    second = asyncio.run(proxy._next_request_id())

    assert first.endswith("_000001")
    assert second.endswith("_000002")


def test_shutdown_closes_resources_and_stops_registry(monkeypatch) -> None:
    proxy = _make_proxy()
    events: list[str] = []

    class _Client:
        async def aclose(self) -> None:
            events.append("http_client")

    class _MemoryHandler:
        async def close(self) -> None:
            events.append("memory_handler")

    class _Registry:
        async def stop_all(self) -> None:
            events.append("registry")

    proxy.http_client = _Client()
    proxy.memory_handler = _MemoryHandler()
    proxy._print_summary = lambda: events.append("summary")
    monkeypatch.setattr(server_module, "get_quota_registry", lambda: _Registry())

    asyncio.run(proxy.shutdown())

    assert events == ["http_client", "memory_handler", "registry", "summary"]
    assert proxy.http_client is None
