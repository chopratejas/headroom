"""Tests for the shared cold-start warmup registry (Unit 1).

Covers:
- WarmupRegistry slot state transitions.
- Preload iterates both Anthropic + OpenAI pipelines and dedupes
  shared transforms by ``id(transform)``.
- Embedder warm-up encode runs once during startup (happy path).
- optimize=False leaves slots null.
- Memory backend init failure yields registry status=error, startup
  still completes, health reports degraded memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

import headroom.proxy.server as server_module
from headroom.proxy.warmup import WarmupRegistry, WarmupSlot

# -------------------------------------------------------------------
# WarmupSlot / WarmupRegistry unit tests
# -------------------------------------------------------------------


def test_warmup_slot_defaults_to_null():
    slot = WarmupSlot()
    assert slot.status == "null"
    assert slot.handle is None
    assert slot.error is None
    assert slot.to_dict() == {"status": "null"}


def test_warmup_slot_transitions():
    slot = WarmupSlot()
    slot.mark_loading()
    assert slot.status == "loading"

    slot.mark_loaded(handle="h", model="x")
    assert slot.status == "loaded"
    assert slot.handle == "h"
    assert slot.info == {"model": "x"}
    assert slot.to_dict() == {"status": "loaded", "info": {"model": "x"}}

    slot.mark_error("boom")
    assert slot.status == "error"
    assert slot.handle is None
    assert slot.error == "boom"
    assert slot.to_dict()["status"] == "error"
    assert slot.to_dict()["error"] == "boom"

    slot.mark_null()
    assert slot.status == "null"
    assert slot.handle is None
    assert slot.error is None


def test_warmup_registry_merges_enabled_status_into_loaded():
    reg = WarmupRegistry()
    reg.merge_transform_status(
        {
            "kompress": "enabled",
            "magika": "enabled",
            "code_aware": "enabled",
            "tree_sitter": "loaded (3 languages)",
            "smart_crusher": "ready",
        }
    )
    out = reg.to_dict()
    assert out["kompress"]["status"] == "loaded"
    assert out["magika"]["status"] == "loaded"
    assert out["code_aware"]["status"] == "loaded"
    assert out["tree_sitter"]["status"] == "loaded"
    assert out["smart_crusher"]["status"] == "loaded"


def test_warmup_registry_preserves_null_for_unavailable():
    reg = WarmupRegistry()
    reg.merge_transform_status({"kompress": "unavailable", "magika": "not installed"})
    assert reg.kompress.status == "null"
    assert reg.magika.status == "null"
    assert reg.kompress.info.get("source_status") == "unavailable"


def test_warmup_registry_to_dict_has_expected_keys():
    reg = WarmupRegistry()
    out = reg.to_dict()
    assert set(out.keys()) == {
        "kompress",
        "magika",
        "code_aware",
        "tree_sitter",
        "smart_crusher",
        "memory_backend",
        "memory_embedder",
    }


# -------------------------------------------------------------------
# Startup orchestration tests — use HeadroomProxy + stubbed transforms
# -------------------------------------------------------------------


@pytest.fixture
def _stub_pipelines(monkeypatch):
    """Build a minimal proxy whose pipelines share a single spy transform.

    The transform's ``eager_load_compressors`` increments a hit counter;
    assertion that dedup prevents double-loading relies on that counter.
    """
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    class SpyTransform:
        def __init__(self) -> None:
            self.hits = 0

        def eager_load_compressors(self) -> dict[str, str]:
            self.hits += 1
            return {
                "kompress": "enabled",
                "magika": "enabled",
                "code_aware": "enabled",
                "smart_crusher": "ready",
            }

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        smart_routing=False,
        code_aware_enabled=False,
    )
    proxy = HeadroomProxy(config)

    spy = SpyTransform()
    # Replace both pipeline transform lists with the SAME instance so the
    # dedupe-by-id() logic is what we actually assert against.
    proxy.anthropic_pipeline.transforms = [spy]
    proxy.openai_pipeline.transforms = [spy]
    return proxy, spy


@pytest.mark.asyncio
async def test_startup_runs_shared_transform_once(_stub_pipelines):
    proxy, spy = _stub_pipelines
    await proxy.startup()
    try:
        assert spy.hits == 1, "shared transform must be eager-loaded exactly once"
        assert proxy.warmup.kompress.status == "loaded"
        assert proxy.warmup.magika.status == "loaded"
        assert proxy.warmup.code_aware.status == "loaded"
        assert proxy.warmup.smart_crusher.status == "loaded"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_optimize_false_leaves_slots_null():
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        smart_routing=False,
    )
    proxy = HeadroomProxy(config)

    called = {"n": 0}

    class SpyTransform:
        def eager_load_compressors(self) -> dict[str, str]:
            called["n"] += 1
            return {"kompress": "enabled"}

    proxy.anthropic_pipeline.transforms = [SpyTransform()]
    proxy.openai_pipeline.transforms = [SpyTransform()]

    await proxy.startup()
    try:
        assert called["n"] == 0, "preload must not run when optimize=False"
        assert proxy.warmup.kompress.status == "null"
        assert proxy.warmup.memory_backend.status == "null"
        assert proxy.warmup.memory_embedder.status == "null"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_memory_embedder_warmup_encodes_once(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    # Swap in a hand-rolled MemoryHandler whose backend exposes a mock
    # embedder. We don't want real ONNX here — just a spy.
    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )
    handler._initialized = True

    embed = AsyncMock(return_value=[0.0])

    class FakeHM:
        def __init__(self):
            self._embedder = type("_E", (), {"embed": embed})()

    class FakeBackend:
        def __init__(self):
            self._hierarchical_memory = FakeHM()

        async def close(self):
            pass

    handler._backend = FakeBackend()
    handler.ensure_initialized = AsyncMock()
    proxy.memory_handler = handler

    await proxy.startup()
    try:
        assert embed.await_count == 1
        assert embed.await_args[0][0] == "warmup"
        assert proxy.warmup.memory_backend.status == "loaded"
        assert proxy.warmup.memory_embedder.status == "loaded"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_memory_backend_error_surfaced_and_health_degraded(tmp_path):
    pytest.importorskip("httpx")
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    async def _boom() -> None:
        raise RuntimeError("synthetic backend init failure")

    handler.ensure_initialized = _boom  # type: ignore[assignment]
    proxy.memory_handler = handler

    # Startup must NOT raise — the memory slot must report error and the
    # rest of the startup pipeline keeps going (quota registry etc.).
    await proxy.startup()
    try:
        assert proxy.warmup.memory_backend.status == "error"
        assert "synthetic" in (proxy.warmup.memory_backend.error or "")
        # Memory embedder stays null because the backend never initialized.
        assert proxy.warmup.memory_embedder.status == "null"
        health = handler.health_status()
        assert health["initialized"] is False
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_logs_token_mode_disabled_features_and_preload_failures(monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=True,
        mode="token",
        smart_routing=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        anthropic_pre_upstream_concurrency=0,
        subscription_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        ccr_proactive_expansion=False,
    )
    proxy = HeadroomProxy(config)

    class FailingTransform:
        def eager_load_compressors(self):
            raise RuntimeError("boom")

    class WeirdTransform:
        def eager_load_compressors(self):
            return "not-a-dict"

    class GoodTransform:
        def eager_load_compressors(self):
            return {"magika": "enabled"}

    proxy.anthropic_pipeline.transforms = [FailingTransform(), WeirdTransform(), GoodTransform()]
    proxy.openai_pipeline.transforms = []

    info_logs: list[str] = []
    warning_logs: list[str] = []
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: info_logs.append(message % args if args else message),
    )
    monkeypatch.setattr(
        server_module.logger,
        "warning",
        lambda message, *args: warning_logs.append(message % args if args else message),
    )

    class _Registry:
        def register(self, tracker) -> None:  # noqa: ANN001
            return None

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            return None

    monkeypatch.setattr(server_module, "get_quota_registry", lambda: _Registry())
    monkeypatch.setattr(server_module, "reset_quota_registry", lambda: None)
    monkeypatch.setattr(server_module, "configure_subscription_tracker", lambda **kwargs: object())
    monkeypatch.setattr(server_module, "get_codex_rate_limit_state", lambda: object())
    monkeypatch.setattr(
        server_module,
        "get_copilot_quota_tracker",
        lambda: type("_Tracker", (), {"is_available": lambda self: False})(),
    )
    monkeypatch.setattr(server_module, "is_telemetry_enabled", lambda: False)

    await proxy.startup()
    try:
        assert any("Prefix freeze: re-freeze after compression" in line for line in info_logs)
        assert any("Read protection window: 30%" in line for line in info_logs)
        assert any("Compression cache: active" in line for line in info_logs)
        assert any("Anthropic pre-upstream concurrency: unbounded" in line for line in info_logs)
        assert any("Smart Routing: DISABLED" in line for line in info_logs)
        assert any("Magika: ENABLED" in line for line in info_logs)
        assert any("Memory: DISABLED" in line for line in info_logs)
        assert any("CCR: DISABLED" in line for line in info_logs)
        assert any("Subscription tracking: DISABLED" in line for line in info_logs)
        assert any("GitHub Copilot quota tracking: DISABLED" in line for line in info_logs)
        assert any("Anonymous telemetry: DISABLED" in line for line in info_logs)
        assert any(
            "Eager preload failed for FailingTransform: boom" in line for line in warning_logs
        )
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_logs_cache_mode_enabled_features_and_tracker_status(monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        mode="cache",
        smart_routing=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        subscription_tracking_enabled=True,
        subscription_poll_interval_s=11,
        subscription_active_window_s=22,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
    )
    proxy = HeadroomProxy(config)

    info_logs: list[str] = []
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: info_logs.append(message % args if args else message),
    )

    class _Registry:
        def __init__(self) -> None:
            self.registered: list[object] = []

        def register(self, tracker) -> None:  # noqa: ANN001
            self.registered.append(tracker)

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            return None

    registry = _Registry()
    monkeypatch.setattr(server_module, "get_quota_registry", lambda: registry)
    monkeypatch.setattr(server_module, "reset_quota_registry", lambda: None)
    monkeypatch.setattr(server_module, "configure_subscription_tracker", lambda **kwargs: object())
    monkeypatch.setattr(server_module, "get_codex_rate_limit_state", lambda: object())
    monkeypatch.setattr(
        server_module,
        "get_copilot_quota_tracker",
        lambda: type("_Tracker", (), {"is_available": lambda self: True})(),
    )
    monkeypatch.setattr(server_module, "is_telemetry_enabled", lambda: True)

    await proxy.startup()
    try:
        assert any("Prefix freeze: strict" in line for line in info_logs)
        assert any("Mutations: latest turn only" in line for line in info_logs)
        assert any("Smart Routing: ENABLED" in line for line in info_logs)
        assert any("CCR (Compress-Cache-Retrieve): ENABLED" in line for line in info_logs)
        assert any("Subscription tracking: ENABLED" in line for line in info_logs)
        assert any("GitHub Copilot quota tracking: ENABLED" in line for line in info_logs)
        assert any("Anonymous telemetry: ENABLED" in line for line in info_logs)
        assert len(registry.registered) == 3
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_marks_memory_slots_null_when_backend_not_initialized(monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    proxy.memory_handler = type(
        "_MemoryHandler",
        (),
        {
            "ensure_initialized": AsyncMock(),
            "health_status": lambda self: {"backend": "local", "initialized": False},
            "warmup_embedder": AsyncMock(return_value=False),
        },
    )()

    info_logs: list[str] = []
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: info_logs.append(message % args if args else message),
    )

    class _Registry:
        def register(self, tracker) -> None:  # noqa: ANN001
            return None

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            return None

    monkeypatch.setattr(server_module, "get_quota_registry", lambda: _Registry())
    monkeypatch.setattr(server_module, "reset_quota_registry", lambda: None)
    monkeypatch.setattr(server_module, "configure_subscription_tracker", lambda **kwargs: object())
    monkeypatch.setattr(server_module, "get_codex_rate_limit_state", lambda: object())
    monkeypatch.setattr(
        server_module,
        "get_copilot_quota_tracker",
        lambda: type("_Tracker", (), {"is_available": lambda self: False})(),
    )
    monkeypatch.setattr(server_module, "is_telemetry_enabled", lambda: False)

    await proxy.startup()
    try:
        assert proxy.warmup.memory_backend.status == "null"
        assert proxy.warmup.memory_embedder.status == "null"
        assert any(
            "Memory: ENABLED (backend=local, initialized=False)" in line for line in info_logs
        )
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("lazy", "Code-Aware: LAZY"),
        ("available", "Code-Aware: available but disabled"),
        ("unavailable", "Code-Aware: not installed"),
    ],
)
async def test_startup_logs_code_aware_status_variants(monkeypatch, status: str, expected: str):
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    proxy._code_aware_status = status

    info_logs: list[str] = []
    monkeypatch.setattr(
        server_module.logger,
        "info",
        lambda message, *args: info_logs.append(message % args if args else message),
    )

    class _Registry:
        def register(self, tracker) -> None:  # noqa: ANN001
            return None

        async def start_all(self) -> None:
            return None

        async def stop_all(self) -> None:
            return None

    monkeypatch.setattr(server_module, "get_quota_registry", lambda: _Registry())
    monkeypatch.setattr(server_module, "reset_quota_registry", lambda: None)
    monkeypatch.setattr(server_module, "configure_subscription_tracker", lambda **kwargs: object())
    monkeypatch.setattr(server_module, "get_codex_rate_limit_state", lambda: object())
    monkeypatch.setattr(
        server_module,
        "get_copilot_quota_tracker",
        lambda: type("_Tracker", (), {"is_available": lambda self: False})(),
    )
    monkeypatch.setattr(server_module, "is_telemetry_enabled", lambda: False)

    await proxy.startup()
    try:
        assert any(expected in line for line in info_logs)
    finally:
        await proxy.shutdown()
