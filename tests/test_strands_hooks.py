from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace


def _load_hooks_module(monkeypatch):
    strands_module = types.ModuleType("strands")
    strands_module.__spec__ = importlib.machinery.ModuleSpec("strands", loader=None)
    hooks_module = types.ModuleType("strands.hooks")
    hooks_module.__spec__ = importlib.machinery.ModuleSpec("strands.hooks", loader=None)
    events_module = types.ModuleType("strands.hooks.events")
    events_module.__spec__ = importlib.machinery.ModuleSpec("strands.hooks.events", loader=None)
    tools_module = types.ModuleType("strands.types.tools")
    tools_module.__spec__ = importlib.machinery.ModuleSpec("strands.types.tools", loader=None)

    class HookProvider:
        pass

    class HookRegistry:
        def __init__(self) -> None:
            self.callbacks: list[tuple[object, object]] = []

        def add_callback(self, event_type, callback) -> None:
            self.callbacks.append((event_type, callback))

    class AfterToolCallEvent:
        def __init__(self, result: dict, tool_use: dict) -> None:
            self.result = result
            self.tool_use = tool_use

    class BeforeToolCallEvent:
        pass

    hooks_module.HookProvider = HookProvider
    hooks_module.HookRegistry = HookRegistry
    events_module.AfterToolCallEvent = AfterToolCallEvent
    events_module.BeforeToolCallEvent = BeforeToolCallEvent
    tools_module.ToolResult = dict

    monkeypatch.setitem(sys.modules, "strands", strands_module)
    monkeypatch.setitem(sys.modules, "strands.hooks", hooks_module)
    monkeypatch.setitem(sys.modules, "strands.hooks.events", events_module)
    monkeypatch.setitem(sys.modules, "strands.types.tools", tools_module)
    monkeypatch.delitem(sys.modules, "headroom.integrations.strands.hooks", raising=False)
    return importlib.import_module("headroom.integrations.strands.hooks")


class _FakeCrusher:
    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[tuple[str, str]] = []
        self.result = SimpleNamespace(compressed="short text", was_modified=True)

    def crush(self, *, content: str, query: str):
        self.calls.append((content, query))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_strands_hook_provider_helpers_and_registration(monkeypatch) -> None:
    hooks = _load_hooks_module(monkeypatch)
    fake_crushers: list[_FakeCrusher] = []

    monkeypatch.setattr(hooks, "SmartCrusherConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        hooks,
        "SmartCrusher",
        lambda config: fake_crushers.append(_FakeCrusher(config)) or fake_crushers[-1],
    )

    provider = hooks.HeadroomHookProvider(
        min_tokens_to_compress=55,
        config=SimpleNamespace(smart_crusher=SimpleNamespace(max_items_after_crush=7)),
    )
    assert hooks.strands_available() is True
    assert provider._initialized is True
    crusher = provider.crusher
    assert crusher is provider.crusher
    assert crusher.config.min_tokens_to_crush == 55
    assert crusher.config.max_items_after_crush == 7
    assert provider.total_tokens_saved == 0
    assert provider.metrics_history == []

    registry = hooks.HookRegistry()
    provider.register_hooks(registry)
    assert registry.callbacks[0][0] is hooks.AfterToolCallEvent

    disabled = hooks.HeadroomHookProvider(compress_tool_outputs=False)
    empty_registry = hooks.HookRegistry()
    disabled.register_hooks(empty_registry)
    assert empty_registry.callbacks == []

    assert provider._estimate_tokens("") == 0
    assert provider._estimate_tokens("abcd1234") == 2

    assert provider._extract_text_content({"content": []}) == ""
    extracted = provider._extract_text_content(
        {
            "content": [
                {"text": "hello"},
                {"json": {"a": 1}},
                {"json": object()},
                "tail",
            ]
        }
    )
    assert "hello" in extracted
    assert '{"a": 1}' in extracted
    assert "tail" in extracted

    result = {"content": []}
    provider._update_result_content(result, "compressed")
    assert result["content"] == [{"text": "compressed"}]

    json_result = {"content": [{"json": {"a": 1}}]}
    provider._update_result_content(json_result, '{"b": 2}')
    assert json_result["content"] == [{"json": {"b": 2}}]
    provider._update_result_content(json_result, "not-json")
    assert json_result["content"] == [{"text": "not-json"}]

    text_result = {"content": [{"text": "original"}]}
    provider._update_result_content(text_result, "new-text")
    assert text_result["content"] == [{"text": "new-text"}]

    weird_result = {"content": ["plain"]}
    provider._update_result_content(weird_result, "fallback")
    assert weird_result["content"] == [{"text": "fallback"}]

    assert disabled._should_skip_compression({"content": [{"text": "x"}]}) == "compression_disabled"
    assert (
        provider._should_skip_compression({"status": "error", "content": [{"text": "x"}]})
        == "error_result_preserved"
    )
    assert provider._should_skip_compression({"content": []}) == "empty_content"
    assert provider._should_skip_compression({"content": [{"text": "ok"}]}) is None


def test_strands_hook_provider_compression_paths_and_metrics(monkeypatch) -> None:
    hooks = _load_hooks_module(monkeypatch)
    fake_crushers: list[_FakeCrusher] = []

    monkeypatch.setattr(hooks, "SmartCrusherConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        hooks,
        "SmartCrusher",
        lambda config: fake_crushers.append(_FakeCrusher(config)) or fake_crushers[-1],
    )
    monkeypatch.setattr(hooks, "uuid4", lambda: "req-1")

    provider = hooks.HeadroomHookProvider(min_tokens_to_compress=5)
    event = hooks.AfterToolCallEvent(
        result={"content": [{"text": "this is a long tool result"}]},
        tool_use={"name": "search", "toolUseId": "tool-1"},
    )
    provider._compress_tool_result(event)
    assert event.result["content"] == [{"text": "short text"}]
    assert provider.total_tokens_saved > 0
    assert provider.metrics_history[-1].was_compressed is True

    below = hooks.AfterToolCallEvent(
        result={"content": [{"text": "tiny"}]},
        tool_use={"name": "search", "toolUseId": "tool-2"},
    )
    provider._compress_tool_result(below)
    assert provider.metrics_history[-1].skip_reason.startswith("below_threshold:")

    fake_crushers[0].result = RuntimeError("boom")
    errored = hooks.AfterToolCallEvent(
        result={"content": [{"text": "this is another long tool result"}]},
        tool_use={"name": "search", "toolUseId": "tool-3"},
    )
    provider._compress_tool_result(errored)
    assert provider.metrics_history[-1].skip_reason == "compression_error:RuntimeError"

    fake_crushers[0].result = SimpleNamespace(
        compressed="this is even longer than before", was_modified=False
    )
    no_reduction = hooks.AfterToolCallEvent(
        result={"content": [{"text": "this is still a long tool result"}]},
        tool_use={"name": "search", "toolUseId": "tool-4"},
    )
    provider._compress_tool_result(no_reduction)
    assert provider.metrics_history[-1].skip_reason == "no_reduction"

    skipped = hooks.AfterToolCallEvent(
        result={"status": "error", "content": [{"text": "bad"}]},
        tool_use={"name": "search", "toolUseId": "tool-5"},
    )
    provider._compress_tool_result(skipped)
    assert provider.metrics_history[-1].skip_reason == "error_result_preserved"

    summary = provider.get_savings_summary()
    assert summary["total_requests"] == len(provider.metrics_history)
    assert summary["compressed_requests"] == 1
    assert summary["total_tokens_saved"] == provider.total_tokens_saved
    assert summary["total_tokens_before"] >= summary["total_tokens_after"]

    for idx in range(110):
        provider._record_metrics(
            request_id=f"req-{idx}",
            tool_name="tool",
            tool_use_id=str(idx),
            tokens_before=10,
            tokens_after=5,
            was_compressed=True,
            skip_reason=None,
        )
    assert len(provider.metrics_history) == 100

    provider.reset()
    assert provider.get_savings_summary() == {
        "total_requests": 0,
        "compressed_requests": 0,
        "total_tokens_saved": 0,
        "average_savings_percent": 0.0,
        "total_tokens_before": 0,
        "total_tokens_after": 0,
    }
