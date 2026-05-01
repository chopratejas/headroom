from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_wrapper_tools(monkeypatch):
    extraction_module = types.ModuleType("headroom.memory.extraction")
    extraction_module.EXTRACTION_SYSTEM_PROMPT = "Extract facts"

    class FakeMemorySystem:
        def __init__(self, backend, user_id, session_id):
            self.backend = backend
            self.user_id = user_id
            self.session_id = session_id
            self.calls = []

        async def process_tool_call(self, name, args):
            self.calls.append((name, args))
            if args.get("explode"):
                raise RuntimeError("boom")
            return {"success": True, "message": f"handled {name}", "args": args}

    system_module = types.ModuleType("headroom.memory.system")
    system_module.MemoryBackend = object
    system_module.MemorySystem = FakeMemorySystem

    tools_module = types.ModuleType("headroom.memory.tools")
    tools_module.get_memory_tools = lambda: [
        {"type": "function", "function": {"name": "memory_save"}}
    ]
    tools_module.get_memory_tools_optimized = lambda: [
        {"type": "function", "function": {"name": "memory_save_optimized"}}
    ]
    tools_module.get_tool_names = lambda: ["memory_save", "memory_search"]

    monkeypatch.setitem(sys.modules, "headroom.memory.extraction", extraction_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.system", system_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.tools", tools_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "wrapper_tools.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.wrapper_tools", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.wrapper_tools", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.wrapper_tools"] = module
    spec.loader.exec_module(module)
    return module, FakeMemorySystem


def _response_with_tool_calls(*tool_calls):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=list(tool_calls),
                )
            )
        ]
    )


def _tool_call(tool_id, name, arguments):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_memory_tools_wrapper_sync_paths(monkeypatch) -> None:
    wrapper_tools, FakeMemorySystem = _load_wrapper_tools(monkeypatch)

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _response_with_tool_calls(
                _tool_call("ok", "memory_save", '{"topic":"python"}'),
                _tool_call("bad-json", "memory_search", "{oops"),
                _tool_call("boom", "memory_save", '{"explode": true}'),
                _tool_call("skip", "non_memory", "{}"),
            )

    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        extra_attr="value",
    )

    wrapper = wrapper_tools.MemoryToolsWrapper(
        client,
        backend=object(),
        user_id="user",
        session_id="sess",
        optimized=True,
        inject_extraction_prompt=True,
    )
    assert isinstance(wrapper.memory, FakeMemorySystem)
    assert wrapper.extra_attr == "value"

    messages = [{"role": "user", "content": "remember this"}]
    response = wrapper.chat.completions.create(
        messages=messages, tools=[{"name": "custom"}], model="gpt"
    )
    assert messages == [{"role": "user", "content": "remember this"}]
    sent = completions.calls[0]
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"] == "Extract facts"
    assert sent["tools"][0]["function"]["name"] == "memory_save_optimized"
    assert sent["tools"][1] == {"name": "custom"}
    assert response._memory_tool_results["ok"]["args"] == {"topic": "python"}
    assert response._memory_tool_results["bad-json"]["success"] is False
    assert response._memory_tool_results["boom"]["success"] is False

    messages_with_system = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "hi"},
    ]
    prepared = wrapper.chat.completions._prepare_messages(messages_with_system)
    assert prepared[0]["content"].endswith("Extract facts")

    no_tools = wrapper.chat.completions._process_memory_tool_calls(
        SimpleNamespace(choices=[]), {"memory_save"}
    )
    assert no_tools == {}

    wrapper_no_auto = wrapper_tools.MemoryToolsWrapper(
        client,
        backend=object(),
        user_id="user",
        auto_handle_tools=False,
    )
    plain = wrapper_no_auto.chat.completions.create(
        messages=[{"role": "user", "content": "x"}], model="gpt"
    )
    assert not hasattr(plain, "_memory_tool_results")


def test_run_async_with_running_loop(monkeypatch) -> None:
    wrapper_tools, _FakeMemorySystem = _load_wrapper_tools(monkeypatch)
    completions = wrapper_tools.MemoryToolsCompletions(
        completions=SimpleNamespace(),
        memory=SimpleNamespace(),
        auto_handle=False,
    )

    monkeypatch.setattr(wrapper_tools.asyncio, "get_running_loop", lambda: object())

    class FakeFuture:
        def result(self):
            return "thread-result"

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, coro):
            coro.close()
            return FakeFuture()

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", FakePool)
    assert completions._run_async(asyncio.sleep(0, result="unused")) == "thread-result"


@pytest.mark.asyncio
async def test_memory_tools_wrapper_async_paths(monkeypatch) -> None:
    wrapper_tools, _FakeMemorySystem = _load_wrapper_tools(monkeypatch)

    class AsyncCompletions:
        def __init__(self):
            self.acreate_calls = []
            self.create_calls = []

        async def acreate(self, **kwargs):
            self.acreate_calls.append(kwargs)
            return _response_with_tool_calls(_tool_call("a", "memory_save", '{"k":1}'))

    async_only = AsyncCompletions()
    memory = SimpleNamespace(
        process_tool_call=lambda name, args: asyncio.sleep(0, result={"message": "ok"})
    )
    completions = wrapper_tools.MemoryToolsCompletions(
        async_only,
        memory,
        auto_handle=True,
        optimized=False,
        inject_extraction_prompt=False,
    )
    response = await completions.acreate(messages=[{"role": "user", "content": "hi"}], model="gpt")
    assert response._memory_tool_results["a"]["message"] == "ok"
    assert async_only.acreate_calls[0]["tools"][0]["function"]["name"] == "memory_save"

    class AsyncCreateOnly:
        async def create(self, **kwargs):
            return _response_with_tool_calls(_tool_call("b", "memory_search", '{"q":"x"}'))

    async_create = wrapper_tools.MemoryToolsCompletions(
        AsyncCreateOnly(),
        memory,
        auto_handle=True,
    )
    response2 = await async_create.acreate(
        messages=[{"role": "user", "content": "hi"}], model="gpt"
    )
    assert response2._memory_tool_results["b"]["message"] == "ok"

    class SyncOnly:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _response_with_tool_calls(
                _tool_call("json", "memory_save", "{bad"),
                _tool_call("err", "memory_search", '{"explode": true}'),
            )

    sync_only = SyncOnly()
    completions3 = wrapper_tools.MemoryToolsCompletions(
        sync_only, wrapper_tools.MemorySystem(object(), "user", None), auto_handle=True
    )

    class FakeLoop:
        async def run_in_executor(self, executor, func):
            return func()

    monkeypatch.setattr(wrapper_tools.asyncio, "get_running_loop", lambda: FakeLoop())
    response3 = await completions3.acreate(
        messages=[{"role": "user", "content": "hi"}], model="gpt"
    )
    assert response3._memory_tool_results["json"]["success"] is False
    assert response3._memory_tool_results["err"]["success"] is False

    wrapped = wrapper_tools.with_memory_tools(
        SimpleNamespace(chat=SimpleNamespace(completions=sync_only)), object(), "user"
    )
    assert isinstance(wrapped, wrapper_tools.MemoryToolsWrapper)
