from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from headroom.proxy.memory_tool_adapter import (
    ANTHROPIC_BETA_HEADER,
    MEMORY_TOOL_NAMES,
    MemoryToolAdapter,
    MemoryToolAdapterConfig,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.saved = []
        self.updated = []
        self.deleted = []
        self.closed = False
        self.search_results = []

    async def save_memory(self, **kwargs):  # noqa: ANN003
        self.saved.append(kwargs)
        return SimpleNamespace(id="mem-saved", content=kwargs["content"])

    async def search_memories(self, **kwargs):  # noqa: ANN003
        self.last_search = kwargs
        return list(self.search_results)

    async def update_memory(self, **kwargs):  # noqa: ANN003
        self.updated.append(kwargs)
        return SimpleNamespace(id=kwargs["memory_id"], content=kwargs["new_content"])

    async def delete_memory(self, memory_id):  # noqa: ANN001, ANN201
        self.deleted.append(memory_id)
        return memory_id != "missing"

    async def close(self) -> None:
        self.closed = True


class _FakeBackendNoUpdate:
    def __init__(self) -> None:
        self.saved = []
        self.deleted = []
        self.closed = False
        self.search_results = []

    async def save_memory(self, **kwargs):  # noqa: ANN003
        self.saved.append(kwargs)
        return SimpleNamespace(id="mem-saved", content=kwargs["content"])

    async def search_memories(self, **kwargs):  # noqa: ANN003
        self.last_search = kwargs
        return list(self.search_results)

    async def delete_memory(self, memory_id):  # noqa: ANN001, ANN201
        self.deleted.append(memory_id)
        return memory_id != "missing"

    async def close(self) -> None:
        self.closed = True


def _result(memory_id: str, content: str, score: float, *, entities: list[str] | None = None):
    return SimpleNamespace(
        memory=SimpleNamespace(
            id=memory_id, content=content, metadata={"virtual_path": "/memories/topic"}
        ),
        score=score,
        related_entities=entities or [],
    )


def test_detect_provider_and_beta_headers_cover_supported_inputs() -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))

    assert adapter.detect_provider({"x-api-key": "k"}, None) == "anthropic"
    assert adapter.detect_provider({"authorization": "Bearer sk-test"}, None) == "openai"
    assert adapter.detect_provider(None, "claude-sonnet") == "anthropic"
    assert adapter.detect_provider(None, "gpt-5") == "openai"
    assert adapter.detect_provider(None, "o3-mini") == "openai"
    assert adapter.detect_provider(None, "gemini-2.5-flash") == "gemini"
    assert adapter.detect_provider(None, "gemma-3") == "gemini"
    assert adapter.detect_provider(None, "mystery-model") == "generic"

    assert adapter.get_beta_headers("anthropic") == {"anthropic-beta": ANTHROPIC_BETA_HEADER}
    adapter.config.use_native_tool = False
    assert adapter.get_beta_headers("anthropic") == {}


def test_inject_tools_respects_provider_modes_and_existing_tools() -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))

    injected, headers = adapter.inject_tools([], "anthropic")
    assert any(tool.get("name") == "memory" for tool in injected)
    assert headers == {"anthropic-beta": ANTHROPIC_BETA_HEADER}

    adapter.config.use_native_tool = False
    existing = [{"name": "memory_search"}]
    injected, headers = adapter.inject_tools(existing, "anthropic")
    assert headers == {}
    assert "memory_search" in {tool["name"] for tool in injected}
    assert MEMORY_TOOL_NAMES <= {tool["name"] for tool in injected}

    openai_tools, _ = adapter.inject_tools(
        [{"type": "function", "function": {"name": "memory_save"}}],
        "openai",
    )
    assert MEMORY_TOOL_NAMES <= {tool.get("function", {}).get("name", "") for tool in openai_tools}

    gemini_tools, _ = adapter.inject_tools([{"name": "memory_search"}], "gemini")
    assert MEMORY_TOOL_NAMES <= {tool["name"] for tool in gemini_tools}

    generic_tools, _ = adapter.inject_tools([], "generic")
    assert MEMORY_TOOL_NAMES <= {tool.get("function", {}).get("name", "") for tool in generic_tools}

    disabled = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True, inject_tools=False))
    assert disabled.inject_tools(None, "openai") == ([], {})


def test_extract_tool_calls_names_ids_and_inputs_cover_all_formats() -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))

    anthropic_response = {
        "content": [{"type": "tool_use", "id": "a1", "name": "memory_search", "input": {"q": 1}}]
    }
    openai_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "o1",
                            "function": {
                                "name": "memory_save",
                                "arguments": json.dumps({"content": "x"}),
                            },
                        }
                    ]
                }
            }
        ]
    }
    gemini_response = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "memory_delete", "args": {"id": 1}}}]}}
        ]
    }
    generic_response = {
        "content": [{"type": "tool_use", "id": "g1", "name": "memory_update", "input": {"x": 1}}],
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "g2", "function": {"name": "other", "arguments": "{bad"}}]
                }
            }
        ],
    }

    assert adapter._extract_tool_calls(anthropic_response, "anthropic")[0]["id"] == "a1"
    assert adapter._extract_tool_calls(openai_response, "openai")[0]["id"] == "o1"
    assert (
        adapter._extract_tool_calls(gemini_response, "gemini")[0]["functionCall"]["name"]
        == "memory_delete"
    )
    assert len(adapter._extract_tool_calls(generic_response, "generic")) == 2

    assert adapter._get_tool_name({"name": "memory_search"}, "anthropic") == "memory_search"
    assert adapter._get_tool_name({"function": {"name": "memory_save"}}, "openai") == "memory_save"
    assert (
        adapter._get_tool_name({"functionCall": {"name": "memory_delete"}}, "gemini")
        == "memory_delete"
    )
    assert (
        adapter._get_tool_name({"function": {"name": "memory_update"}}, "generic")
        == "memory_update"
    )

    assert adapter._get_tool_id({"id": "a1"}, "anthropic") == "a1"
    assert adapter._get_tool_id({"id": "o1"}, "openai") == "o1"
    assert (
        adapter._get_tool_id({"functionCall": {"name": "memory_search"}}, "gemini")
        == "memory_search"
    )
    assert adapter._get_tool_id({"id": "g1"}, "generic") == "g1"

    assert adapter._get_tool_input({"input": {"q": 1}}, "anthropic") == {"q": 1}
    assert adapter._get_tool_input(
        {"function": {"arguments": json.dumps({"content": "x"})}},
        "openai",
    ) == {"content": "x"}
    assert adapter._get_tool_input({"function": {"arguments": "{bad"}}, "openai") == {}
    assert adapter._get_tool_input({"functionCall": {"args": {"id": 1}}}, "gemini") == {"id": 1}
    assert adapter._get_tool_input({"input": {"x": 1}}, "generic") == {"x": 1}
    assert adapter._get_tool_input({"function": {"arguments": "{bad"}}, "generic") == {}


def test_has_memory_tool_calls_and_format_tool_result_cover_variants() -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))

    assert (
        adapter.has_memory_tool_calls(
            {"content": [{"type": "tool_use", "name": "memory"}]},
            "anthropic",
        )
        is True
    )
    assert (
        adapter.has_memory_tool_calls(
            {"choices": [{"message": {"tool_calls": [{"function": {"name": "memory_save"}}]}}]},
            "openai",
        )
        is True
    )
    assert (
        adapter.has_memory_tool_calls({"choices": [{"message": {"tool_calls": []}}]}, "openai")
        is False
    )

    assert adapter._format_tool_result("id1", "ok", "anthropic") == {
        "type": "tool_result",
        "tool_use_id": "id1",
        "content": "ok",
    }
    assert adapter._format_tool_result("id1", "ok", "openai") == {
        "role": "tool",
        "tool_call_id": "id1",
        "content": "ok",
    }
    assert adapter._format_tool_result("id1", "ok", "gemini") == {
        "functionResponse": {"name": "id1", "response": {"result": "ok"}}
    }
    assert adapter._format_tool_result("id1", "ok", "generic") == {
        "role": "tool",
        "tool_call_id": "id1",
        "content": "ok",
    }


@pytest.mark.asyncio
async def test_handle_tool_calls_formats_results_and_skips_non_memory_tools(monkeypatch) -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))

    async def fake_init() -> None:
        return None

    async def fake_native(input_data, user_id):  # noqa: ANN001, ANN201
        return "native-result"

    async def fake_custom(tool_name, input_data, user_id):  # noqa: ANN001, ANN201
        return f"custom:{tool_name}:{user_id}"

    monkeypatch.setattr(adapter, "_ensure_initialized", fake_init)
    monkeypatch.setattr(adapter, "_execute_native_tool", fake_native)
    monkeypatch.setattr(adapter, "_execute_custom_tool", fake_custom)

    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "1", "function": {"name": "memory_save", "arguments": "{}"}},
                        {"id": "2", "function": {"name": "other", "arguments": "{}"}},
                    ]
                }
            }
        ]
    }
    results = await adapter.handle_tool_calls(response, "user-1", "openai")
    assert results == [
        {"role": "tool", "tool_call_id": "1", "content": "custom:memory_save:user-1"}
    ]

    native_results = await adapter.handle_tool_calls(
        {"content": [{"type": "tool_use", "id": "n1", "name": "memory", "input": {}}]},
        "user-2",
        "anthropic",
    )
    assert native_results == [
        {"type": "tool_result", "tool_use_id": "n1", "content": "native-result"}
    ]


@pytest.mark.asyncio
async def test_native_tool_commands_and_custom_tools_cover_success_and_errors() -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))
    backend = _FakeBackend()
    adapter._backend = backend

    backend.search_results = [_result("m1", "alpha memory", 0.91), _result("m2", "beta" * 80, 0.55)]
    assert "Found 2 memories matching 'alpha'" in await adapter._semantic_search("alpha", "user-1")
    assert "Memory System (2 memories stored)" in await adapter._get_memory_overview("user-1")
    assert "Please provide a search query" in await adapter._native_view(
        {"path": "/memories/search/"}, "user-1"
    )
    assert "Found 2 memories matching 'alpha'" in await adapter._native_view(
        {"path": "/memories/search/alpha"}, "user-1"
    )
    assert "Memory System (2 memories stored)" in await adapter._native_view(
        {"path": "/memories"}, "user-1"
    )

    assert await adapter._native_create(
        {"path": "/memories/topic.txt", "file_text": "saved text"}, "user-1"
    ) == ("File created successfully at: /memories/topic.txt")
    assert (
        await adapter._native_create({"path": "/memories/topic.txt"}, "user-1")
        == "Error: file_text is required"
    )

    backend.search_results = [_result("m1", "replace this value", 0.99)]
    assert await adapter._native_update({"old_str": "replace", "new_str": "updated"}, "user-1") == (
        "The memory has been edited."
    )
    assert backend.updated[0]["memory_id"] == "m1"
    assert (
        await adapter._native_update({"new_str": "updated"}, "user-1")
        == "Error: old_str is required"
    )

    backend_no_update = _FakeBackendNoUpdate()
    backend_no_update.search_results = [_result("m2", "replace again", 0.99)]
    adapter._backend = backend_no_update
    assert await adapter._native_update({"old_str": "replace", "new_str": "done"}, "user-1") == (
        "The memory has been edited."
    )
    assert backend_no_update.deleted == ["m2"]

    backend_delete = _FakeBackend()
    backend_delete.search_results = [_result("m3", "topic", 0.9)]
    adapter._backend = backend_delete
    assert (
        await adapter._native_delete({"path": "/memories/topic"}, "user-1")
        == "Successfully deleted /memories/topic"
    )
    backend_delete.search_results = []
    assert await adapter._native_delete({"path": "/memories/missing"}, "user-1") == (
        "Error: The path /memories/missing does not exist"
    )

    adapter._backend = None
    assert (
        await adapter._execute_native_tool({"command": "view"}, "user-1")
        == "Error: Memory backend not initialized"
    )


@pytest.mark.asyncio
async def test_custom_tool_execution_and_lifecycle_cover_branches(monkeypatch) -> None:
    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))
    backend = _FakeBackend()
    adapter._backend = backend

    saved = json.loads(
        await adapter._execute_save({"content": "remember this", "importance": 0.9}, "user-1")
    )
    assert saved["status"] == "saved"
    assert json.loads(await adapter._execute_save({}, "user-1"))["error"] == "content is required"

    backend.search_results = [_result("m1", "remember this", 0.912, entities=["alice"])]
    searched = json.loads(
        await adapter._execute_search({"query": "remember", "top_k": 3}, "user-1")
    )
    assert searched["status"] == "found"
    assert searched["memories"][0]["entities"] == ["alice"]
    assert json.loads(await adapter._execute_search({}, "user-1"))["error"] == "query is required"

    updated = json.loads(
        await adapter._execute_update({"memory_id": "m1", "new_content": "updated"}, "user-1")
    )
    assert updated["status"] == "updated"
    assert (
        json.loads(await adapter._execute_update({"new_content": "x"}, "user-1"))["error"]
        == "memory_id is required"
    )
    assert (
        json.loads(await adapter._execute_update({"memory_id": "m1"}, "user-1"))["error"]
        == "new_content is required"
    )

    backend_no_update = _FakeBackendNoUpdate()
    adapter._backend = backend_no_update
    updated = json.loads(
        await adapter._execute_update({"memory_id": "m2", "new_content": "replacement"}, "user-1")
    )
    assert updated["note"] == "Replaced via delete+save"

    adapter._backend = backend
    deleted = json.loads(await adapter._execute_delete({"memory_id": "m1"}, "user-1"))
    assert deleted["status"] == "deleted"
    not_found = json.loads(await adapter._execute_delete({"memory_id": "missing"}, "user-1"))
    assert not_found["status"] == "not_found"
    assert (
        json.loads(await adapter._execute_delete({}, "user-1"))["error"] == "memory_id is required"
    )

    adapter._backend = None
    assert json.loads(
        await adapter._execute_custom_tool("memory_save", {"content": "x"}, "user-1")
    )["error"] == ("Memory backend not initialized")

    adapter._backend = backend
    assert (
        json.loads(await adapter._execute_custom_tool("unknown", {}, "user-1"))["error"]
        == "Unknown tool: unknown"
    )

    async def boom(input_data, user_id):  # noqa: ANN001, ANN201
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_execute_save", boom)
    failed = json.loads(
        await adapter._execute_custom_tool("memory_save", {"content": "x"}, "user-1")
    )
    assert failed["status"] == "error"
    assert failed["error"] == "boom"

    await adapter.close()
    assert backend.closed is True
    assert adapter._backend is None
    assert adapter._initialized is False


@pytest.mark.asyncio
async def test_ensure_initialized_uses_local_backend_module(monkeypatch) -> None:
    created = {}
    fake_module = types.ModuleType("headroom.memory.backends.local")

    class LocalBackendConfig:
        def __init__(self, db_path):  # noqa: ANN001
            self.db_path = db_path

    class LocalBackend:
        def __init__(self, config):  # noqa: ANN001
            created["db_path"] = config.db_path

        async def _ensure_initialized(self):
            created["initialized"] = True

    fake_module.LocalBackend = LocalBackend
    fake_module.LocalBackendConfig = LocalBackendConfig
    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", fake_module)

    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True, db_path="test-memory.db"))
    await adapter._ensure_initialized()
    await adapter._ensure_initialized()

    assert created == {"db_path": "test-memory.db", "initialized": True}
    assert adapter._initialized is True

    disabled = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=False))
    await disabled._ensure_initialized()
    assert disabled._initialized is False
