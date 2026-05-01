from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_mcp_module = types.ModuleType("mcp")
_mcp_server_module = types.ModuleType("mcp.server")
_mcp_stdio_module = types.ModuleType("mcp.server.stdio")
_mcp_types_module = types.ModuleType("mcp.types")


class _StubTool:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _StubTextContent:
    def __init__(self, *, type: str, text: str) -> None:
        self.type = type
        self.text = text


_mcp_server_module.Server = object
_mcp_stdio_module.stdio_server = lambda: None
_mcp_types_module.TextContent = _StubTextContent
_mcp_types_module.Tool = _StubTool

sys.modules.setdefault("mcp", _mcp_module)
sys.modules.setdefault("mcp.server", _mcp_server_module)
sys.modules.setdefault("mcp.server.stdio", _mcp_stdio_module)
sys.modules.setdefault("mcp.types", _mcp_types_module)

mcp_server = importlib.import_module("headroom.memory.mcp_server")


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._list_tools = None
        self._call_tool = None
        self.run = AsyncMock()

    def list_tools(self):
        def decorator(func):
            self._list_tools = func
            return func

        return decorator

    def call_tool(self):
        def decorator(func):
            self._call_tool = func
            return func

        return decorator

    def create_initialization_options(self):
        return {"name": self.name}


@pytest.mark.asyncio
async def test_warm_up_backend_indexes_memories() -> None:
    mem_without_embedding = SimpleNamespace(id="m1", content="first", embedding=None)
    mem_with_embedding = SimpleNamespace(id="m2", content="second", embedding=[9])
    backend = SimpleNamespace(
        _ensure_initialized=AsyncMock(),
        _hierarchical_memory=SimpleNamespace(
            _embedder=SimpleNamespace(
                embed=AsyncMock(side_effect=[["warm"], ["first-embedding"]]),
                embed_batch=AsyncMock(return_value=[["first-embedding"], [9]]),
            ),
            _store=SimpleNamespace(save=AsyncMock(), save_batch=AsyncMock()),
            _vector_index=SimpleNamespace(index=AsyncMock(), index_batch=AsyncMock()),
        ),
        get_user_memories=AsyncMock(return_value=[mem_without_embedding, mem_with_embedding]),
    )

    await mcp_server._warm_up_backend(backend, "user-1")

    assert mem_without_embedding.embedding == ["first-embedding"]

    stored_ids = []
    for call in backend._hierarchical_memory._store.save.await_args_list:
        stored_ids.append(call.args[0].id)
    for call in backend._hierarchical_memory._store.save_batch.await_args_list:
        stored_ids.extend(mem.id for mem in call.args[0])
    assert stored_ids == ["m1"]

    indexed_ids = []
    for call in backend._hierarchical_memory._vector_index.index.await_args_list:
        indexed_ids.append(call.args[0].id)
    for call in backend._hierarchical_memory._vector_index.index_batch.await_args_list:
        indexed_ids.extend(mem.id for mem in call.args[0])
    assert sorted(indexed_ids) == ["m1", "m2"]

    backend._hierarchical_memory = None
    await mcp_server._warm_up_backend(backend, "user-1")


@pytest.mark.asyncio
async def test_handle_search_filters_results_and_reports_errors() -> None:
    active = SimpleNamespace(
        score=0.91,
        memory=SimpleNamespace(id="a1", content="current fact", superseded_by=None),
        related_entities=["repo", "coverage"],
    )
    stale = SimpleNamespace(
        score=0.9,
        memory=SimpleNamespace(id="b2", content="old fact", superseded_by=None),
        related_entities=[],
    )
    superseded = SimpleNamespace(
        score=0.99,
        memory=SimpleNamespace(id="c3", content="superseded", superseded_by="newer"),
        related_entities=[],
    )
    backend = SimpleNamespace(
        search_memories=AsyncMock(return_value=[active, stale, superseded]),
        get_memory=AsyncMock(
            side_effect=[
                SimpleNamespace(superseded_by=None),
                SimpleNamespace(superseded_by="newer"),
            ]
        ),
    )

    results = await mcp_server._handle_search(backend, {"query": "fact", "top_k": 1}, "user-1")
    assert "current fact" in results[0].text
    assert "Related: repo, coverage" in results[0].text
    assert "old fact" not in results[0].text

    assert "query is required" in (await mcp_server._handle_search(backend, {}, "user-1"))[0].text

    backend.search_memories = AsyncMock(return_value=[])
    assert (await mcp_server._handle_search(backend, {"query": "miss"}, "user-1"))[
        0
    ].text == "No memories found."

    backend.search_memories = AsyncMock(side_effect=RuntimeError("search broke"))
    assert (
        "Search error: search broke"
        in (await mcp_server._handle_search(backend, {"query": "fact"}, "user-1"))[0].text
    )


@pytest.mark.asyncio
async def test_handle_save_supports_supersede_save_and_errors() -> None:
    existing = SimpleNamespace(
        score=0.9,
        memory=SimpleNamespace(id="abc12345", content="old", superseded_by=None),
    )
    backend = SimpleNamespace(
        search_memories=AsyncMock(
            side_effect=[
                [existing],
                [],
            ]
        ),
        update_memory=AsyncMock(return_value=SimpleNamespace(id="def67890")),
        save_memory=AsyncMock(return_value=SimpleNamespace(id="ghi54321")),
    )

    result = await mcp_server._handle_save(
        backend,
        {"facts": [" updated fact ", " ", "new fact"], "importance": 0.9},
        "user-1",
    )
    assert "Saved 1 new, updated 1 existing (2 total)" in result[0].text
    assert "updated [abc12345" in result[0].text
    assert "saved [ghi54321]" in result[0].text

    compat = await mcp_server._handle_save(backend, {"content": "legacy fact"}, "user-1")
    assert "Saved" in compat[0].text

    missing = await mcp_server._handle_save(backend, {}, "user-1")
    assert "facts array is required" in missing[0].text

    backend.search_memories = AsyncMock(side_effect=RuntimeError("lookup failed"))
    backend.save_memory = AsyncMock(side_effect=RuntimeError("save failed"))
    errored = await mcp_server._handle_save(backend, {"facts": ["broken"]}, "user-1")
    assert "Save error: save failed" in errored[0].text


@pytest.mark.asyncio
async def test_create_memory_server_run_and_main(monkeypatch, tmp_path: Path) -> None:
    fake_server = _FakeServer("headroom-memory")
    fake_backend = SimpleNamespace()
    monkeypatch.setattr(mcp_server, "Server", lambda name: fake_server)
    monkeypatch.setattr(
        mcp_server, "LocalBackendConfig", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(mcp_server, "LocalBackend", lambda config: fake_backend)
    monkeypatch.setattr(mcp_server, "_warm_up_backend", AsyncMock())
    monkeypatch.setattr(
        mcp_server,
        "_handle_search",
        AsyncMock(return_value=[mcp_server.TextContent(type="text", text="search ok")]),
    )
    monkeypatch.setattr(
        mcp_server,
        "_handle_save",
        AsyncMock(return_value=[mcp_server.TextContent(type="text", text="save ok")]),
    )

    server = mcp_server.create_memory_server(str(tmp_path / "memory.db"), user_id="alice")
    assert server is fake_server
    tools = await fake_server._list_tools()
    assert {tool.name for tool in tools} == {"memory_search", "memory_save"}
    await asyncio.sleep(0)

    assert (await fake_server._call_tool("memory_search", {"query": "x"}))[0].text == "search ok"
    assert (await fake_server._call_tool("memory_save", {"facts": ["x"]}))[0].text == "save ok"
    assert "Unknown tool: other" in (await fake_server._call_tool("other", {}))[0].text

    class _FakeStdio:
        async def __aenter__(self):
            return ("read-stream", "write-stream")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mcp_server, "stdio_server", lambda: _FakeStdio())
    await mcp_server._run(str(tmp_path / "memory.db"), "alice")
    fake_server.run.assert_awaited_once()

    captured = {}
    monkeypatch.setattr(mcp_server.logging, "basicConfig", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setenv("USERNAME", "env-user")
    monkeypatch.setattr(mcp_server.sys, "argv", ["prog", "--db", "custom.db"])

    def _consume(coro):
        captured["run_coro"] = coro
        coro.close()

    monkeypatch.setattr(mcp_server.asyncio, "run", _consume)
    mcp_server.main()
    assert captured["stream"] is mcp_server.sys.stderr
    assert captured["level"] == mcp_server.logging.INFO
    assert captured["format"] == "%(name)s: %(message)s"
    assert mcp_server.os.environ["HF_HUB_OFFLINE"] == "1"
    assert mcp_server.os.environ["TRANSFORMERS_OFFLINE"] == "1"
