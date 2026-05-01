from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from headroom.ccr import mcp_server


@dataclass
class _FakeTextContent:
    type: str
    text: str


@dataclass
class _FakeTool:
    name: str
    description: str
    inputSchema: dict


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.list_tools_handler = None
        self.call_tool_handler = None

    def list_tools(self):
        def decorator(fn):
            self.list_tools_handler = fn
            return fn

        return decorator

    def call_tool(self):
        def decorator(fn):
            self.call_tool_handler = fn
            return fn

        return decorator

    def create_initialization_options(self):
        return {}

    async def run(self, *args, **kwargs) -> None:
        return None


def _make_server(
    monkeypatch: pytest.MonkeyPatch, *, read_enabled: bool = False
) -> mcp_server.HeadroomMCPServer:
    monkeypatch.setattr(mcp_server, "MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_server, "Server", _FakeServer)
    monkeypatch.setattr(mcp_server, "TextContent", _FakeTextContent, raising=False)
    monkeypatch.setattr(mcp_server, "Tool", _FakeTool, raising=False)
    monkeypatch.setattr(mcp_server, "_READ_ENABLED", read_enabled)
    return mcp_server.HeadroomMCPServer(proxy_url="http://proxy", check_proxy=True)


def test_format_summary_shared_events_and_session_stats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
    monkeypatch.setattr(mcp_server, "fcntl", None)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", tmp_path / "session_stats.jsonl")
    monkeypatch.setattr(mcp_server.os, "getpid", lambda: 42)
    monkeypatch.setattr(mcp_server.time, "time", lambda: 1000.0)

    mcp_server.SHARED_STATS_FILE.write_text(
        "\n".join(
            [
                json.dumps({"type": "compress", "timestamp": 995.0, "pid": 1}),
                json.dumps({"type": "compress", "timestamp": 100.0, "pid": 2}),
                "{bad json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    events = mcp_server._read_shared_events(window_seconds=10)
    assert events == [{"type": "compress", "timestamp": 995.0, "pid": 1}]
    pruned = mcp_server.SHARED_STATS_FILE.read_text(encoding="utf-8")
    assert '"timestamp": 100.0' not in pruned

    appended: list[dict] = []
    monkeypatch.setattr(
        mcp_server, "_append_shared_event", lambda event: appended.append(dict(event))
    )
    stats = mcp_server.SessionStats(started_at=900.0)
    for idx in range(55):
        stats.record_compression(100 + idx, 40 + idx, "route")
    stats.record_retrieval("0123456789abcdef")
    payload = stats.to_dict()

    assert payload["session_duration_seconds"] == 100
    assert payload["compressions"] == 55
    assert payload["retrievals"] == 1
    assert payload["total_tokens_saved"] > 0
    assert payload["savings_percent"] > 0
    assert len(payload["recent_events"]) == 10
    assert appended[-1]["hash"] == "0123456789ab"

    summary_text = mcp_server._format_session_summary(
        {
            "mode": "cache",
            "api_requests": 12,
            "primary_model": "claude-sonnet",
            "compression": {
                "requests_compressed": 5,
                "avg_compression_pct": 48,
                "best_compression_pct": 71,
                "best_detail": "content_router",
                "total_tokens_removed": 12000,
            },
            "uncompressed_requests": {
                "prefix_frozen": 2,
                "too_small": 3,
            },
            "cost": {
                "without_headroom_usd": 4.5,
                "with_headroom_usd": 2.25,
                "total_saved_usd": 2.25,
                "savings_pct": 50,
                "breakdown": {
                    "cache_savings_usd": 1.0,
                    "compression_savings_usd": 1.25,
                },
            },
            "tip": "Keep using cache mode for stable prefixes.",
        },
        {"compressions": 3, "total_tokens_saved": 200},
    )
    assert "Compression (5 requests compressed)" in summary_text
    assert "Uncompressed requests (5)" in summary_text
    assert "Cost Impact:" in summary_text
    assert "MCP Tool: 3 compressions, 200 tokens saved" in summary_text
    assert "Tip: Keep using cache mode" in summary_text


@pytest.mark.asyncio
async def test_headroom_mcp_server_retrieve_stats_and_cleanup(monkeypatch) -> None:
    server = _make_server(monkeypatch)

    class _Entry:
        original_content = "full text"
        original_item_count = 7
        compressed_item_count = 2
        retrieval_count = 1

    class _Store:
        def __init__(self) -> None:
            self.mode = "search"

        def search(self, hash_key, query):
            return [{"match": query}] if self.mode == "search" else []

        def retrieve(self, hash_key):
            return _Entry() if self.mode == "retrieve" else None

        def get_stats(self):
            return {"entry_count": 3, "max_entries": 500}

    store = _Store()
    server._local_store = store

    result = await server._retrieve_content("abc123", "needle")
    assert result["source"] == "local"
    assert result["count"] == 1

    store.mode = "retrieve"
    result = await server._retrieve_content("abc123", None)
    assert result["source"] == "local"
    assert result["original_content"] == "full text"

    store.mode = "none"
    monkeypatch.setattr(mcp_server, "HTTPX_AVAILABLE", True)
    server._retrieve_via_proxy = lambda hash_key, query: _return_async(
        {"hash": hash_key, "value": "proxy"}
    )  # type: ignore[method-assign]
    result = await server._retrieve_content("abc123", None)
    assert result["source"] == "proxy"

    server._retrieve_via_proxy = lambda hash_key, query: _return_async({"error": "missing"})  # type: ignore[method-assign]
    result = await server._retrieve_content("abc123", None)
    assert result["error"].startswith("Content not found")

    proxy_stats = server._extract_proxy_stats(
        {
            "requests_total": 4,
            "tokens_saved_total": 1000,
            "cache": {"hits": 3, "misses": 1, "hit_rate": 75},
            "cost": {"total_saved": 1.25},
        }
    )
    assert proxy_stats == {
        "requests_total": 4,
        "tokens_saved_total": 1000,
        "cache": {"hits": 3, "misses": 1, "hit_rate": 75},
        "cost_saved_usd": 1.25,
    }

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = _Client()
    server._http_client = client
    await server.cleanup()
    assert client.closed is True


@pytest.mark.asyncio
async def test_headroom_mcp_server_handles_stats_and_cached_reads(monkeypatch, tmp_path) -> None:
    server = _make_server(monkeypatch, read_enabled=True)

    class _Store:
        def __init__(self) -> None:
            self.saved = []

        def get_stats(self):
            return {"entry_count": 2, "max_entries": 500}

        def exists(self, hash_key):
            return True

        def store(self, **kwargs):
            self.saved.append(kwargs)
            return "hash-123"

    store = _Store()
    server._local_store = store
    server._stats.total_input_tokens = 100
    server._stats.total_output_tokens = 40
    server._stats.total_tokens_saved = 60
    server._stats.compressions = 2

    monkeypatch.setattr(
        mcp_server,
        "_read_shared_events",
        lambda: [
            {"type": "compress", "pid": 999, "input_tokens": 100, "output_tokens": 40},
            {"type": "retrieve", "pid": 999},
        ],
    )
    server._fetch_full_proxy_stats = lambda: _return_async(  # type: ignore[method-assign]
        {
            "summary": {
                "mode": "token",
                "api_requests": 5,
                "primary_model": "claude",
                "compression": {
                    "requests_compressed": 1,
                    "avg_compression_pct": 20,
                    "total_tokens_removed": 50,
                },
            }
        }
    )

    formatted = await server._handle_stats()
    assert "Headroom Session Summary" in formatted[0].text

    server._fetch_full_proxy_stats = lambda: _return_async(  # type: ignore[method-assign]
        {
            "requests_total": 9,
            "tokens_saved_total": 1234,
            "caching": {"cache_hits": 4, "cache_misses": 2, "cache_hit_rate": 66},
            "cost": {"saved": 0.75},
        }
    )
    raw_stats = await server._handle_stats()
    payload = json.loads(raw_stats[0].text)
    assert payload["sub_agents"]["compressions"] == 1
    assert payload["combined"]["total_compressions"] == 3
    assert payload["proxy"]["cache"]["hits"] == 4

    file_path = tmp_path / "example.txt"
    file_path.write_text("first line\nsecond line", encoding="utf-8")

    fresh = await server._handle_read({"file_path": str(file_path)})
    assert "first line" in fresh[0].text
    assert str(file_path.resolve()) in server._file_cache

    cached = await server._handle_read({"file_path": str(file_path)})
    cached_payload = json.loads(cached[0].text)
    assert cached_payload["status"] == "cached"
    assert cached_payload["hash"] == "hash-123"

    missing = await server._handle_read({"file_path": str(tmp_path / "missing.txt")})
    assert "File not found" in missing[0].text

    not_file = await server._handle_read({"file_path": str(tmp_path)})
    assert "Not a file" in not_file[0].text


async def _return_async(value):
    return value
