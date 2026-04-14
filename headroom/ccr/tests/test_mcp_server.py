"""Tests for HeadroomMCPServer — stdio-based MCP server.

Covers:
    - Server initialization
    - SessionStats tracking
    - Tool handlers (compress, retrieve, stats)
    - Error handling
    - Shared stats (cross-process)
    - Factory function
    - run_http method
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# SessionStats
# ============================================================


class TestSessionStats:

    def test_initial_state(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        assert stats.compressions == 0
        assert stats.retrievals == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_tokens_saved == 0

    def test_record_compression(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        with patch("headroom.ccr.mcp_server._append_shared_event"):
            stats.record_compression(1000, 200, "smart_crusher")

        assert stats.compressions == 1
        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 200
        assert stats.total_tokens_saved == 800

    def test_record_multiple_compressions(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        with patch("headroom.ccr.mcp_server._append_shared_event"):
            stats.record_compression(1000, 200, "smart_crusher")
            stats.record_compression(2000, 500, "passthrough")

        assert stats.compressions == 2
        assert stats.total_input_tokens == 3000
        assert stats.total_output_tokens == 700
        assert stats.total_tokens_saved == 2300

    def test_record_retrieval(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        with patch("headroom.ccr.mcp_server._append_shared_event"):
            stats.record_retrieval("abc123")

        assert stats.retrievals == 1
        assert len(stats.events) == 1
        assert stats.events[0]["type"] == "retrieve"

    def test_to_dict(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        with patch("headroom.ccr.mcp_server._append_shared_event"):
            stats.record_compression(5000, 1000, "smart_crusher")
            stats.record_retrieval("hash1")

        d = stats.to_dict()
        assert d["compressions"] == 1
        assert d["retrievals"] == 1
        assert d["total_tokens_saved"] == 4000
        assert d["savings_percent"] == 80.0
        assert "session_duration_seconds" in d

    def test_events_capped_at_50(self):
        from headroom.ccr.mcp_server import SessionStats

        stats = SessionStats()
        with patch("headroom.ccr.mcp_server._append_shared_event"):
            for i in range(60):
                stats.record_compression(100, 50, f"s{i}")

        assert stats.compressions == 60
        assert len(stats.events) == 50


# ============================================================
# HeadroomMCPServer — Initialization
# ============================================================


class TestServerInit:

    def test_creates_server_instance(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        server = HeadroomMCPServer(proxy_url="http://localhost:8787", check_proxy=False)
        assert server.proxy_url == "http://localhost:8787"
        assert server.check_proxy is False
        assert server.server is not None

    def test_default_proxy_url(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer, DEFAULT_PROXY_URL

        server = HeadroomMCPServer(check_proxy=False)
        assert server.proxy_url == DEFAULT_PROXY_URL

    def test_lazy_store_initialization(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        server = HeadroomMCPServer(check_proxy=False)
        assert server._local_store is None
        store = server._get_local_store()
        assert store is not None
        assert server._local_store is store

    def test_has_run_stdio(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        server = HeadroomMCPServer(check_proxy=False)
        assert hasattr(server, "run_stdio")
        assert asyncio.iscoroutinefunction(server.run_stdio)

    def test_has_run_http(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        server = HeadroomMCPServer(check_proxy=False)
        assert hasattr(server, "run_http")
        assert callable(server.run_http)

    def test_has_cleanup(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        server = HeadroomMCPServer(check_proxy=False)
        assert asyncio.iscoroutinefunction(server.cleanup)


# ============================================================
# HeadroomMCPServer — headroom_compress
# ============================================================


class TestCompressTool:

    @pytest.fixture
    def server(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        return HeadroomMCPServer(check_proxy=False)

    async def test_compress_returns_result(self, server, mock_compress_result, mock_store):
        with (
            patch("headroom.compress.compress", return_value=mock_compress_result),
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            result = await server._handle_compress({"content": "large content"})

        data = json.loads(result[0].text)
        assert data["hash"] == "abc123hash"
        assert data["original_tokens"] == 5000
        assert data["compressed_tokens"] == 1000
        assert data["tokens_saved"] == 4000

    async def test_compress_missing_content(self, server):
        result = await server._handle_compress({})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_compress_empty_content(self, server):
        result = await server._handle_compress({"content": ""})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_compress_tracks_stats(self, server, mock_compress_result, mock_store):
        with (
            patch("headroom.compress.compress", return_value=mock_compress_result),
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            await server._handle_compress({"content": "test"})

        assert server._stats.compressions == 1
        assert server._stats.total_tokens_saved == 4000

    async def test_compress_stores_in_local_store(self, server, mock_compress_result, mock_store):
        with (
            patch("headroom.compress.compress", return_value=mock_compress_result),
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            await server._handle_compress({"content": "store this"})

        mock_store.store.assert_called_once()


# ============================================================
# HeadroomMCPServer — headroom_retrieve
# ============================================================


class TestRetrieveTool:

    @pytest.fixture
    def server(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        return HeadroomMCPServer(check_proxy=False)

    async def test_retrieve_from_local_store(self, server, mock_store):
        with (
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            result = await server._handle_retrieve({"hash": "abc123"})

        data = json.loads(result[0].text)
        assert data["source"] == "local"
        assert data["hash"] == "abc123"
        assert data["original_content"] == "full original content here"

    async def test_retrieve_with_query(self, server, mock_store):
        with (
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            result = await server._handle_retrieve({"hash": "abc123", "query": "error"})

        data = json.loads(result[0].text)
        assert data["source"] == "local"
        assert data["query"] == "error"
        assert data["count"] == 1

    async def test_retrieve_missing_hash(self, server):
        result = await server._handle_retrieve({})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_retrieve_not_found(self, server):
        store = MagicMock()
        store.retrieve.return_value = None
        store.search.return_value = []

        with patch.object(server, "_get_local_store", return_value=store):
            result = await server._handle_retrieve({"hash": "gone"})

        data = json.loads(result[0].text)
        assert "error" in data

    async def test_retrieve_tracks_stats(self, server, mock_store):
        with (
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            await server._handle_retrieve({"hash": "abc123"})

        assert server._stats.retrievals == 1


# ============================================================
# HeadroomMCPServer — headroom_stats
# ============================================================


class TestStatsTool:

    @pytest.fixture
    def server(self):
        from headroom.ccr.mcp_server import HeadroomMCPServer

        return HeadroomMCPServer(check_proxy=False)

    async def test_stats_empty_session(self, server):
        with patch("headroom.ccr.mcp_server._read_shared_events", return_value=[]):
            result = await server._handle_stats()

        data = json.loads(result[0].text)
        assert data["compressions"] == 0
        assert data["retrievals"] == 0

    async def test_stats_after_compression(self, server, mock_compress_result, mock_store):
        with (
            patch("headroom.compress.compress", return_value=mock_compress_result),
            patch.object(server, "_get_local_store", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            await server._handle_compress({"content": "test"})

        with patch("headroom.ccr.mcp_server._read_shared_events", return_value=[]):
            result = await server._handle_stats()

        data = json.loads(result[0].text)
        assert data["compressions"] == 1
        assert data["total_tokens_saved"] == 4000

    async def test_stats_includes_store_info(self, server, mock_store):
        server._local_store = mock_store

        with patch("headroom.ccr.mcp_server._read_shared_events", return_value=[]):
            result = await server._handle_stats()

        data = json.loads(result[0].text)
        assert "store" in data
        assert data["store"]["entries"] == 5


# ============================================================
# Shared Stats
# ============================================================


class TestSharedStats:

    def test_append_and_read(self, monkeypatch, tmp_path):
        from headroom.ccr.mcp_server import _append_shared_event, _read_shared_events

        stats_file = tmp_path / "session_stats.jsonl"
        monkeypatch.setattr("headroom.ccr.mcp_server.SHARED_STATS_FILE", stats_file)
        monkeypatch.setattr("headroom.ccr.mcp_server.SHARED_STATS_DIR", tmp_path)
        monkeypatch.setattr("os.getpid", lambda: 12345)
        monkeypatch.setattr("time.time", lambda: 1000.0)

        _append_shared_event({"type": "compress", "input_tokens": 500})

        events = _read_shared_events(window_seconds=7200)
        assert len(events) == 1
        assert events[0]["type"] == "compress"
        assert events[0]["pid"] == 12345

    def test_prunes_old_events(self, monkeypatch, tmp_path):
        from headroom.ccr.mcp_server import _append_shared_event, _read_shared_events

        stats_file = tmp_path / "session_stats.jsonl"
        monkeypatch.setattr("headroom.ccr.mcp_server.SHARED_STATS_FILE", stats_file)
        monkeypatch.setattr("headroom.ccr.mcp_server.SHARED_STATS_DIR", tmp_path)
        monkeypatch.setattr("os.getpid", lambda: 1)

        # Write old event directly to file
        old_event = json.dumps({"timestamp": 100.0, "pid": 1, "type": "compress"})
        new_event = json.dumps({"timestamp": 9000.0, "pid": 1, "type": "retrieve"})
        stats_file.write_text(old_event + "\n" + new_event + "\n")

        monkeypatch.setattr("time.time", lambda: 9000.0)
        events = _read_shared_events(window_seconds=100)
        assert len(events) == 1
        assert events[0]["type"] == "retrieve"


# ============================================================
# Factory
# ============================================================


class TestFactory:

    def test_creates_server(self):
        from headroom.ccr.mcp_server import create_ccr_mcp_server

        server = create_ccr_mcp_server(proxy_url="http://test:8787")
        assert server.proxy_url == "http://test:8787"

    def test_default_proxy_url(self):
        from headroom.ccr.mcp_server import create_ccr_mcp_server, DEFAULT_PROXY_URL

        server = create_ccr_mcp_server()
        assert server.proxy_url == DEFAULT_PROXY_URL
