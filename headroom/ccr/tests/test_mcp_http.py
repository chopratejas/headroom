"""Tests for MCP Streamable HTTP transport (mcp_http.py).

Covers:
    - create_mcp_http_server creates a FastMCP instance
    - Stateless HTTP mode
    - Tools are registered (3 tools)
    - Tool invocation through FastMCP (compress, retrieve, stats)
    - run_standalone function
    - Proxy fallback config
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# create_mcp_http_server
# ============================================================


class TestCreateMCPHTTPServer:

    def test_creates_fastmcp_instance(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(proxy_url="http://localhost:8787", check_proxy=False)
        assert mcp is not None
        assert mcp.name == "Headroom MCP"

    def test_stateless_mode(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        assert mcp.settings.stateless_http is True

    def test_custom_proxy_url(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(proxy_url="http://custom:9999", check_proxy=False)
        assert mcp is not None

    def test_three_tools_registered(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]

        assert "headroom_compress" in names
        assert "headroom_retrieve" in names
        assert "headroom_stats" in names
        assert len(tools) == 3


# ============================================================
# Tool: headroom_compress
# ============================================================


class TestHTTPCompressTool:

    async def test_compress_returns_json(self, mock_compress_result, mock_store):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)

        with (
            patch("headroom.compress.compress", return_value=mock_compress_result),
            patch("headroom.cache.compression_store.CompressionStore", return_value=mock_store),
            patch("headroom.ccr.mcp_server._append_shared_event"),
        ):
            result = await mcp._tool_manager.call_tool("headroom_compress", {"content": "large data"})

        data = json.loads(result)
        assert "hash" in data
        assert data["original_tokens"] == 5000
        assert data["compressed_tokens"] == 1000

    async def test_compress_empty_content(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        result = await mcp._tool_manager.call_tool("headroom_compress", {"content": ""})
        data = json.loads(result)
        assert "error" in data


# ============================================================
# Tool: headroom_retrieve
# ============================================================


class TestHTTPRetrieveTool:

    async def test_retrieve_missing_hash(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        result = await mcp._tool_manager.call_tool("headroom_retrieve", {"hash": ""})
        data = json.loads(result)
        assert "error" in data

    async def test_retrieve_not_found(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        result = await mcp._tool_manager.call_tool("headroom_retrieve", {"hash": "nonexistent"})
        data = json.loads(result)
        assert "error" in data or "hash" in data


# ============================================================
# Tool: headroom_stats
# ============================================================


class TestHTTPStatsTool:

    async def test_stats_returns_json(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        result = await mcp._tool_manager.call_tool("headroom_stats", {})
        data = json.loads(result)
        assert "compressions" in data
        assert "retrievals" in data
        assert data["compressions"] == 0


# ============================================================
# run_standalone
# ============================================================


class TestRunStandalone:

    def test_function_exists(self):
        from headroom.ccr.mcp_http import run_standalone

        assert callable(run_standalone)

    def test_create_and_run_pattern(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)

        with patch.object(mcp, "run") as mock_run:
            mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)

        mock_run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",
            port=8080,
        )


# ============================================================
# Proxy Config
# ============================================================


class TestProxyConfig:

    def test_check_proxy_false(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(check_proxy=False)
        assert mcp is not None

    def test_check_proxy_true(self):
        from headroom.ccr.mcp_http import create_mcp_http_server

        mcp = create_mcp_http_server(proxy_url="http://proxy:8787", check_proxy=True)
        assert mcp is not None
