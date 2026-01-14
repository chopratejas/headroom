"""CCR MCP Server - Exposes headroom_retrieve as an MCP tool.

This MCP server allows LLMs to retrieve compressed content via MCP instead
of through injected tool definitions. It connects to the Headroom proxy's
CompressionStore to serve retrieval requests.

Usage:
    # As standalone server (stdio transport)
    python -m headroom.ccr.mcp_server

    # With custom proxy URL
    python -m headroom.ccr.mcp_server --proxy-url http://localhost:8787

    # Add to Claude Code's MCP config (~/.claude/mcp.json):
    {
        "mcpServers": {
            "headroom": {
                "command": "python",
                "args": ["-m", "headroom.ccr.mcp_server"]
            }
        }
    }

When MCP is configured, the proxy will detect the tool is already present
and skip tool injection, avoiding duplicate tools.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    stdio_server = None

# Try to import httpx for proxy communication
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from .tool_injection import CCR_TOOL_NAME

logger = logging.getLogger("headroom.ccr.mcp")

# Default proxy URL (can be overridden via env or args)
DEFAULT_PROXY_URL = os.environ.get("HEADROOM_PROXY_URL", "http://127.0.0.1:8787")


class CCRMCPServer:
    """MCP Server that exposes headroom_retrieve tool.

    This server can operate in two modes:
    1. HTTP mode: Calls the proxy's /v1/retrieve endpoint (default)
    2. Direct mode: Uses CompressionStore directly (same process)

    HTTP mode is recommended as it ensures consistency with the proxy.
    """

    def __init__(
        self,
        proxy_url: str = DEFAULT_PROXY_URL,
        direct_mode: bool = False,
    ):
        """Initialize CCR MCP Server.

        Args:
            proxy_url: URL of the Headroom proxy server.
            direct_mode: If True, access CompressionStore directly instead of via HTTP.
        """
        self.proxy_url = proxy_url
        self.direct_mode = direct_mode
        self._http_client: httpx.AsyncClient | None = None

        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not installed. Install with: pip install mcp")

        if not direct_mode and not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx not installed (required for HTTP mode). Install with: pip install httpx"
            )

        self.server = Server("headroom-ccr")
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return available tools."""
            return [
                Tool(
                    name=CCR_TOOL_NAME,
                    description=(
                        "Retrieve original uncompressed content that was compressed "
                        "to save tokens. Use this when you need more data than what's "
                        "shown in compressed tool results. The hash is provided in "
                        "compression markers like [N items compressed... hash=abc123]."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hash": {
                                "type": "string",
                                "description": "Hash key from the compression marker (e.g., 'abc123' from hash=abc123)",
                            },
                            "query": {
                                "type": "string",
                                "description": (
                                    "Optional search query to filter results. "
                                    "If provided, only returns items matching the query. "
                                    "If omitted, returns all original items."
                                ),
                            },
                        },
                        "required": ["hash"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            if name != CCR_TOOL_NAME:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": f"Unknown tool: {name}"}),
                    )
                ]

            hash_key = arguments.get("hash")
            query = arguments.get("query")

            if not hash_key:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "hash parameter is required"}),
                    )
                ]

            # Retrieve content
            try:
                if self.direct_mode:
                    result = await self._retrieve_direct(hash_key, query)
                else:
                    result = await self._retrieve_via_proxy(hash_key, query)

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}),
                    )
                ]

    async def _retrieve_via_proxy(
        self,
        hash_key: str,
        query: str | None,
    ) -> dict[str, Any]:
        """Retrieve content via proxy's HTTP endpoint."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        url = f"{self.proxy_url}/v1/retrieve"
        payload = {"hash": hash_key}
        if query:
            payload["query"] = query

        response = await self._http_client.post(url, json=payload)

        if response.status_code == 404:
            return {
                "error": "Entry not found or expired (TTL: 5 minutes)",
                "hash": hash_key,
            }

        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def _retrieve_direct(
        self,
        hash_key: str,
        query: str | None,
    ) -> dict[str, Any]:
        """Retrieve content directly from CompressionStore."""
        from headroom.cache.compression_store import get_compression_store

        store = get_compression_store()

        if query:
            results = store.search(hash_key, query)
            return {
                "hash": hash_key,
                "query": query,
                "results": results,
                "count": len(results),
            }
        else:
            entry = store.retrieve(hash_key)
            if entry:
                return {
                    "hash": hash_key,
                    "original_content": entry.original_content,
                    "original_item_count": entry.original_item_count,
                    "compressed_item_count": entry.compressed_item_count,
                    "retrieval_count": entry.retrieval_count,
                }
            return {
                "error": "Entry not found or expired (TTL: 5 minutes)",
                "hash": hash_key,
            }

    async def run_stdio(self):
        """Run the server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            logger.info(f"CCR MCP Server starting (proxy: {self.proxy_url})")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def cleanup(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()


def create_ccr_mcp_server(
    proxy_url: str = DEFAULT_PROXY_URL,
    direct_mode: bool = False,
) -> CCRMCPServer:
    """Create a CCR MCP server instance.

    Args:
        proxy_url: URL of the Headroom proxy server.
        direct_mode: If True, access CompressionStore directly.

    Returns:
        CCRMCPServer instance.

    Example:
        ```python
        server = create_ccr_mcp_server()
        await server.run_stdio()
        ```
    """
    return CCRMCPServer(proxy_url=proxy_url, direct_mode=direct_mode)


async def main():
    """Run the CCR MCP server."""
    parser = argparse.ArgumentParser(
        description="CCR MCP Server - Retrieve compressed content via MCP"
    )
    parser.add_argument(
        "--proxy-url",
        default=DEFAULT_PROXY_URL,
        help=f"Headroom proxy URL (default: {DEFAULT_PROXY_URL})",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct CompressionStore access instead of HTTP",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    server = create_ccr_mcp_server(
        proxy_url=args.proxy_url,
        direct_mode=args.direct,
    )

    try:
        await server.run_stdio()
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
