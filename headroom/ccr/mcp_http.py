"""Streamable HTTP transport for Headroom's MCP server.

Provides Headroom's compression tools (headroom_compress, headroom_retrieve,
headroom_stats) over MCP's Streamable HTTP protocol, enabling remote access
from agents running on different machines.

Two modes:
    1. Standalone: ``headroom mcp serve --transport http --port 8080``
    2. Embedded in proxy: mounted at ``/mcp`` on the existing proxy (port 8787)

Uses FastMCP from the ``mcp`` package with ``stateless_http=True``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger("headroom.ccr.mcp_http")

# Tool names (shared with mcp_server.py)
COMPRESS_TOOL_NAME = "headroom_compress"
CCR_TOOL_NAME = "headroom_retrieve"
STATS_TOOL_NAME = "headroom_stats"

DEFAULT_PROXY_URL = os.environ.get("HEADROOM_PROXY_URL", "http://127.0.0.1:8787")


def create_mcp_http_server(
    proxy_url: str = DEFAULT_PROXY_URL,
    check_proxy: bool = True,
) -> Any:
    """Create a FastMCP server with Headroom's tools for streamable HTTP transport.

    Returns a FastMCP instance that can be:
    - Run standalone: ``mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)``
    - Mounted in another app: ``mcp.streamable_http_app()`` (if available)
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError("MCP SDK not installed. Install with: pip install 'headroom-ai[mcp]'")

    mcp = FastMCP(
        "Headroom MCP",
        stateless_http=True,
    )

    # Lazy-initialized shared state
    _state: dict[str, Any] = {
        "store": None,
        "http_client": None,
        "stats": None,
    }

    def _get_store() -> Any:
        """Get or create the compression store."""
        if _state["store"] is None:
            from headroom.cache.compression_store import CompressionStore

            _state["store"] = CompressionStore(max_entries=500, default_ttl=3600)
        return _state["store"]

    def _get_stats() -> Any:
        """Get or create session stats tracker."""
        if _state["stats"] is None:
            from headroom.ccr.mcp_server import SessionStats

            _state["stats"] = SessionStats()
        return _state["stats"]

    async def _get_http_client() -> Any:
        """Get or create HTTP client for proxy communication."""
        if _state["http_client"] is None:
            try:
                import httpx

                _state["http_client"] = httpx.AsyncClient(timeout=15.0)
            except ImportError:
                return None
        return _state["http_client"]

    # --- Tool: headroom_compress ---

    @mcp.tool(
        name=COMPRESS_TOOL_NAME,
        description=(
            "Compress content to save context window space. "
            "Use this on large tool outputs, file contents, search results, "
            "or any content you want to shrink before reasoning over it. "
            "The original is stored and can be retrieved later via headroom_retrieve. "
            "Returns compressed text + a hash for retrieval."
        ),
    )
    async def headroom_compress(content: str) -> str:
        """Compress content and store original for later retrieval."""
        if not content:
            return json.dumps({"error": "content parameter is required"})

        from headroom.compress import compress

        loop = asyncio.get_running_loop()

        def _do_compress() -> dict[str, Any]:
            messages = [{"role": "tool", "content": content}]
            result = compress(messages, model="claude-sonnet-4-5-20250929")

            compressed_content = result.messages[0].get("content", content)
            input_tokens = result.tokens_before
            output_tokens = result.tokens_after

            store = _get_store()
            hash_key = store.store(
                original=content,
                compressed=compressed_content
                if isinstance(compressed_content, str)
                else json.dumps(compressed_content),
                original_tokens=input_tokens,
                compressed_tokens=output_tokens,
                compression_strategy="mcp_compress",
                ttl=3600,
            )

            stats = _get_stats()
            strategy = (
                ", ".join(result.transforms_applied)
                if result.transforms_applied
                else "passthrough"
            )
            stats.record_compression(input_tokens, output_tokens, strategy)

            savings_pct = (
                round((1 - result.compression_ratio) * 100, 1)
                if result.compression_ratio < 1.0
                else 0
            )

            return {
                "compressed": compressed_content,
                "hash": hash_key,
                "original_tokens": input_tokens,
                "compressed_tokens": output_tokens,
                "tokens_saved": max(0, input_tokens - output_tokens),
                "savings_percent": savings_pct,
                "transforms": result.transforms_applied,
                "note": f"Original stored with hash={hash_key}. Use headroom_retrieve to get full content later.",
            }

        result = await loop.run_in_executor(None, _do_compress)
        return json.dumps(result, indent=2)

    # --- Tool: headroom_retrieve ---

    @mcp.tool(
        name=CCR_TOOL_NAME,
        description=(
            "Retrieve original uncompressed content by hash. "
            "Use this when you need full details from previously compressed content. "
            "The hash comes from headroom_compress results or from compression "
            "markers like [N items compressed... hash=abc123]."
        ),
    )
    async def headroom_retrieve(hash: str, query: str | None = None) -> str:
        """Retrieve original content by hash, optionally filtering with a query."""
        if not hash:
            return json.dumps({"error": "hash parameter is required"})

        # Check local store first
        store = _get_store()
        if query:
            results = store.search(hash, query)
            if results:
                _get_stats().record_retrieval(hash)
                return json.dumps(
                    {
                        "hash": hash,
                        "source": "local",
                        "query": query,
                        "results": results,
                        "count": len(results),
                    },
                    indent=2,
                )
        else:
            entry = store.retrieve(hash)
            if entry:
                _get_stats().record_retrieval(hash)
                return json.dumps(
                    {
                        "hash": hash,
                        "source": "local",
                        "original_content": entry.original_content,
                        "original_item_count": entry.original_item_count,
                        "compressed_item_count": entry.compressed_item_count,
                        "retrieval_count": entry.retrieval_count,
                    },
                    indent=2,
                )

        # Fall back to proxy
        if check_proxy:
            client = await _get_http_client()
            if client:
                try:
                    payload: dict[str, str] = {"hash": hash}
                    if query:
                        payload["query"] = query
                    response = await client.post(f"{proxy_url}/v1/retrieve", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        result["source"] = "proxy"
                        _get_stats().record_retrieval(hash)
                        return json.dumps(result, indent=2)
                except Exception:
                    pass

        return json.dumps(
            {
                "error": "Content not found. It may have expired or the hash may be incorrect.",
                "hash": hash,
                "hint": "Content compressed via headroom_compress is stored for the session. "
                "Content compressed by the proxy has a shorter TTL (5 minutes).",
            },
            indent=2,
        )

    # --- Tool: headroom_stats ---

    @mcp.tool(
        name=STATS_TOOL_NAME,
        description=(
            "Show compression statistics for this session: "
            "total compressions, tokens saved, estimated cost savings, "
            "and recent compression events."
        ),
    )
    async def headroom_stats() -> str:
        """Return session compression statistics."""
        stats = _get_stats().to_dict()

        if _state["store"] is not None:
            store_stats = _state["store"].get_stats()
            stats["store"] = {
                "entries": store_stats.get("entry_count", 0),
                "max_entries": store_stats.get("max_entries", 0),
            }

        # Try to get proxy stats
        if check_proxy:
            client = await _get_http_client()
            if client:
                try:
                    response = await client.get(f"{proxy_url}/stats")
                    if response.status_code == 200:
                        proxy_data = response.json()
                        tokens = proxy_data.get("tokens", {})
                        stats["proxy"] = {
                            "requests": proxy_data.get("requests", {}).get("total", 0),
                            "tokens_saved": tokens.get("saved", 0),
                            "savings_percent": tokens.get("savings_percent", 0),
                        }
                except Exception:
                    pass

        return json.dumps(stats, indent=2)

    return mcp


def run_standalone(
    host: str = "0.0.0.0",
    port: int = 8080,
    proxy_url: str = DEFAULT_PROXY_URL,
    check_proxy: bool = True,
) -> None:
    """Run the MCP HTTP server standalone (not embedded in proxy)."""
    mcp = create_mcp_http_server(proxy_url=proxy_url, check_proxy=check_proxy)
    logger.info(f"Starting Headroom MCP HTTP server on {host}:{port}/mcp")
    mcp.run(transport="streamable-http", host=host, port=port)
