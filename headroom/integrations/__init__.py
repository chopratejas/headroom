"""Headroom integrations with popular LLM frameworks.

Available integrations:
- LangChain: HeadroomChatModel, HeadroomCallbackHandler, optimize_messages
- MCP: HeadroomMCPCompressor, compress_tool_result, HeadroomMCPClientWrapper

Install LangChain support: pip install headroom[langchain]
"""

from .langchain import (
    HeadroomChatModel,
    HeadroomCallbackHandler,
    optimize_messages,
    HeadroomRunnable,
)

from .mcp import (
    HeadroomMCPCompressor,
    HeadroomMCPClientWrapper,
    MCPCompressionResult,
    MCPToolProfile,
    compress_tool_result,
    compress_tool_result_with_metrics,
    create_headroom_mcp_proxy,
    DEFAULT_MCP_PROFILES,
)

__all__ = [
    # LangChain
    "HeadroomChatModel",
    "HeadroomCallbackHandler",
    "optimize_messages",
    "HeadroomRunnable",
    # MCP
    "HeadroomMCPCompressor",
    "HeadroomMCPClientWrapper",
    "MCPCompressionResult",
    "MCPToolProfile",
    "compress_tool_result",
    "compress_tool_result_with_metrics",
    "create_headroom_mcp_proxy",
    "DEFAULT_MCP_PROFILES",
]
