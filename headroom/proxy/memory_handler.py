"""Memory integration handler for the proxy server.

This module provides memory capabilities for the Headroom proxy:
1. MemoryHandler - Unified handler for memory operations
   - inject_tools() - Add memory tools to requests
   - search_and_format_context() - Search memories, format for injection
   - has_memory_tool_calls() - Detect memory tool usage in response
   - handle_memory_tool_calls() - Execute tools, return results

Usage:
    config = MemoryConfig(enabled=True, backend="local")
    handler = MemoryHandler(config)

    # Inject tools into request
    tools, was_injected = handler.inject_tools(existing_tools, "anthropic")

    # Search and inject context
    context = await handler.search_and_format_context(user_id, messages)

    # Handle tool calls in response
    if handler.has_memory_tool_calls(response, "anthropic"):
        results = await handler.handle_memory_tool_calls(response, user_id, "anthropic")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)

# Memory tool names for detection (Headroom's custom tools)
MEMORY_TOOL_NAMES = {"memory_save", "memory_search", "memory_update", "memory_delete"}

# Anthropic's native memory tool name
NATIVE_MEMORY_TOOL_NAME = "memory"

# Beta header required for native memory tool
NATIVE_MEMORY_BETA_HEADER = "context-management-2025-06-27"

# Native memory tool type
NATIVE_MEMORY_TOOL_TYPE = "memory_20250818"


@dataclass
class MemoryConfig:
    """Configuration for memory handler."""

    enabled: bool = False
    backend: Literal["local", "qdrant-neo4j"] = "local"
    db_path: str = "headroom_memory.db"
    inject_tools: bool = True
    inject_context: bool = True
    top_k: int = 10
    min_similarity: float = 0.3
    # Native memory tool (Anthropic's built-in memory_20250818)
    use_native_tool: bool = False
    native_memory_dir: str = ""  # Directory for native memory files (default: ~/.headroom/memories)
    # Qdrant+Neo4j config
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    # Memory Bridge (bidirectional markdown <-> Headroom sync)
    bridge_enabled: bool = False
    bridge_md_paths: list[str] = field(default_factory=list)
    bridge_md_format: str = "auto"
    bridge_auto_import: bool = False
    bridge_export_path: str = ""


class MemoryHandler:
    """Unified handler for memory operations in the proxy.

    Responsibilities:
    1. Initialize and manage memory backend
    2. Inject memory tools into requests
    3. Search and inject relevant memories as context
    4. Handle memory tool calls in responses

    Supports two modes:
    - Custom tools: Headroom's memory_save, memory_search, etc. (default)
    - Native tool: Anthropic's memory_20250818 built-in tool (experimental)
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._backend: LocalBackend | Any = None
        self._initialized = False
        self._memory_tools: list[dict[str, Any]] | None = None
        # Native memory tool directory
        self._native_memory_dir: Path | None = None
        if config.use_native_tool:
            self._init_native_memory_dir()
        # Memory Bridge
        self._bridge: Any = None  # MemoryBridge, lazy imported

    def _init_native_memory_dir(self) -> None:
        """Initialize native memory directory."""
        if self.config.native_memory_dir:
            self._native_memory_dir = Path(self.config.native_memory_dir)
        else:
            # Default: ~/.headroom/memories
            self._native_memory_dir = Path.home() / ".headroom" / "memories"

        # Create directory if it doesn't exist
        self._native_memory_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Memory: Native memory directory: {self._native_memory_dir}")

    def get_beta_headers(self) -> dict[str, str]:
        """Get beta headers required for native memory tool.

        Returns:
            Dict with beta headers to add to request, or empty dict.
        """
        if self.config.use_native_tool and self.config.inject_tools:
            return {"anthropic-beta": NATIVE_MEMORY_BETA_HEADER}
        return {}

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of memory backend."""
        if self._initialized:
            return

        if not self.config.enabled:
            return

        if self.config.backend == "local":
            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

            backend_config = LocalBackendConfig(db_path=self.config.db_path)
            self._backend = LocalBackend(backend_config)
            await self._backend._ensure_initialized()
            logger.info(f"Memory: Initialized LocalBackend at {self.config.db_path}")

        elif self.config.backend == "qdrant-neo4j":
            try:
                from headroom.memory.backends.direct_mem0 import (
                    DirectMem0Adapter,
                    Mem0Config,
                )

                mem0_config = Mem0Config(
                    qdrant_host=self.config.qdrant_host,
                    qdrant_port=self.config.qdrant_port,
                    neo4j_uri=self.config.neo4j_uri,
                    neo4j_user=self.config.neo4j_user,
                    neo4j_password=self.config.neo4j_password,
                    enable_graph=True,
                )
                self._backend = DirectMem0Adapter(mem0_config)
                logger.info(
                    f"Memory: Initialized Qdrant+Neo4j backend "
                    f"({self.config.qdrant_host}:{self.config.qdrant_port})"
                )
            except ImportError as e:
                logger.error(
                    f"Memory: Failed to import qdrant-neo4j dependencies: {e}. "
                    "Install with: pip install mem0ai qdrant-client neo4j"
                )
                raise
        else:
            raise ValueError(f"Unknown memory backend: {self.config.backend}")

        self._initialized = True

        # Auto-import from Memory Bridge if configured
        if self.config.bridge_enabled and self.config.bridge_auto_import:
            await self._init_and_import_bridge()

    async def _init_and_import_bridge(self) -> None:
        """Initialize the Memory Bridge and run auto-import."""
        if self._bridge is not None:
            return
        try:
            from headroom.memory.bridge import MemoryBridge
            from headroom.memory.bridge_config import BridgeConfig, MarkdownFormat

            bridge_config = BridgeConfig(
                md_paths=[Path(p) for p in self.config.bridge_md_paths],
                md_format=MarkdownFormat(self.config.bridge_md_format),
                auto_import_on_startup=True,
                export_path=Path(self.config.bridge_export_path)
                if self.config.bridge_export_path
                else None,
            )
            self._bridge = MemoryBridge(bridge_config, self._backend)
            stats = await self._bridge.import_from_markdown()
            logger.info(
                f"Memory Bridge: Auto-imported {stats.sections_imported} sections "
                f"({stats.sections_skipped_duplicate} duplicates skipped)"
            )
        except Exception as e:
            logger.warning(f"Memory Bridge: Auto-import failed: {e}")

    def _get_memory_tools(self) -> list[dict[str, Any]]:
        """Get memory tool definitions (cached)."""
        if self._memory_tools is None:
            from headroom.memory.tools import get_memory_tools_optimized

            self._memory_tools = get_memory_tools_optimized()
        return self._memory_tools

    def inject_tools(
        self,
        tools: list[dict[str, Any]] | None,
        provider: str = "anthropic",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject memory tools into tools list.

        Args:
            tools: Existing tools list (may be None).
            provider: Provider for tool format ("anthropic" or "openai").

        Returns:
            Tuple of (updated_tools, was_injected).
        """
        if not self.config.inject_tools:
            return tools or [], False

        tools = list(tools) if tools else []

        # Use native memory tool if configured
        if self.config.use_native_tool:
            return self._inject_native_tool(tools)

        # Check which tools are already present
        existing_names: set[str] = set()
        for tool in tools:
            name = tool.get("name") or tool.get("function", {}).get("name")
            if name:
                existing_names.add(name)

        # Add missing memory tools
        was_injected = False
        for memory_tool in self._get_memory_tools():
            tool_name = memory_tool["function"]["name"]
            if tool_name in existing_names:
                continue

            # Convert to provider format
            if provider == "anthropic":
                tools.append(
                    {
                        "name": tool_name,
                        "description": memory_tool["function"]["description"],
                        "input_schema": memory_tool["function"]["parameters"],
                    }
                )
            else:
                # OpenAI format
                tools.append(memory_tool)

            was_injected = True

        return tools, was_injected

    def _inject_native_tool(self, tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        """Inject Anthropic's native memory tool (memory_20250818).

        This uses Anthropic's built-in memory tool format which may be
        allowed by Claude Code subscription credentials (unlike custom tools).

        Returns:
            Tuple of (updated_tools, was_injected).
        """
        # Check if native memory tool already present
        for tool in tools:
            if tool.get("type") == NATIVE_MEMORY_TOOL_TYPE:
                return tools, False
            if tool.get("name") == NATIVE_MEMORY_TOOL_NAME:
                return tools, False

        # Add native memory tool
        native_tool = {
            "type": NATIVE_MEMORY_TOOL_TYPE,
            "name": NATIVE_MEMORY_TOOL_NAME,
        }
        tools.append(native_tool)

        logger.info(
            f"Memory: Injected native memory tool ({NATIVE_MEMORY_TOOL_TYPE}). "
            f"Beta header required: {NATIVE_MEMORY_BETA_HEADER}"
        )
        return tools, True

    async def search_and_format_context(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Search memories and format as context injection.

        Args:
            user_id: User identifier for memory scoping.
            messages: Conversation messages (used to extract query).

        Returns:
            Formatted context string, or None if no relevant memories.
        """
        if not self.config.inject_context:
            return None

        await self._ensure_initialized()
        if not self._backend:
            return None

        # Extract query from last user message
        query = self._extract_user_query(messages)
        if not query:
            logger.debug("Memory: No user query found for context search")
            return None

        try:
            # Search memories
            results = await self._backend.search_memories(
                query=query,
                user_id=user_id,
                top_k=self.config.top_k,
                include_related=True,
            )

            if not results:
                logger.debug(f"Memory: No memories found for user {user_id}")
                return None

            # Filter by minimum similarity
            filtered_results = [r for r in results if r.score >= self.config.min_similarity]

            if not filtered_results:
                logger.debug(
                    f"Memory: {len(results)} memories found but none above threshold "
                    f"{self.config.min_similarity}"
                )
                return None

            # Format as context
            memory_lines = []
            for i, result in enumerate(filtered_results, 1):
                memory_lines.append(f"{i}. {result.memory.content}")
                if hasattr(result, "related_entities") and result.related_entities:
                    entities_str = ", ".join(result.related_entities[:3])
                    memory_lines.append(f"   (Related: {entities_str})")

            context = f"""## Relevant Memories for This User

The following information was previously saved about this user:

{chr(10).join(memory_lines)}

Use this context to provide personalized and contextually relevant responses."""

            logger.info(
                f"Memory: Injecting {len(filtered_results)} memories "
                f"({len(context)} chars) for user {user_id}"
            )
            return context

        except Exception as e:
            logger.warning(f"Memory: Search failed for user {user_id}: {e}")
            return None

    def _extract_user_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract the user query from the last user message."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")

            if isinstance(content, str):
                return content[:500]  # Limit query length

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text", ""))
                        if text:
                            return text[:500]

        return ""

    def has_memory_tool_calls(
        self,
        response: dict[str, Any],
        provider: str = "anthropic",
    ) -> bool:
        """Check if response contains memory tool calls."""
        tool_calls = self._extract_tool_calls(response, provider)
        for tc in tool_calls:
            name = tc.get("name") or tc.get("function", {}).get("name")
            # Check for both custom and native memory tools
            if name in MEMORY_TOOL_NAMES or name == NATIVE_MEMORY_TOOL_NAME:
                return True
        return False

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from response based on provider format."""
        if provider == "anthropic":
            content = response.get("content", [])
            if isinstance(content, list):
                return [block for block in content if block.get("type") == "tool_use"]
            return []

        elif provider == "openai":
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return list(message.get("tool_calls", []) or [])
            return []

        return []

    async def handle_memory_tool_calls(
        self,
        response: dict[str, Any],
        user_id: str,
        provider: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """Execute memory tool calls and return results.

        Args:
            response: The API response containing tool calls.
            user_id: User identifier for memory operations.
            provider: Provider format ("anthropic" or "openai").

        Returns:
            List of tool results in provider format.
        """
        tool_calls = self._extract_tool_calls(response, provider)
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc.get("name") or tc.get("function", {}).get("name")
            tool_id = tc.get("id", "")

            # Parse input data
            if provider == "anthropic":
                input_data = tc.get("input", {})
            else:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    input_data = json.loads(args_str)
                except json.JSONDecodeError:
                    input_data = {}

            # Handle native memory tool
            if tool_name == NATIVE_MEMORY_TOOL_NAME:
                result_content = await self._execute_native_memory_tool(input_data, user_id)
            elif tool_name in MEMORY_TOOL_NAMES:
                # Custom memory tools need backend
                await self._ensure_initialized()
                if not self._backend:
                    continue
                result_content = await self._execute_memory_tool(tool_name, input_data, user_id)
            else:
                continue

            # Format result based on provider
            if provider == "anthropic":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                )
            else:
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_content,
                    }
                )

            logger.info(f"Memory: Executed {tool_name} for user {user_id}")

        return results

    async def _execute_memory_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        user_id: str,
    ) -> str:
        """Execute a memory tool and return result string."""
        try:
            if tool_name == "memory_save":
                return await self._execute_save(input_data, user_id)
            elif tool_name == "memory_search":
                return await self._execute_search(input_data, user_id)
            elif tool_name == "memory_update":
                return await self._execute_update(input_data, user_id)
            elif tool_name == "memory_delete":
                return await self._execute_delete(input_data, user_id)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error(f"Memory: Tool {tool_name} failed: {e}")
            return json.dumps({"status": "error", "error": str(e)})

    async def _execute_save(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_save tool."""
        content = input_data.get("content", "")
        if not content:
            return json.dumps({"status": "error", "error": "content is required"})

        # Extract parameters
        importance = input_data.get("importance", 0.5)
        facts = input_data.get("facts")
        entities = input_data.get("entities")
        extracted_entities = input_data.get("extracted_entities")
        relationships = input_data.get("relationships")
        extracted_relationships = input_data.get("extracted_relationships")

        # Call backend
        memory = await self._backend.save_memory(
            content=content,
            user_id=user_id,
            importance=importance,
            facts=facts,
            entities=entities,
            extracted_entities=extracted_entities,
            relationships=relationships,
            extracted_relationships=extracted_relationships,
        )

        return json.dumps(
            {
                "status": "saved",
                "memory_id": memory.id,
                "content": (
                    memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
                ),
            }
        )

    async def _execute_search(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_search tool."""
        query = input_data.get("query", "")
        if not query:
            return json.dumps({"status": "error", "error": "query is required"})

        top_k = input_data.get("top_k", 10)
        include_related = input_data.get("include_related", True)
        entities_filter = input_data.get("entities")

        results = await self._backend.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            include_related=include_related,
            entities=entities_filter,
        )

        return json.dumps(
            {
                "status": "found",
                "count": len(results),
                "memories": [
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "score": round(r.score, 3),
                        "entities": (
                            r.related_entities[:5]
                            if hasattr(r, "related_entities") and r.related_entities
                            else []
                        ),
                    }
                    for r in results
                ],
            }
        )

    async def _execute_update(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_update tool."""
        memory_id = input_data.get("memory_id", "")
        new_content = input_data.get("new_content", "")

        if not memory_id:
            return json.dumps({"status": "error", "error": "memory_id is required"})
        if not new_content:
            return json.dumps({"status": "error", "error": "new_content is required"})

        reason = input_data.get("reason")

        # Check if backend has update_memory method
        if hasattr(self._backend, "update_memory"):
            memory = await self._backend.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                reason=reason,
                user_id=user_id,
            )
            return json.dumps({"status": "updated", "memory_id": memory.id})
        else:
            # Fallback: delete old, save new
            await self._backend.delete_memory(memory_id)
            memory = await self._backend.save_memory(
                content=new_content,
                user_id=user_id,
                importance=0.5,
            )
            return json.dumps(
                {
                    "status": "updated",
                    "memory_id": memory.id,
                    "note": "Replaced via delete+save",
                }
            )

    async def _execute_delete(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_delete tool."""
        memory_id = input_data.get("memory_id", "")
        if not memory_id:
            return json.dumps({"status": "error", "error": "memory_id is required"})

        deleted = await self._backend.delete_memory(memory_id)

        return json.dumps(
            {
                "status": "deleted" if deleted else "not_found",
                "memory_id": memory_id,
            }
        )

    # =========================================================================
    # Native Memory Tool (Anthropic's memory_20250818)
    # =========================================================================
    #
    # HYBRID ARCHITECTURE:
    # Claude uses Anthropic's native memory tool interface (file operations),
    # but we translate these to our semantic vector store backend.
    #
    # This gives us:
    # - Native tool format (subscription-safe, approved by Anthropic)
    # - Semantic search (our vector embeddings under the hood)
    # - Best of both worlds
    #
    # Translation mapping:
    #   view /memories              → Show overview + search instructions
    #   view /memories/search/X     → Semantic search for X
    #   view /memories/recent       → Recent memories
    #   view /memories/<path>       → Find memory by path/topic
    #   create /memories/<path>     → Save to vector store (path as tag)
    #   delete /memories/<path>     → Delete from vector store
    #   str_replace                 → Update memory content
    # =========================================================================

    async def _execute_native_memory_tool(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute Anthropic's native memory tool with semantic backend.

        This is a TRANSLATION LAYER: Claude thinks it's doing file operations,
        but we're actually using our semantic vector store.

        Commands:
        - view: Semantic search or list memories
        - create: Save to vector store
        - str_replace: Update memory content
        - insert: Append to memory
        - delete: Remove from vector store
        - rename: Update memory tags/path
        """
        # Ensure our semantic backend is initialized
        await self._ensure_initialized()

        command = input_data.get("command", "")

        try:
            if command == "view":
                return await self._native_view_semantic(input_data, user_id)
            elif command == "create":
                return await self._native_create_semantic(input_data, user_id)
            elif command == "str_replace":
                return await self._native_update_semantic(input_data, user_id)
            elif command == "insert":
                return await self._native_append_semantic(input_data, user_id)
            elif command == "delete":
                return await self._native_delete_semantic(input_data, user_id)
            elif command == "rename":
                return await self._native_rename_semantic(input_data, user_id)
            else:
                return f"Error: Unknown command '{command}'"
        except Exception as e:
            logger.error(f"Memory: Native tool error: {e}")
            return f"Error: {e}"

    def _resolve_native_path(self, path: str, user_id: str) -> Path:
        """Resolve path within user's memory directory safely.

        Prevents path traversal attacks by ensuring path stays within
        the user's memory directory.
        """
        assert self._native_memory_dir is not None

        # User-scoped memory directory
        user_dir = self._native_memory_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Normalize path (remove /memories prefix if present)
        if path.startswith("/memories"):
            path = path[len("/memories") :]
        if path.startswith("/"):
            path = path[1:]

        # Resolve and validate
        resolved = (user_dir / path).resolve()

        # Security: ensure path is within user directory
        try:
            resolved.relative_to(user_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}") from None

        return resolved

    def _native_view(self, input_data: dict[str, Any], user_id: str) -> str:
        """View directory contents or file contents."""
        path = input_data.get("path", "/memories")
        view_range = input_data.get("view_range")

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"The path {path} does not exist. Please provide a valid path."

        if resolved.is_dir():
            # List directory contents
            lines = [
                f"Here're the files and directories up to 2 levels deep in {path}, "
                "excluding hidden items and node_modules:"
            ]

            def get_size(p: Path) -> str:
                if p.is_file():
                    size = p.stat().st_size
                    if size < 1024:
                        return f"{size}B"
                    elif size < 1024 * 1024:
                        return f"{size / 1024:.1f}K"
                    else:
                        return f"{size / (1024 * 1024):.1f}M"
                return "4.0K"  # Default for directories

            def list_recursive(p: Path, rel_path: str, depth: int) -> None:
                if depth > 2:
                    return
                if p.name.startswith(".") or p.name == "node_modules":
                    return

                lines.append(f"{get_size(p)}\t{rel_path}")

                if p.is_dir() and depth < 2:
                    try:
                        for child in sorted(p.iterdir()):
                            child_rel = (
                                f"{rel_path}/{child.name}"
                                if rel_path != path
                                else f"{path}/{child.name}"
                            )
                            list_recursive(child, child_rel, depth + 1)
                    except PermissionError:
                        pass

            list_recursive(resolved, path, 0)
            return "\n".join(lines)

        else:
            # Read file contents with line numbers
            try:
                content = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = resolved.read_text(encoding="latin-1")

            lines_content = content.split("\n")

            if len(lines_content) > 999999:
                return f"File {path} exceeds maximum line limit of 999,999 lines."

            # Apply view_range if specified
            start_line = 1
            end_line = len(lines_content)
            if view_range and len(view_range) >= 2:
                start_line = max(1, view_range[0])
                end_line = min(len(lines_content), view_range[1])

            result_lines = [f"Here's the content of {path} with line numbers:"]
            for i, line in enumerate(lines_content[start_line - 1 : end_line], start=start_line):
                result_lines.append(f"{i:6d}\t{line}")

            return "\n".join(result_lines)

    def _native_create(self, input_data: dict[str, Any], user_id: str) -> str:
        """Create a new file."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if resolved.exists():
            return f"Error: File {path} already exists"

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(file_text, encoding="utf-8")
        logger.info(f"Memory: Native create: {path} for user {user_id}")

        return f"File created successfully at: {path}"

    def _native_str_replace(self, input_data: dict[str, Any], user_id: str) -> str:
        """Replace text in a file."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist. Please provide a valid path."

        if resolved.is_dir():
            return f"Error: The path {path} does not exist. Please provide a valid path."

        content = resolved.read_text(encoding="utf-8")

        # Check for occurrences
        occurrences = content.count(old_str)
        if occurrences == 0:
            return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if occurrences > 1:
            # Find line numbers
            lines = content.split("\n")
            found_lines = []
            for i, line in enumerate(lines, 1):
                if old_str in line:
                    found_lines.append(str(i))
            return (
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines: {', '.join(found_lines)}. Please ensure it is unique"
            )

        # Perform replacement
        new_content = content.replace(old_str, new_str, 1)
        resolved.write_text(new_content, encoding="utf-8")

        # Show snippet around the change
        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if new_str in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet_lines = ["The memory file has been edited."]
                for j in range(start, end):
                    snippet_lines.append(f"{j + 1:6d}\t{lines[j]}")
                return "\n".join(snippet_lines)

        return "The memory file has been edited."

    def _native_insert(self, input_data: dict[str, Any], user_id: str) -> str:
        """Insert text at a specific line."""
        path = input_data.get("path", "")
        insert_line = input_data.get("insert_line", 0)
        insert_text = input_data.get("insert_text", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"

        if resolved.is_dir():
            return f"Error: The path {path} does not exist"

        content = resolved.read_text(encoding="utf-8")
        lines = content.split("\n")
        n_lines = len(lines)

        if insert_line < 0 or insert_line > n_lines:
            return (
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: [0, {n_lines}]"
            )

        # Insert at specified line
        lines.insert(insert_line, insert_text.rstrip("\n"))

        resolved.write_text("\n".join(lines), encoding="utf-8")

        return f"The file {path} has been edited."

    def _native_delete_file(self, input_data: dict[str, Any], user_id: str) -> str:
        """Delete a file or directory."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"

        import shutil

        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

        logger.info(f"Memory: Native delete: {path} for user {user_id}")
        return f"Successfully deleted {path}"

    def _native_rename(self, input_data: dict[str, Any], user_id: str) -> str:
        """Rename or move a file/directory."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"

        resolved_old = self._resolve_native_path(old_path, user_id)
        resolved_new = self._resolve_native_path(new_path, user_id)

        if not resolved_old.exists():
            return f"Error: The path {old_path} does not exist"

        if resolved_new.exists():
            return f"Error: The destination {new_path} already exists"

        # Create parent directory if needed
        resolved_new.parent.mkdir(parents=True, exist_ok=True)

        resolved_old.rename(resolved_new)

        logger.info(f"Memory: Native rename: {old_path} -> {new_path} for user {user_id}")
        return f"Successfully renamed {old_path} to {new_path}"

    # =========================================================================
    # Semantic Translation Methods (Native Tool → Vector Store)
    # =========================================================================

    async def _native_view_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle VIEW command with semantic search capabilities.

        Path patterns:
        - /memories              → Overview + search instructions
        - /memories/search/X     → Semantic search for X
        - /memories/recent       → Recent memories (last 10)
        - /memories/all          → List all memories (paginated)
        - /memories/<topic>      → Search by topic/path
        """
        path = input_data.get("path", "/memories")

        # Normalize path
        if path.startswith("/memories"):
            subpath = path[len("/memories") :].lstrip("/")
        else:
            subpath = path.lstrip("/")

        # CASE 1: /memories/search/<query> → Semantic search
        if subpath.startswith("search/"):
            query = subpath[len("search/") :]
            if not query:
                return "Error: Please provide a search query. Example: view /memories/search/food preferences"
            return await self._semantic_search(query, user_id)

        # CASE 2: /memories/recent → Recent memories
        if subpath == "recent":
            return await self._get_recent_memories(user_id, limit=10)

        # CASE 3: /memories/all → List all (paginated)
        if subpath == "all":
            return await self._list_all_memories(user_id, limit=20)

        # CASE 4: /memories (root) → Overview with instructions
        if not subpath or subpath == "":
            return await self._get_memory_overview(user_id)

        # CASE 5: /memories/<something> → Search by topic
        # Treat the path as a search query
        return await self._semantic_search(subpath.replace("/", " ").replace("_", " "), user_id)

    async def _semantic_search(self, query: str, user_id: str, top_k: int = 5) -> str:
        """Perform semantic search and format results."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query=query,
                user_id=user_id,
                top_k=top_k,
                include_related=True,
            )

            if not results:
                return f"No memories found matching '{query}'.\n\nTip: Try a broader search term, or use 'view /memories/recent' to see recent memories."

            lines = [f"Found {len(results)} memories matching '{query}':\n"]
            for i, r in enumerate(results, 1):
                score_pct = int(r.score * 100)
                content_preview = r.memory.content[:200]
                if len(r.memory.content) > 200:
                    content_preview += "..."

                lines.append(f"{i:6d}\t[{score_pct}% match] {content_preview}")

                # Show related entities if available
                if hasattr(r, "related_entities") and r.related_entities:
                    entities = ", ".join(r.related_entities[:3])
                    lines.append(f"      \t   Related: {entities}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: Semantic search failed: {e}")
            return f"Error searching memories: {e}"

    async def _get_recent_memories(self, user_id: str, limit: int = 10) -> str:
        """Get most recent memories."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Use a generic query to get recent items
            # Most backends will return by recency when query is broad
            results = await self._backend.search_memories(
                query="recent memories",
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return "No memories stored yet.\n\nTo save a memory, use: create /memories/<topic>.txt with your content"

            lines = ["Recent memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:150]
                if len(r.memory.content) > 150:
                    content_preview += "..."
                # Format timestamp if available
                timestamp = ""
                if hasattr(r.memory, "created_at") and r.memory.created_at:
                    timestamp = f" ({r.memory.created_at})"
                lines.append(f"{i:6d}\t{content_preview}{timestamp}")
            lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: Get recent failed: {e}")
            return f"Error getting recent memories: {e}"

    async def _list_all_memories(self, user_id: str, limit: int = 20) -> str:
        """List all memories (paginated)."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Get all memories with a broad search
            results = await self._backend.search_memories(
                query="*",  # Broad query
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return "No memories stored yet."

            lines = [f"Showing up to {limit} memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:100]
                if len(r.memory.content) > 100:
                    content_preview += "..."
                lines.append(f"{i:6d}\t{content_preview}")

            if len(results) >= limit:
                lines.append(f"\n(Showing first {limit}. Use search to find specific memories.)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: List all failed: {e}")
            return f"Error listing memories: {e}"

    async def _get_memory_overview(self, user_id: str) -> str:
        """Get memory directory overview with search instructions."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Get count of memories
            results = await self._backend.search_memories(
                query="*",
                user_id=user_id,
                top_k=100,  # Just to get a count
            )
            count = len(results) if results else 0

            # Get a few recent as preview
            preview_lines = []
            if results:
                for r in results[:3]:
                    preview = r.memory.content[:60]
                    if len(r.memory.content) > 60:
                        preview += "..."
                    preview_lines.append(f"  • {preview}")

            overview = f"""Here're the files and directories up to 2 levels deep in /memories:
4.0K\t/memories

📁 Memory System ({count} memories stored)

To SEARCH memories (semantic):
  view /memories/search/<your query>
  Example: view /memories/search/food preferences
  Example: view /memories/search/work projects

To see RECENT memories:
  view /memories/recent

To see ALL memories:
  view /memories/all

To SAVE a new memory:
  create /memories/<topic>.txt "your content here"
  Example: create /memories/preferences.txt "User likes pizza"
"""

            if preview_lines:
                overview += "\nRecent memories:\n" + "\n".join(preview_lines)

            return overview

        except Exception as e:
            logger.error(f"Memory: Overview failed: {e}")
            # Return basic help even on error
            return """📁 Memory System

To SEARCH memories: view /memories/search/<query>
To see RECENT: view /memories/recent
To SAVE: create /memories/<topic>.txt "content"
"""

    async def _native_create_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle CREATE command - save to semantic vector store."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"
        if not file_text:
            return "Error: file_text is required (the memory content)"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Extract topic from path for metadata
            topic = (
                path.replace("/memories/", "")
                .replace("/", "_")
                .replace(".txt", "")
                .replace(".md", "")
            )

            # Save to our semantic backend
            memory = await self._backend.save_memory(
                content=file_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic},
            )

            logger.info(f"Memory: Semantic create: {path} -> id={memory.id} for user {user_id}")
            return f"File created successfully at: {path}"

        except Exception as e:
            logger.error(f"Memory: Semantic create failed: {e}")
            return f"Error: {e}"

    async def _native_update_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle STR_REPLACE command - update memory content."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memory containing old_str
            results = await self._backend.search_memories(
                query=old_str,
                user_id=user_id,
                top_k=5,
            )

            # Find exact match
            matching_memory = None
            for r in results:
                if old_str in r.memory.content:
                    matching_memory = r.memory
                    break

            if not matching_memory:
                return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in memories."

            # Check for multiple occurrences
            if matching_memory.content.count(old_str) > 1:
                return f"No replacement was performed. Multiple occurrences of old_str `{old_str}`. Please ensure it is unique."

            # Perform replacement
            new_content = matching_memory.content.replace(old_str, new_str, 1)

            # Update via delete + create (or update if backend supports it)
            if hasattr(self._backend, "update_memory"):
                await self._backend.update_memory(
                    memory_id=matching_memory.id,
                    new_content=new_content,
                    user_id=user_id,
                )
            else:
                await self._backend.delete_memory(matching_memory.id)
                await self._backend.save_memory(
                    content=new_content,
                    user_id=user_id,
                    importance=0.5,
                )

            # Show snippet around the change
            lines = new_content.split("\n")
            snippet = "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(lines[:5]))

            logger.info(f"Memory: Semantic update for user {user_id}")
            return f"The memory file has been edited.\n{snippet}"

        except Exception as e:
            logger.error(f"Memory: Semantic update failed: {e}")
            return f"Error: {e}"

    async def _native_append_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle INSERT command - append to memory or create new."""
        path = input_data.get("path", "")
        insert_text = input_data.get("insert_text", "")
        _insert_line = input_data.get("insert_line", 0)  # Unused in semantic mode

        if not path:
            return "Error: path is required"
        if not insert_text:
            return "Error: insert_text is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # For semantic backend, append is just creating a new memory
            # with the additional context
            topic = path.replace("/memories/", "").replace("/", "_").replace(".txt", "")

            await self._backend.save_memory(
                content=insert_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic, "appended": True},
            )

            logger.info(f"Memory: Semantic append: {path} for user {user_id}")
            return f"The file {path} has been edited."

        except Exception as e:
            logger.error(f"Memory: Semantic append failed: {e}")
            return f"Error: {e}"

    async def _native_delete_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle DELETE command - remove from vector store."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memories with this path
            topic = (
                path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {path} does not exist"

            # Delete matching memories
            deleted_count = 0
            for r in results:
                # Check if metadata matches path
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == path or r.score > 0.8:
                    await self._backend.delete_memory(r.memory.id)
                    deleted_count += 1

            if deleted_count == 0:
                return f"Error: The path {path} does not exist"

            logger.info(
                f"Memory: Semantic delete: {path} ({deleted_count} memories) for user {user_id}"
            )
            return f"Successfully deleted {path}"

        except Exception as e:
            logger.error(f"Memory: Semantic delete failed: {e}")
            return f"Error: {e}"

    async def _native_rename_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle RENAME command - update memory path/topic."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memories with old path
            old_topic = (
                old_path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=old_topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {old_path} does not exist"

            # Update metadata for matching memories (re-save with new path)
            new_topic = new_path.replace("/memories/", "").replace("/", "_").replace(".txt", "")
            renamed_count = 0

            for r in results:
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == old_path or r.score > 0.8:
                    # Delete old and create with new path
                    await self._backend.delete_memory(r.memory.id)
                    await self._backend.save_memory(
                        content=r.memory.content,
                        user_id=user_id,
                        importance=getattr(r.memory, "importance", 0.5),
                        metadata={"virtual_path": new_path, "topic": new_topic},
                    )
                    renamed_count += 1

            if renamed_count == 0:
                return f"Error: The path {old_path} does not exist"

            logger.info(f"Memory: Semantic rename: {old_path} -> {new_path} for user {user_id}")
            return f"Successfully renamed {old_path} to {new_path}"

        except Exception as e:
            logger.error(f"Memory: Semantic rename failed: {e}")
            return f"Error: {e}"

    async def close(self) -> None:
        """Close the memory backend."""
        if self._backend and hasattr(self._backend, "close"):
            await self._backend.close()
        self._backend = None
        self._initialized = False
        logger.info("Memory: Handler closed")
