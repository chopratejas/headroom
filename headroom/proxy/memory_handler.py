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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)

# Memory tool names for detection
MEMORY_TOOL_NAMES = {"memory_save", "memory_search", "memory_update", "memory_delete"}


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
    # Qdrant+Neo4j config
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


class MemoryHandler:
    """Unified handler for memory operations in the proxy.

    Responsibilities:
    1. Initialize and manage memory backend
    2. Inject memory tools into requests
    3. Search and inject relevant memories as context
    4. Handle memory tool calls in responses
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._backend: LocalBackend | Any = None
        self._initialized = False
        self._memory_tools: list[dict[str, Any]] | None = None

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
            if name in MEMORY_TOOL_NAMES:
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
        await self._ensure_initialized()
        if not self._backend:
            return []

        tool_calls = self._extract_tool_calls(response, provider)
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc.get("name") or tc.get("function", {}).get("name")
            if tool_name not in MEMORY_TOOL_NAMES:
                continue

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

            # Execute the tool
            result_content = await self._execute_memory_tool(tool_name, input_data, user_id)

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

    async def close(self) -> None:
        """Close the memory backend."""
        if self._backend and hasattr(self._backend, "close"):
            await self._backend.close()
        self._backend = None
        self._initialized = False
        logger.info("Memory: Handler closed")
