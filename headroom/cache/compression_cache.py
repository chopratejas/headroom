"""Content-addressed compression cache with LRU eviction.

Used in "token headroom mode" to avoid re-compressing messages across turns.
Maps original content hashes to their compressed versions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry storing compressed text and metadata."""

    compressed: str
    tokens_saved: int


def _is_tool_result_message(msg: dict) -> bool:
    """Check if a message is a tool result in Anthropic or OpenAI format."""
    # OpenAI format: role="tool"
    if msg.get("role") == "tool":
        return True
    # Anthropic format: role="user" with content list containing tool_result blocks
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
    return False


def _extract_tool_result_content(msg: dict) -> str | None:
    """Extract text content from a tool result message (both formats)."""
    # OpenAI format
    if msg.get("role") == "tool":
        content = msg.get("content")
        return content if isinstance(content, str) else None
    # Anthropic format
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    return inner
    return None


def _swap_tool_result_content(msg: dict, new_content: str) -> dict:
    """Deep copy msg and replace tool result content with new_content."""
    new_msg = copy.deepcopy(msg)
    # OpenAI format
    if new_msg.get("role") == "tool":
        new_msg["content"] = new_content
        return new_msg
    # Anthropic format
    content = new_msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"] = new_content
                break
    return new_msg


class CompressionCache:
    """Content-addressed cache mapping content hashes to compressed versions.

    Uses an OrderedDict for O(1) LRU eviction. Entries are evicted oldest-first
    when the cache exceeds max_entries.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._total_tokens_saved: int = 0

    def get_compressed(self, hash: str) -> str | None:
        """Retrieve compressed content by hash, refreshing LRU position on hit."""
        entry = self._cache.get(hash)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._cache.move_to_end(hash)
        return entry.compressed

    def store_compressed(self, hash: str, compressed: str, tokens_saved: int) -> None:
        """Store a compressed version keyed by content hash.

        If the hash already exists, the entry is overwritten and moved to the
        end (most recently used). When the cache exceeds max_entries, the oldest
        entry is evicted.
        """
        if hash in self._cache:
            old_entry = self._cache[hash]
            self._total_tokens_saved -= old_entry.tokens_saved
            del self._cache[hash]

        self._cache[hash] = _CacheEntry(compressed=compressed, tokens_saved=tokens_saved)
        self._total_tokens_saved += tokens_saved

        while len(self._cache) > self.max_entries:
            _, evicted = self._cache.popitem(last=False)
            self._total_tokens_saved -= evicted.tokens_saved

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "tokens_saved": self._total_tokens_saved,
        }

    @staticmethod
    def content_hash(content: str | list) -> str:
        """Compute a truncated SHA-256 hash for string or list content.

        For list content (Anthropic-format messages with type/text/content fields),
        the list is JSON-serialized with sorted keys for deterministic hashing.
        """
        if isinstance(content, list):
            raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
        else:
            raw = content
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def compute_frozen_count(self, messages: list[dict]) -> int:
        """Count consecutive stable messages from the start.

        A message is stable if it is a plain user/assistant/system message,
        an assistant message with tool_use blocks, or a tool_result whose
        content hash is already in the cache. The first unstable tool_result
        (cache miss) stops the count.
        """
        count = 0
        for msg in messages:
            if _is_tool_result_message(msg):
                content = _extract_tool_result_content(msg)
                if content is not None:
                    h = self.content_hash(content)
                    if h not in self._cache:
                        break
                else:
                    # tool_result with non-string content; treat as unstable
                    break
            # Regular user/assistant/system messages and assistant+tool_use
            # are always stable — fall through.
            count += 1
        return count

    def apply_cached(self, messages: list[dict]) -> list[dict]:
        """Return a new list with cached compressions swapped into tool results.

        Never mutates the input list or any message dict within it.
        Output always has the same length as input.
        """
        result: list[dict] = []
        for msg in messages:
            if _is_tool_result_message(msg):
                content = _extract_tool_result_content(msg)
                if content is not None:
                    h = self.content_hash(content)
                    compressed = self.get_compressed(h)
                    if compressed is not None:
                        result.append(_swap_tool_result_content(msg, compressed))
                        continue
            result.append(msg)
        return result

    def update_from_result(self, originals: list[dict], compressed: list[dict]) -> None:
        """Cache new compressions by comparing original and compressed messages.

        Index-aligned: for each position, if both are tool results and the
        content differs, store the mapping original_hash -> compressed_content.
        """
        if len(originals) != len(compressed):
            logger.warning(
                "update_from_result: length mismatch (originals=%d, compressed=%d), skipping",
                len(originals),
                len(compressed),
            )
            return

        for orig, comp in zip(originals, compressed):
            orig_content = _extract_tool_result_content(orig)
            comp_content = _extract_tool_result_content(comp)
            if orig_content is None or comp_content is None:
                continue
            if orig_content == comp_content:
                continue
            h = self.content_hash(orig_content)
            tokens_saved = len(orig_content) // 4 - len(comp_content) // 4
            self.store_compressed(h, comp_content, tokens_saved=max(tokens_saved, 0))
