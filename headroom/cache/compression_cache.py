"""Content-addressed compression cache with LRU eviction.

Used in "token headroom mode" to avoid re-compressing messages across turns.
Maps original content hashes to their compressed versions.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _CacheEntry:
    """Internal cache entry storing compressed text and metadata."""

    compressed: str
    tokens_saved: int


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
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
