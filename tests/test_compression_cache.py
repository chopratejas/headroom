"""Tests for CompressionCache with LRU eviction."""

from __future__ import annotations

import pytest

from headroom.cache.compression_cache import CompressionCache


@pytest.fixture
def cache() -> CompressionCache:
    return CompressionCache()


@pytest.fixture
def small_cache() -> CompressionCache:
    return CompressionCache(max_entries=3)


class TestCompressionCache:
    def test_cache_miss_returns_none(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("some content")
        assert cache.get_compressed(h) is None

    def test_store_and_retrieve(self, cache: CompressionCache) -> None:
        content = "hello world this is a long message"
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "hello world...compressed", tokens_saved=15)
        assert cache.get_compressed(h) == "hello world...compressed"

    def test_different_content_different_hash(self) -> None:
        h1 = CompressionCache.content_hash("content A")
        h2 = CompressionCache.content_hash("content B")
        assert h1 != h2

    def test_overwrite_same_hash(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("some content")
        cache.store_compressed(h, "v1", tokens_saved=10)
        cache.store_compressed(h, "v2", tokens_saved=20)
        assert cache.get_compressed(h) == "v2"

    def test_stats_tracking(self, cache: CompressionCache) -> None:
        h = CompressionCache.content_hash("content")
        cache.store_compressed(h, "compressed", tokens_saved=5)

        # One hit
        cache.get_compressed(h)
        # One miss
        cache.get_compressed("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["tokens_saved"] == 5

    def test_eviction_at_max_entries(self, small_cache: CompressionCache) -> None:
        h1 = CompressionCache.content_hash("a")
        h2 = CompressionCache.content_hash("b")
        h3 = CompressionCache.content_hash("c")
        h4 = CompressionCache.content_hash("d")

        small_cache.store_compressed(h1, "ca", tokens_saved=1)
        small_cache.store_compressed(h2, "cb", tokens_saved=1)
        small_cache.store_compressed(h3, "cc", tokens_saved=1)

        # Adding a 4th should evict the oldest (h1)
        small_cache.store_compressed(h4, "cd", tokens_saved=1)

        assert small_cache.get_compressed(h1) is None
        assert small_cache.get_compressed(h2) == "cb"
        assert small_cache.get_compressed(h4) == "cd"

    def test_access_refreshes_lru(self, small_cache: CompressionCache) -> None:
        h1 = CompressionCache.content_hash("a")
        h2 = CompressionCache.content_hash("b")
        h3 = CompressionCache.content_hash("c")
        h4 = CompressionCache.content_hash("d")

        small_cache.store_compressed(h1, "ca", tokens_saved=1)
        small_cache.store_compressed(h2, "cb", tokens_saved=1)
        small_cache.store_compressed(h3, "cc", tokens_saved=1)

        # Access h1 to refresh it
        small_cache.get_compressed(h1)

        # Adding h4 should evict h2 (oldest untouched), not h1
        small_cache.store_compressed(h4, "cd", tokens_saved=1)

        assert small_cache.get_compressed(h1) == "ca"
        assert small_cache.get_compressed(h2) is None
        assert small_cache.get_compressed(h4) == "cd"

    def test_content_hash_list_content(self) -> None:
        """content_hash handles Anthropic-format list content."""
        list_content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        h = CompressionCache.content_hash(list_content)
        assert isinstance(h, str)
        assert len(h) == 16

        # Same content produces same hash
        assert CompressionCache.content_hash(list_content) == h

    def test_content_hash_string_length(self) -> None:
        h = CompressionCache.content_hash("test")
        assert len(h) == 16
