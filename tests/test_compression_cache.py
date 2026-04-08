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


class TestCompressionCacheFrozenCount:
    def test_empty_cache_returns_zero(self, cache: CompressionCache) -> None:
        assert cache.compute_frozen_count([]) == 0

    def test_user_assistant_always_stable(self, cache: CompressionCache) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        assert cache.compute_frozen_count(messages) == 3

    def test_tool_result_with_cache_hit_is_stable(self, cache: CompressionCache) -> None:
        tool_content = "tool output data"
        h = CompressionCache.content_hash(tool_content)
        cache.store_compressed(h, "compressed tool output", tokens_saved=5)

        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "my_tool", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        assert cache.compute_frozen_count(messages) == 3

    def test_tool_result_cache_miss_stops_frozen(self, cache: CompressionCache) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "uncached stuff"}
                ],
            },
            {"role": "user", "content": "follow up"},
        ]
        assert cache.compute_frozen_count(messages) == 1

    def test_frozen_count_with_dropped_messages(self, cache: CompressionCache) -> None:
        cached_content = "cached tool output"
        h = CompressionCache.content_hash(cached_content)
        cache.store_compressed(h, "compressed", tokens_saved=3)

        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": cached_content}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "not cached"}],
            },
        ]
        assert cache.compute_frozen_count(messages) == 2

    def test_stable_hash_allows_frozen_count_past_uncached_tool_result(
        self, cache: CompressionCache
    ) -> None:
        """Tool_results marked stable should not stop the frozen count walk."""
        tool_content = "excluded Read output — big file contents"
        h = CompressionCache.content_hash(tool_content)
        cache.mark_stable(h)

        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
            {"role": "user", "content": "follow up"},
        ]
        # Without mark_stable, this would stop at msg[1] → frozen=1.
        # With stable hash, the walk continues past msg[1] → frozen=3.
        assert cache.compute_frozen_count(messages) == 3

    def test_update_from_result_identical_content_marks_stable(
        self, cache: CompressionCache
    ) -> None:
        """When orig == compressed, update_from_result marks the hash as stable."""
        tool_content = "unchanged tool output"
        originals = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        # Compressed is identical to originals (no compression happened)
        compressed = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
        ]
        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash(tool_content)
        assert h in cache._stable_hashes

        # Frozen count should now walk past this tool_result
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}],
            },
            {"role": "user", "content": "more stuff"},
        ]
        assert cache.compute_frozen_count(messages) == 3

    def test_mark_stable_from_messages(self, cache: CompressionCache) -> None:
        """mark_stable_from_messages records hashes for tool_results."""
        content_a = "tool output A"
        content_b = "tool output B"
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content_a}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": content_b}],
            },
        ]
        # Mark first 2 messages (msg[0] + msg[1])
        cache.mark_stable_from_messages(messages, 2)

        ha = CompressionCache.content_hash(content_a)
        hb = CompressionCache.content_hash(content_b)
        assert ha in cache._stable_hashes
        assert hb not in cache._stable_hashes  # msg[2] not included

    def test_should_defer_compression_new_content(self, cache: CompressionCache) -> None:
        """First-time content should be deferred."""
        h = CompressionCache.content_hash("brand new content")
        assert cache.should_defer_compression(h, ttl_seconds=300, batch_window=30) is True

    def test_should_defer_compression_near_ttl(self, cache: CompressionCache) -> None:
        """Content near TTL boundary should NOT be deferred."""
        import time

        h = CompressionCache.content_hash("old content")
        # Backdate first_seen to simulate age near TTL
        cache._first_seen[h] = time.time() - 280  # 280s old, TTL=300, window=30
        assert cache.should_defer_compression(h, ttl_seconds=300, batch_window=30) is False


class TestCompressionCacheApplyAndUpdate:
    def test_apply_cached_swaps_tool_results(self, cache: CompressionCache) -> None:
        original_content = "big tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "small output", tokens_saved=5)

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": original_content}
                ],
            },
        ]
        result = cache.apply_cached(messages)
        assert result[1]["content"][0]["content"] == "small output"

    def test_apply_cached_preserves_uncached_messages(self, cache: CompressionCache) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = cache.apply_cached(messages)
        assert result[0] is messages[0]
        assert result[1] is messages[1]

    def test_apply_cached_never_adds_messages(self, cache: CompressionCache) -> None:
        # Store something in cache that doesn't correspond to any message
        cache.store_compressed("orphan_hash", "orphan_value", tokens_saved=1)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = cache.apply_cached(messages)
        assert len(result) == len(messages)

    def test_update_from_result_caches_changes(self, cache: CompressionCache) -> None:
        originals = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "original output"}
                ],
            },
        ]
        compressed = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "compressed output"}
                ],
            },
        ]
        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash("original output")
        assert cache.get_compressed(h) == "compressed output"

    def test_update_from_result_ignores_unchanged(self, cache: CompressionCache) -> None:
        originals = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "same content"}
                ],
            },
        ]
        compressed = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "same content"}
                ],
            },
        ]
        cache.update_from_result(originals, compressed)
        h = CompressionCache.content_hash("same content")
        assert cache.get_compressed(h) is None

    def test_apply_does_not_modify_original_messages(self, cache: CompressionCache) -> None:
        original_content = "big tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "small output", tokens_saved=5)

        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": original_content}],
        }
        messages = [msg]
        cache.apply_cached(messages)

        # Original must be untouched
        assert msg["content"][0]["content"] == original_content

    def test_openai_format_tool_result(self, cache: CompressionCache) -> None:
        original_content = "openai tool output"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, "compressed openai", tokens_saved=4)

        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": original_content},
        ]
        result = cache.apply_cached(messages)
        assert result[0]["content"] == "compressed openai"
        # Original untouched
        assert messages[0]["content"] == original_content
