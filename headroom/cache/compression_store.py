"""Compression Store for CCR (Compress-Cache-Retrieve) architecture.

This module implements reversible compression: when SmartCrusher compresses
tool outputs, the original data is cached here for on-demand retrieval.

Key insight from research: REVERSIBLE compression beats irreversible compression.
If the LLM needs data that was compressed away, it can retrieve it instantly.

Features:
- Thread-safe in-memory storage with TTL expiration
- BM25-based search within cached content
- Retrieval event tracking for feedback loop
- Automatic eviction when capacity is reached

Usage:
    store = get_compression_store()

    # Store compressed content
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_tokens=1000,
        compressed_tokens=100,
        tool_name="search_api",
    )

    # Retrieve later
    entry = store.retrieve(hash_key)

    # Or search within
    results = store.search(hash_key, "user query")
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..relevance.bm25 import BM25Scorer

logger = logging.getLogger(__name__)


@dataclass
class CompressionEntry:
    """A cached compression entry with metadata for retrieval and feedback."""

    hash: str
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None
    tool_call_id: str | None
    query_context: str | None
    created_at: float
    ttl: int = 300  # 5 minutes default

    # TOIN integration: Store the tool signature hash for retrieval correlation
    # This MUST match the hash used by SmartCrusher when recording compression
    tool_signature_hash: str | None = None
    compression_strategy: str | None = None  # Strategy used for compression

    # Feedback tracking
    retrieval_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() - self.created_at > self.ttl

    def record_access(self, query: str | None = None) -> None:
        """Record an access to this entry for feedback tracking."""
        self.retrieval_count += 1
        self.last_accessed = time.time()
        if query and query not in self.search_queries:
            self.search_queries.append(query)
            # Keep only last 10 queries
            if len(self.search_queries) > 10:
                self.search_queries = self.search_queries[-10:]


@dataclass
class RetrievalEvent:
    """Event logged when content is retrieved from cache."""

    hash: str
    query: str | None
    items_retrieved: int
    total_items: int
    tool_name: str | None
    timestamp: float
    retrieval_type: str  # "full" or "search"
    tool_signature_hash: str | None = None  # For TOIN correlation


class CompressionStore:
    """Thread-safe store for compressed content with retrieval support.

    This is the core of the CCR architecture. When SmartCrusher compresses
    an array, the original content is stored here. If the LLM needs more
    data, it can retrieve from this cache instantly.

    Design principles:
    - Zero external dependencies (pure Python)
    - Thread-safe for concurrent access
    - TTL-based expiration (default 5 minutes)
    - LRU-style eviction when capacity is reached
    - Built-in BM25 search for filtering
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 300,
        enable_feedback: bool = True,
    ):
        """Initialize the compression store.

        Args:
            max_entries: Maximum number of entries to store.
            default_ttl: Default TTL in seconds (5 minutes).
            enable_feedback: Whether to track retrieval events.
        """
        self._store: dict[str, CompressionEntry] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._enable_feedback = enable_feedback

        # Feedback tracking
        self._retrieval_events: list[RetrievalEvent] = []
        self._max_events = 1000  # Keep last 1000 events
        self._pending_feedback_events: list[RetrievalEvent] = []

        # MEDIUM FIX #16: Use a min-heap for O(log n) eviction instead of O(n)
        # Heap entries are (created_at, hash_key) tuples
        self._eviction_heap: list[tuple[float, str]] = []
        # CRITICAL FIX: Track stale entries count to know when heap cleanup is needed
        self._stale_heap_entries = 0
        # Threshold for triggering heap rebuild (when 50% are stale)
        self._heap_rebuild_threshold = 0.5

        # BM25 scorer for search
        self._scorer = BM25Scorer()

    def store(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        tool_signature_hash: str | None = None,
        compression_strategy: str | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store compressed content and return hash for retrieval.

        Args:
            original: Original JSON content before compression.
            compressed: Compressed JSON content.
            original_tokens: Token count of original content.
            compressed_tokens: Token count of compressed content.
            original_item_count: Number of items in original array.
            compressed_item_count: Number of items after compression.
            tool_name: Name of the tool that produced this output.
            tool_call_id: ID of the tool call.
            query_context: User query context for relevance matching.
            tool_signature_hash: Hash from ToolSignature for TOIN correlation.
            compression_strategy: Strategy used for compression.
            ttl: Custom TTL in seconds (uses default if not specified).

        Returns:
            Hash key for retrieving this content.
        """
        # Generate hash from original content
        # CRITICAL FIX #5: Use 24 chars (96 bits) instead of 16 (64 bits) for better
        # collision resistance. Birthday paradox: 50% collision at sqrt(2^n) entries.
        # - 64 bits: ~4 billion entries for 50% collision
        # - 96 bits: ~280 trillion entries for 50% collision
        hash_key = hashlib.sha256(original.encode()).hexdigest()[:24]

        entry = CompressionEntry(
            hash=hash_key,
            original_content=original,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_item_count,
            compressed_item_count=compressed_item_count,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=query_context,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
            tool_signature_hash=tool_signature_hash,
            compression_strategy=compression_strategy,
        )

        # Process pending feedback BEFORE acquiring lock for eviction.
        # This ensures feedback from entries about to be evicted is captured.
        if self._enable_feedback:
            self.process_pending_feedback()

        with self._lock:
            self._evict_if_needed()

            # CRITICAL FIX: Hash collision detection
            # If hash already exists with DIFFERENT content, log a warning.
            # This indicates either a hash collision or duplicate store calls.
            existing = self._store.get(hash_key)
            if existing is not None:
                if existing.original_content != original:
                    # True hash collision - different content, same hash
                    # This is extremely rare with SHA256[:24] but should be logged
                    logger.warning(
                        "Hash collision detected: hash=%s tool=%s (existing_len=%d, new_len=%d)",
                        hash_key,
                        tool_name,
                        len(existing.original_content),
                        len(original),
                    )
                else:
                    # Same content being stored again - this is fine, just update
                    logger.debug(
                        "Duplicate store for hash=%s, updating entry",
                        hash_key,
                    )
                # Mark old heap entry as stale since we're replacing
                self._stale_heap_entries += 1

            self._store[hash_key] = entry
            # MEDIUM FIX #16: Add to eviction heap for O(log n) eviction
            heapq.heappush(self._eviction_heap, (entry.created_at, hash_key))

        return hash_key

    def retrieve(
        self,
        hash_key: str,
        query: str | None = None,
    ) -> CompressionEntry | None:
        """Retrieve original content by hash.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for feedback tracking.

        Returns:
            CompressionEntry if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._store.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._store[hash_key]
                # CRITICAL FIX: Track stale heap entry
                self._stale_heap_entries += 1
                return None

            # Track access for feedback
            entry.record_access(query)

            # Log retrieval event
            if self._enable_feedback:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=entry.original_item_count,
                    total_items=entry.original_item_count,
                    tool_name=entry.tool_name,
                    retrieval_type="full",
                    tool_signature_hash=entry.tool_signature_hash,
                )

            # CRITICAL: Make a deep copy to return (entry could be modified/evicted after lock release)
            # The entry contains mutable fields (search_queries list) that must be copied
            result_entry = replace(entry, search_queries=list(entry.search_queries))

        # Process feedback immediately to ensure TOIN learns in real-time
        if self._enable_feedback:
            self.process_pending_feedback()

        return result_entry

    def search(
        self,
        hash_key: str,
        query: str,
        max_results: int = 20,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search within cached content using BM25.

        Args:
            hash_key: Hash key of cached content.
            query: Search query.
            max_results: Maximum number of results to return.
            score_threshold: Minimum BM25 score to include.

        Returns:
            List of matching items from original content.
        """
        # Get entry without logging (we'll log the search separately)
        entry = self._get_entry_for_search(hash_key, query)
        if entry is None:
            return []

        try:
            items = json.loads(entry.original_content)
            if not isinstance(items, list):
                return []
        except json.JSONDecodeError:
            return []

        if not items:
            return []

        # Score each item using BM25
        item_strs = [json.dumps(item, default=str) for item in items]
        scores = self._scorer.score_batch(item_strs, query)

        # Filter and sort by score
        scored_items = [
            (items[i], scores[i].score)
            for i in range(len(items))
            if scores[i].score >= score_threshold
        ]
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = [item for item, _ in scored_items[:max_results]]

        # Log retrieval event
        if self._enable_feedback:
            with self._lock:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=len(results),
                    total_items=len(items),
                    tool_name=entry.tool_name,
                    retrieval_type="search",
                    tool_signature_hash=entry.tool_signature_hash,
                )
            # Process feedback immediately to ensure TOIN learns in real-time
            self.process_pending_feedback()

        return results

    def _get_entry_for_search(
        self,
        hash_key: str,
        query: str | None = None,
    ) -> CompressionEntry | None:
        """Get entry without logging retrieval (used by search to avoid double-logging).

        CRITICAL FIX #4: Returns a copy of the entry to prevent race conditions.
        The caller may use the entry after we release the lock, and another thread
        could modify or evict the original entry.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for access tracking.

        Returns:
            CompressionEntry copy if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._store.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._store[hash_key]
                # CRITICAL FIX: Track stale heap entry
                self._stale_heap_entries += 1
                return None

            # Track access but don't log retrieval event (search will log separately)
            entry.record_access(query)

            # CRITICAL FIX #4: Return a copy to prevent race conditions
            # The entry contains mutable fields (search_queries list) that could be
            # modified by other threads after we release the lock
            return replace(entry, search_queries=list(entry.search_queries))

    def exists(self, hash_key: str, clean_expired: bool = False) -> bool:
        """Check if a hash key exists and is not expired.

        Args:
            hash_key: The hash key to check.
            clean_expired: If True, delete the entry if expired.
                          LOW FIX #20: Default False to make this a pure check.

        Returns:
            True if the entry exists and is not expired.
        """
        with self._lock:
            entry = self._store.get(hash_key)
            if entry is None:
                return False
            if entry.is_expired():
                # LOW FIX #20: Only delete if explicitly requested
                # This makes exists() a pure check by default
                if clean_expired:
                    del self._store[hash_key]
                    # CRITICAL FIX: Track stale heap entry
                    self._stale_heap_entries += 1
                return False
            return True

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics for monitoring."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            total_original_tokens = sum(e.original_tokens for e in self._store.values())
            total_compressed_tokens = sum(e.compressed_tokens for e in self._store.values())
            total_retrievals = sum(e.retrieval_count for e in self._store.values())

            return {
                "entry_count": len(self._store),
                "max_entries": self._max_entries,
                "total_original_tokens": total_original_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "total_retrievals": total_retrievals,
                "event_count": len(self._retrieval_events),
            }

    def get_retrieval_events(
        self,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[RetrievalEvent]:
        """Get recent retrieval events for feedback analysis.

        Args:
            limit: Maximum number of events to return.
            tool_name: Filter by tool name if specified.

        Returns:
            List of recent retrieval events (copies to prevent mutation).
        """
        with self._lock:
            # MEDIUM FIX #17: Take a slice copy immediately to avoid race conditions
            # if another thread modifies _retrieval_events after we release the lock
            events_copy = list(self._retrieval_events)

        # Filter and slice outside lock (safe since we have a copy)
        if tool_name:
            events_copy = [e for e in events_copy if e.tool_name == tool_name]

        return list(reversed(events_copy[-limit:]))

    def clear(self) -> None:
        """Clear all entries. Mainly for testing."""
        with self._lock:
            self._store.clear()
            self._retrieval_events.clear()
            self._pending_feedback_events.clear()
            self._eviction_heap.clear()  # MEDIUM FIX #16: Clear heap too
            self._stale_heap_entries = 0  # CRITICAL FIX: Reset stale counter

    def _evict_if_needed(self) -> None:
        """Evict old entries if at capacity. Must be called with lock held.

        MEDIUM FIX #16: Use heap for O(log n) eviction instead of O(n) scan.
        CRITICAL FIX: Track and clean stale heap entries to prevent memory leak.
        """
        # First, remove expired entries
        self._clean_expired()

        # CRITICAL FIX: Rebuild heap if too many stale entries
        # This prevents unbounded heap growth when entries are deleted/replaced
        heap_size = len(self._eviction_heap)
        if heap_size > 0:
            stale_ratio = self._stale_heap_entries / heap_size
            if stale_ratio >= self._heap_rebuild_threshold:
                self._rebuild_heap()

        # If still at capacity, remove oldest entries using heap
        while len(self._store) >= self._max_entries and self._eviction_heap:
            # Pop oldest from heap (O(log n))
            created_at, hash_key = heapq.heappop(self._eviction_heap)

            # Check if entry still exists and matches timestamp
            # (entry might have been deleted or replaced)
            entry = self._store.get(hash_key)
            if entry is not None and entry.created_at == created_at:
                # HIGH FIX: Track eviction as "successful compression" if never retrieved
                # This prevents state divergence between store and feedback loop
                if self._enable_feedback and entry.retrieval_count == 0:
                    # Entry was never retrieved = compression was successful
                    # Notify feedback system so it knows this strategy worked
                    self._record_eviction_success(entry)
                del self._store[hash_key]
            else:
                # CRITICAL FIX: This was a stale entry, decrement counter
                # (we already popped it, so the stale entry is now gone)
                if self._stale_heap_entries > 0:
                    self._stale_heap_entries -= 1

    def _clean_expired(self) -> None:
        """Remove expired entries. Must be called with lock held.

        CRITICAL FIX: Track stale heap entries when deleting to prevent memory leak.
        """
        expired_keys = [key for key, entry in self._store.items() if entry.is_expired()]
        for key in expired_keys:
            del self._store[key]
            # CRITICAL FIX: Increment stale counter - the heap still has an entry
            # for this key that will be stale when we try to evict
            self._stale_heap_entries += 1

    def _rebuild_heap(self) -> None:
        """Rebuild heap from current store entries. Must be called with lock held.

        CRITICAL FIX: This removes stale heap entries that accumulate when entries
        are deleted or replaced. Without this, the heap grows unboundedly.
        """
        # Build new heap from current store entries only
        self._eviction_heap = [
            (entry.created_at, hash_key) for hash_key, entry in self._store.items()
        ]
        heapq.heapify(self._eviction_heap)
        # Reset stale counter - heap is now clean
        self._stale_heap_entries = 0
        logger.debug(
            "Rebuilt eviction heap: %d entries",
            len(self._eviction_heap),
        )

    def _record_eviction_success(self, entry: CompressionEntry) -> None:
        """Record successful compression when an entry is evicted without retrieval.

        HIGH FIX: State divergence on eviction
        When an entry is evicted and was NEVER retrieved, this indicates the
        compression was fully successful - the LLM never needed the original data.
        We notify the feedback system so it can learn from this success.

        Must be called with lock held (entry data access).
        Actual feedback notification happens outside lock.

        Args:
            entry: The entry being evicted.
        """
        # Capture entry data while we have the lock
        tool_name = entry.tool_name
        sig_hash = entry.tool_signature_hash
        strategy = entry.compression_strategy

        # We can't call feedback while holding the lock (would cause deadlock)
        # Instead, queue this for deferred processing
        if sig_hash is not None and strategy is not None:
            # Create a synthetic "success" event that we'll process later
            # Use a special retrieval type to indicate this was an eviction success
            success_event = RetrievalEvent(
                hash=entry.hash,
                query=None,
                items_retrieved=0,  # No retrieval happened
                total_items=entry.original_item_count,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type="eviction_success",  # Special marker
                tool_signature_hash=sig_hash,
            )
            self._pending_feedback_events.append(success_event)
            logger.debug(
                "Recorded eviction success: hash=%s strategy=%s",
                entry.hash[:8],
                strategy,
            )

    def _log_retrieval(
        self,
        hash_key: str,
        query: str | None,
        items_retrieved: int,
        total_items: int,
        tool_name: str | None,
        retrieval_type: str,
        tool_signature_hash: str | None = None,
    ) -> None:
        """Log a retrieval event. Must be called with lock held."""
        event = RetrievalEvent(
            hash=hash_key,
            query=query,
            items_retrieved=items_retrieved,
            total_items=total_items,
            tool_name=tool_name,
            timestamp=time.time(),
            retrieval_type=retrieval_type,
            tool_signature_hash=tool_signature_hash,
        )

        self._retrieval_events.append(event)

        # Keep only recent events
        if len(self._retrieval_events) > self._max_events:
            self._retrieval_events = self._retrieval_events[-self._max_events :]

        # Queue event for feedback processing (will be processed after lock release)
        # This is safe because process_pending_feedback() uses the lock to atomically
        # swap out the pending list before processing
        self._pending_feedback_events.append(event)

    def process_pending_feedback(self) -> None:
        """Process pending feedback events.

        Forwards events to:
        1. CompressionFeedback - for learning compression hints
        2. TelemetryCollector - for the data flywheel
        3. TOIN - for cross-user intelligence network

        This is called automatically on each retrieval to ensure the
        feedback loop operates in real-time.
        """
        from ..telemetry import get_telemetry_collector
        from ..telemetry.toin import get_toin
        from .compression_feedback import get_compression_feedback

        # Get pending events and related entry data atomically
        with self._lock:
            events = self._pending_feedback_events
            self._pending_feedback_events = []

            # Gather entry data while holding lock to avoid race conditions
            event_data: list[tuple[RetrievalEvent, str | None, str | None, str | None]] = []
            for event in events:
                entry = self._store.get(event.hash)
                if entry:
                    # Use the ACTUAL tool_signature_hash stored during compression
                    # This MUST match the hash used by SmartCrusher
                    event_data.append(
                        (
                            event,
                            entry.tool_name,
                            entry.tool_signature_hash,  # The correct hash!
                            entry.compression_strategy,
                        )
                    )
                else:
                    event_data.append((event, None, None, None))

        # Process outside lock
        if event_data:
            feedback = get_compression_feedback()
            telemetry = get_telemetry_collector()
            toin = get_toin()

            for event, _tool_name, sig_hash, strategy in event_data:
                # Notify feedback system (pass strategy for success rate tracking)
                feedback.record_retrieval(event, strategy=strategy)

                # Extract query fields if present
                query_fields = None
                if event.query:
                    # Extract field:value patterns
                    query_fields = re.findall(r"(\w+)[=:]", event.query)

                # Notify telemetry for data flywheel
                try:
                    if sig_hash is not None:
                        telemetry.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query_fields=query_fields,
                        )
                except Exception:
                    # Telemetry should never break the feedback loop
                    logger.debug("Telemetry record_retrieval failed", exc_info=True)

                # Notify TOIN for cross-user learning
                try:
                    if sig_hash is not None:
                        toin.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query=event.query,
                            query_fields=query_fields,
                            strategy=strategy,  # Pass strategy for success rate tracking
                        )
                except Exception:
                    # TOIN should never break the feedback loop
                    logger.debug("TOIN record_retrieval failed", exc_info=True)


# Global store instance (lazy initialization)
_compression_store: CompressionStore | None = None
_store_lock = threading.Lock()


def get_compression_store(
    max_entries: int = 1000,
    default_ttl: int = 300,
) -> CompressionStore:
    """Get the global compression store instance.

    Uses lazy initialization with singleton pattern.

    Args:
        max_entries: Maximum entries (only used on first call).
        default_ttl: Default TTL (only used on first call).

    Returns:
        Global CompressionStore instance.
    """
    global _compression_store

    if _compression_store is None:
        with _store_lock:
            # Double-check after acquiring lock
            if _compression_store is None:
                _compression_store = CompressionStore(
                    max_entries=max_entries,
                    default_ttl=default_ttl,
                )

    return _compression_store


def reset_compression_store() -> None:
    """Reset the global compression store. Mainly for testing."""
    global _compression_store

    with _store_lock:
        if _compression_store is not None:
            _compression_store.clear()
        _compression_store = None
