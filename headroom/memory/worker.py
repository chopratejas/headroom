"""Background worker for batched memory extraction.

Collects conversations in a queue and processes them in batches,
reducing LLM calls and improving efficiency.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.memory.extractor import MemoryExtractor
    from headroom.memory.store import SQLiteMemoryStore


logger = logging.getLogger(__name__)


class ExtractionWorker:
    """Background worker that batches memory extractions.

    Features:
    - Collects conversations in a queue
    - Processes in batches (configurable size and timeout)
    - Persists pending work to SQLite for crash recovery
    - Thread-safe, daemon thread (stops with main program)

    Usage:
        worker = ExtractionWorker(store, extractor)
        worker.start()
        worker.schedule("alice", "I prefer Python", "Great choice!")
        # ... later, memories are extracted and saved automatically
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        extractor: MemoryExtractor,
        batch_size: int = 10,
        max_wait_seconds: float = 30.0,
    ):
        """Initialize the worker.

        Args:
            store: Memory store for saving extracted memories
            extractor: Extractor for processing conversations
            batch_size: Max conversations per batch
            max_wait_seconds: Max time to wait before processing partial batch
        """
        self.store = store
        self.extractor = extractor
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds

        self._queue: list[tuple[str, str, str]] = []  # (user_id, query, response)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Process any pending extractions from previous runs (crash recovery)
        self._recover_pending()

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop the worker.

        Args:
            wait: If True, process remaining queue before stopping
            timeout: Max time to wait for remaining work
        """
        if not self._running:
            return

        self._running = False
        self._event.set()  # Wake up the thread

        if wait and self._thread:
            self._thread.join(timeout=timeout)

    def schedule(self, user_id: str, query: str, response: str) -> None:
        """Schedule a conversation for memory extraction.

        Non-blocking - returns immediately and extracts in background.

        Args:
            user_id: User identifier
            query: User's message
            response: Assistant's response
        """
        # Persist to SQLite first (crash recovery)
        from headroom.memory.store import PendingExtraction

        pending = PendingExtraction(
            user_id=user_id,
            query=query,
            response=response,
        )
        self.store.queue_extraction(pending)

        # Add to in-memory queue
        with self._lock:
            self._queue.append((user_id, query, response))

            # Wake up worker if batch is full
            if len(self._queue) >= self.batch_size:
                self._event.set()

    def flush(self, timeout: float = 60.0) -> bool:
        """Force immediate processing of all queued extractions.

        Blocks until all pending extractions are processed or timeout.

        Args:
            timeout: Max time to wait in seconds

        Returns:
            True if all extractions completed, False if timed out
        """
        # Signal worker to process immediately by temporarily setting max_wait to 0
        original_max_wait = self.max_wait_seconds
        self.max_wait_seconds = 0
        self._event.set()

        # Wait for queue to empty
        start = time.time()
        while time.time() - start < timeout:
            pending = self.store.get_pending_extractions(limit=1, status="pending")
            if not pending:
                self.max_wait_seconds = original_max_wait
                return True
            time.sleep(0.5)

        self.max_wait_seconds = original_max_wait
        return False

    def _run(self) -> None:
        """Main worker loop."""
        last_process_time = time.time()

        while self._running:
            # Wait for batch to fill or timeout
            self._event.wait(timeout=1.0)
            self._event.clear()

            now = time.time()
            time_since_last = now - last_process_time

            with self._lock:
                should_process = len(self._queue) >= self.batch_size or (
                    self._queue and time_since_last >= self.max_wait_seconds
                )

                if should_process:
                    batch = self._queue[: self.batch_size]
                    self._queue = self._queue[self.batch_size :]
                else:
                    batch = []

            if batch:
                self._process_batch(batch)
                last_process_time = time.time()

        # Process remaining queue on shutdown
        with self._lock:
            remaining = self._queue[:]
            self._queue = []

        if remaining:
            self._process_batch(remaining)

    def _process_batch(self, batch: list[tuple[str, str, str]]) -> None:
        """Process a batch of conversations.

        Args:
            batch: List of (user_id, query, response) tuples
        """
        logger.debug(f"Processing batch of {len(batch)} conversations")

        try:
            # Extract memories
            result = self.extractor.extract_batch(batch)

            # Save memories
            for user_id, memories in result.items():
                for memory in memories:
                    self.store.save(user_id, memory)
                    logger.debug(f"Saved memory for {user_id}: {memory.content[:50]}...")

            # Mark pending extractions as done
            # Note: In a production system, we'd track exact IDs
            # For simplicity, we clear pending by matching user/query/response
            self._mark_batch_done(batch)

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            self._mark_batch_failed(batch)

    def _recover_pending(self) -> None:
        """Recover pending extractions from previous runs."""
        pending = self.store.get_pending_extractions(limit=100, status="pending")

        if not pending:
            return

        logger.info(f"Recovering {len(pending)} pending extractions")

        with self._lock:
            for p in pending:
                self._queue.append((p.user_id, p.query, p.response))

        # Trigger processing
        self._event.set()

    def _mark_batch_done(self, batch: list[tuple[str, str, str]]) -> None:
        """Mark batch items as completed in the pending table."""
        # Get pending extractions and mark matching ones as done
        pending = self.store.get_pending_extractions(limit=100)

        for user_id, query, response in batch:
            for p in pending:
                if p.user_id == user_id and p.query == query and p.response == response:
                    self.store.delete_extraction(p.id)
                    break

    def _mark_batch_failed(self, batch: list[tuple[str, str, str]]) -> None:
        """Mark batch items as failed in the pending table."""
        pending = self.store.get_pending_extractions(limit=100)

        for user_id, query, response in batch:
            for p in pending:
                if p.user_id == user_id and p.query == query and p.response == response:
                    self.store.update_extraction_status(p.id, "failed")
                    break

    def _cleanup(self) -> None:
        """Cleanup on program exit."""
        if self._running:
            self.stop(wait=True, timeout=2.0)

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)
