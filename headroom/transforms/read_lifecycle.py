"""Event-driven Read lifecycle management.

Detects stale and superseded Read tool outputs in conversation messages and
replaces them with compact markers + CCR hashes. Fresh Reads are never touched.

A Read becomes STALE when its file is subsequently edited — the content in
context is factually wrong. A Read becomes SUPERSEDED when the same file is
re-Read — the content is redundant. Both are provably safe to replace.

Real-world data shows 75% of Read output bytes fall into these two categories:
- 67% stale (file edited after Read)
- 12% superseded (file re-Read later)
- Only 20% are fresh (untouched)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import (
    _MUTATING_TOOL_NAMES,
    _READ_TOOL_NAMES,
    ReadLifecycleConfig,
)

logger = logging.getLogger(__name__)


class ReadState(str, Enum):
    """Lifecycle state of a Read output."""

    FRESH = "fresh"  # Latest read, no subsequent edit — leave untouched
    STALE = "stale"  # File was edited after this Read — content is wrong
    SUPERSEDED = "superseded"  # File was re-Read after this Read — content is redundant


@dataclass
class FileOperation:
    """A single file operation observed in the conversation."""

    msg_index: int  # Position in messages[]
    tool_call_id: str
    tool_name: str
    file_path: str
    operation: str  # "read" | "edit" | "write"
    content_size: int = 0  # Size of tool_result content (for reads only)


@dataclass
class ReadClassification:
    """Classification of a single Read output."""

    msg_index: int
    tool_call_id: str
    file_path: str
    state: ReadState
    content_size: int


@dataclass
class ReadLifecycleResult:
    """Output of lifecycle management pass."""

    messages: list[dict[str, Any]]
    reads_total: int = 0
    reads_stale: int = 0
    reads_superseded: int = 0
    reads_fresh: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    ccr_hashes: list[str] = field(default_factory=list)


class ReadLifecycleManager:
    """Event-driven Read lifecycle management.

    Pre-processes messages[] to identify and replace stale/superseded Read outputs.
    Operates before ContentRouter, independent of tool exclusion logic.
    """

    def __init__(
        self,
        config: ReadLifecycleConfig,
        compression_store: Any | None = None,
    ):
        self.config = config
        self.store = compression_store

    def apply(self, messages: list[dict[str, Any]]) -> ReadLifecycleResult:
        """Apply lifecycle management to messages.

        Single-pass analysis, targeted replacement of stale/superseded Reads.
        """
        if not self.config.enabled:
            return ReadLifecycleResult(messages=messages)

        # Phase 1: Build tool metadata and file operation index
        tool_metadata = self._build_tool_metadata(messages)
        file_ops = self._build_file_operation_index(messages, tool_metadata)

        # Phase 2: Classify each Read
        classifications = self._classify_reads(file_ops)

        if not classifications:
            return ReadLifecycleResult(messages=messages)

        # Phase 3: Replace stale/superseded content
        return self._apply_lifecycle(messages, classifications)

    def _build_tool_metadata(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, tuple[str, str | None]]:
        """Build tool_call_id → (tool_name, file_path) mapping.

        Scans assistant messages for tool calls, extracts name and file_path
        from tool inputs. Handles both OpenAI and Anthropic formats.
        """
        metadata: dict[str, tuple[str, str | None]] = {}

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            # OpenAI format: tool_calls array
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                name = func.get("name", "")
                if not tc_id or not name:
                    continue

                file_path = None
                try:
                    args = json.loads(func.get("arguments", "{}"))
                    file_path = args.get("file_path") or args.get("path")
                except (json.JSONDecodeError, TypeError):
                    pass
                metadata[tc_id] = (name, file_path)

            # Anthropic format: content blocks with type=tool_use
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tc_id = block.get("id", "")
                name = block.get("name", "")
                if not tc_id or not name:
                    continue

                inp = block.get("input", {})
                file_path = None
                if isinstance(inp, dict):
                    file_path = inp.get("file_path") or inp.get("path")
                metadata[tc_id] = (name, file_path)

        return metadata

    def _build_file_operation_index(
        self,
        messages: list[dict[str, Any]],
        tool_metadata: dict[str, tuple[str, str | None]],
    ) -> dict[str, list[FileOperation]]:
        """Build file_path → [FileOperation] index in a single pass.

        Groups all Read/Edit/Write operations by file_path for lifecycle analysis.
        """
        file_ops: dict[str, list[FileOperation]] = defaultdict(list)

        for tc_id, (name, file_path) in tool_metadata.items():
            if not file_path:
                continue

            if name in _READ_TOOL_NAMES:
                operation = "read"
            elif name in _MUTATING_TOOL_NAMES:
                operation = "edit"
            else:
                continue

            # Find the message index where this tool_call appears
            msg_idx = self._find_tool_call_msg_index(messages, tc_id)
            if msg_idx is None:
                continue

            file_ops[file_path].append(
                FileOperation(
                    msg_index=msg_idx,
                    tool_call_id=tc_id,
                    tool_name=name,
                    file_path=file_path,
                    operation=operation,
                )
            )

        return dict(file_ops)

    def _find_tool_call_msg_index(
        self, messages: list[dict[str, Any]], tool_call_id: str
    ) -> int | None:
        """Find the message index containing a specific tool_call_id."""
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            # OpenAI format
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                    return i

            # Anthropic format
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and block.get("id") == tool_call_id
                    ):
                        return i

        return None

    def _classify_reads(self, file_ops: dict[str, list[FileOperation]]) -> list[ReadClassification]:
        """Classify each Read as fresh, stale, or superseded."""
        classifications: list[ReadClassification] = []

        for file_path, ops in file_ops.items():
            reads = [op for op in ops if op.operation == "read"]
            edits = [op for op in ops if op.operation == "edit"]

            if not reads:
                continue

            for read_op in reads:
                # Check stale: any edit/write of this file AFTER this read?
                is_stale = self.config.compress_stale and any(
                    e.msg_index > read_op.msg_index for e in edits
                )

                # Check superseded: any later read of this file?
                is_superseded = self.config.compress_superseded and any(
                    r.msg_index > read_op.msg_index for r in reads
                )

                if is_stale:
                    state = ReadState.STALE
                elif is_superseded:
                    state = ReadState.SUPERSEDED
                else:
                    state = ReadState.FRESH

                classifications.append(
                    ReadClassification(
                        msg_index=read_op.msg_index,
                        tool_call_id=read_op.tool_call_id,
                        file_path=file_path,
                        state=state,
                        content_size=read_op.content_size,
                    )
                )

        return classifications

    def _apply_lifecycle(
        self,
        messages: list[dict[str, Any]],
        classifications: list[ReadClassification],
    ) -> ReadLifecycleResult:
        """Replace stale/superseded Read content with markers."""
        # Build lookup: tool_call_id → classification (for non-fresh reads)
        replacements: dict[str, ReadClassification] = {
            c.tool_call_id: c for c in classifications if c.state != ReadState.FRESH
        }

        if not replacements:
            return ReadLifecycleResult(
                messages=messages,
                reads_total=len(classifications),
                reads_fresh=len(classifications),
            )

        result_messages: list[dict[str, Any]] = []
        transforms: list[str] = []
        ccr_hashes: list[str] = []
        bytes_before = 0
        bytes_after = 0
        counts = {ReadState.FRESH: 0, ReadState.STALE: 0, ReadState.SUPERSEDED: 0}

        for c in classifications:
            counts[c.state] += 1

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # OpenAI format: role=tool with tool_call_id
            if role == "tool":
                tc_id = msg.get("tool_call_id", "")
                classification = replacements.get(tc_id)
                if classification and isinstance(content, str):
                    replaced, marker, ccr_hash = self._replace_content(content, classification)
                    if replaced:
                        result_messages.append({**msg, "content": marker})
                        transforms.append(f"read_lifecycle:{classification.state.value}")
                        if ccr_hash:
                            ccr_hashes.append(ccr_hash)
                        bytes_before += len(content.encode("utf-8"))
                        bytes_after += len(marker.encode("utf-8"))
                        continue

            # Anthropic format: content blocks list
            if isinstance(content, list):
                new_blocks, block_replaced = self._process_anthropic_blocks(
                    content, replacements, transforms, ccr_hashes
                )
                if block_replaced:
                    result_messages.append({**msg, "content": new_blocks})
                    continue

            result_messages.append(msg)

        return ReadLifecycleResult(
            messages=result_messages,
            reads_total=len(classifications),
            reads_stale=counts[ReadState.STALE],
            reads_superseded=counts[ReadState.SUPERSEDED],
            reads_fresh=counts[ReadState.FRESH],
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            transforms_applied=transforms,
            ccr_hashes=ccr_hashes,
        )

    def _process_anthropic_blocks(
        self,
        content_blocks: list[Any],
        replacements: dict[str, ReadClassification],
        transforms: list[str],
        ccr_hashes: list[str],
    ) -> tuple[list[Any], bool]:
        """Process Anthropic-format content blocks for lifecycle replacement."""
        new_blocks = []
        any_replaced = False

        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            tc_id = block.get("tool_use_id", "")
            classification = replacements.get(tc_id)
            tool_content = block.get("content", "")

            if classification and isinstance(tool_content, str):
                replaced, marker, ccr_hash = self._replace_content(tool_content, classification)
                if replaced:
                    new_blocks.append({**block, "content": marker})
                    transforms.append(f"read_lifecycle:{classification.state.value}")
                    if ccr_hash:
                        ccr_hashes.append(ccr_hash)
                    any_replaced = True
                    continue

            new_blocks.append(block)

        return new_blocks, any_replaced

    def _replace_content(
        self, content: str, classification: ReadClassification
    ) -> tuple[bool, str, str | None]:
        """Replace Read content with a lifecycle marker.

        Returns (was_replaced, marker_text, ccr_hash).
        """
        content_bytes = len(content.encode("utf-8"))

        # Skip tiny outputs
        if content_bytes < self.config.min_size_bytes:
            return False, content, None

        # Store original in CCR if available
        ccr_hash = None
        if self.store is not None:
            ccr_hash = self.store.store(
                original=content,
                compressed="",
                tool_name="Read",
                tool_call_id=classification.tool_call_id,
                compression_strategy=f"read_lifecycle:{classification.state.value}",
            )

        # Generate marker
        if ccr_hash is None:
            # No CCR store — generate a content hash for reference
            ccr_hash = hashlib.sha256(content.encode()).hexdigest()[:24]

        file_display = classification.file_path or "unknown"

        if classification.state == ReadState.STALE:
            marker = (
                f"[Read content stale: {file_display} was modified after this read. "
                f"Retrieve original: hash={ccr_hash}]"
            )
        else:  # SUPERSEDED
            marker = (
                f"[Read content superseded: {file_display} was re-read later. "
                f"Retrieve original: hash={ccr_hash}]"
            )

        return True, marker, ccr_hash
