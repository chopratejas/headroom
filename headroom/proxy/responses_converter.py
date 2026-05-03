"""Convert between OpenAI Responses API items and Chat Completions messages.

The Responses API uses a flat item model where function_call and
function_call_output are top-level items, content parts use input_text /
output_text types, and reasoning items must be preserved verbatim.

The Headroom compression pipeline works on Chat Completions messages
(role + content / tool_calls).  This module converts back and forth so
the existing pipeline can compress Responses API input without changes.

Pattern follows the Gemini converter in server.py (_gemini_contents_to_messages).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Content part types that indicate non-text media (must be preserved, not compressed)
_NON_TEXT_CONTENT_TYPES = frozenset(
    {
        "input_image",
        "input_file",
        "input_audio",
        "image_url",
        "image_file",
    }
)

# Top-level keys that the Chat Completions schema understands; everything
# else (notably Codex's `phase` field, but also any future Responses API
# extension) gets stashed in the side-channel and restored on rebuild.
# Per PR-A8 / P0-7 + P4-44 the previous "preserved by accident via
# copy.copy" path was one refactor away from silent data loss; this list
# is the explicit contract.
_CHAT_COMPLETIONS_KNOWN_KEYS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "type",
    }
)


def responses_items_to_messages(
    items: list[dict[str, Any]],
    *,
    request_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Convert Responses API input items to Chat Completions messages.

    Args:
        items: The ``input`` array from a ``/v1/responses`` request.
            Contains a mix of message items, function_call items,
            function_call_output items, reasoning items, etc.
        request_id: Optional request id for structured log lines when
            unknown item types are encountered (PR-A8 / P4-47).

    Returns:
        (messages, preserved_indices) where:
        - messages: OpenAI Chat Completions format messages suitable for
          the Headroom compression pipeline.
        - preserved_indices: Indices into *items* for entries that must
          be restored verbatim (reasoning, images, unknown types).
    """
    if not items:
        return [], []

    messages: list[dict[str, Any]] = []
    preserved_indices: list[int] = []
    pending_tool_calls: list[tuple[int, dict[str, Any]]] = []

    for idx, item in enumerate(items):
        item_type = item.get("type")
        role = item.get("role")

        # --- Reasoning items: preserve exactly ---
        if item_type == "reasoning":
            _flush_pending(messages, pending_tool_calls)
            preserved_indices.append(idx)
            continue

        # --- function_call items: accumulate, flush as one assistant message ---
        if item_type == "function_call":
            pending_tool_calls.append((idx, item))
            continue

        # --- function_call_output items: convert to role=tool ---
        if item_type == "function_call_output":
            _flush_pending(messages, pending_tool_calls)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }
            )
            continue

        # --- Message items (role-based, with or without type="message") ---
        if role is not None:
            _flush_pending(messages, pending_tool_calls)
            content = item.get("content", "")

            # Handle content part arrays
            if isinstance(content, list):
                if _has_non_text_parts(content):
                    preserved_indices.append(idx)
                    continue
                content = _extract_text_from_parts(content)

            mapped_role = "system" if role == "developer" else role
            messages.append({"role": mapped_role, "content": content})
            continue

        # --- Unknown item type: preserve + warn ---
        # PR-A8 / P4-47: emit a structured warning so operators see new
        # Codex item types in flight (apply_patch V4A, local_shell_call
        # argv, MCP items, compaction items). Until now this branch
        # survived only because the catch-all is conservative; one
        # refactor away from silent data loss without this signal.
        _flush_pending(messages, pending_tool_calls)
        logger.warning(
            "unknown_responses_item_type item_type=%s item_id=%s request_id=%s",
            item.get("type"),
            item.get("id"),
            request_id or "",
        )
        preserved_indices.append(idx)

    # Flush any trailing tool calls
    _flush_pending(messages, pending_tool_calls)

    return messages, preserved_indices


def messages_to_responses_items(
    messages: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    preserved_indices: list[int],
) -> list[dict[str, Any]]:
    """Convert compressed Chat Completions messages back to Responses API items.

    Uses a two-pass approach:
    1. Index compressed messages by call_id (for tool outputs) and collect
       regular messages in order.
    2. Walk original_items, restoring preserved items and substituting
       compressed content where applicable.

    Args:
        messages: Compressed messages from the pipeline.
        original_items: The original ``input`` array (pre-compression).
        preserved_indices: Indices returned by ``responses_items_to_messages``.

    Returns:
        New items list with compressed content, ready to send to OpenAI.
    """
    if not original_items:
        return []

    preserved_set = frozenset(preserved_indices)

    # --- Pass 1: Index compressed messages ---
    tool_outputs: dict[str, str] = {}  # call_id → compressed output
    regular_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tool_outputs[msg.get("tool_call_id", "")] = msg.get("content", "")
        elif role == "assistant" and msg.get("tool_calls"):
            # function_call items pass through uncompressed — skip
            pass
        else:
            regular_msgs.append(msg)

    # --- Pass 2: Reconstruct items ---
    result: list[dict[str, Any]] = []
    reg_idx = 0

    for orig_idx, item in enumerate(original_items):
        # Preserved items go back exactly as they were
        if orig_idx in preserved_set:
            result.append(item)
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            # Small — pass through unmodified
            result.append(item)

        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            compressed = tool_outputs.get(call_id, item.get("output", ""))
            result.append({**item, "output": compressed})

        else:
            # Regular message — take next compressed message
            if reg_idx < len(regular_msgs):
                msg = regular_msgs[reg_idx]
                reg_idx += 1
                result.append(_reconstruct_item(item, msg))
            else:
                # Safety: more original items than compressed messages
                result.append(item)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush_pending(
    messages: list[dict[str, Any]],
    pending: list[tuple[int, dict[str, Any]]],
) -> None:
    """Flush accumulated function_call items as one assistant message."""
    if not pending:
        return
    tool_calls = []
    for _idx, item in pending:
        tool_calls.append(
            {
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    )
    pending.clear()


def _has_non_text_parts(content: list[dict[str, Any]]) -> bool:
    """Check if a content array contains non-text parts (images, files, audio)."""
    return any(p.get("type") in _NON_TEXT_CONTENT_TYPES for p in content)


def _extract_text_from_parts(content: list[dict[str, Any]]) -> str:
    """Extract text from Responses API content parts.

    Handles both input (input_text) and output (output_text) part types,
    plus standard ``text`` parts.
    """
    parts = []
    for p in content:
        ptype = p.get("type", "")
        if ptype in ("input_text", "output_text", "text"):
            parts.append(p.get("text", ""))
    return "\n".join(parts) if parts else ""


def _reconstruct_item(
    original: dict[str, Any],
    compressed_msg: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a Responses API item from its original structure + compressed text.

    Preserves the original content format: if the original had
    ``content: [{"type": "input_text", ...}]``, the compressed text goes
    back into that same structure rather than being flattened to a string.

    PR-A8 / P0-7 + P4-44: phase preservation is now **explicit**. The
    previous code path relied on ``copy.copy(original)`` to "accidentally"
    carry the ``phase`` field through; here we copy + explicitly verify
    every Responses-API-only key (anything outside
    ``_CHAT_COMPLETIONS_KNOWN_KEYS``) is present after the rebuild.

    Multi-text-part contract: when ``original`` has N text parts and
    compression collapses them into a single string (the pipeline
    currently joins with ``\\n`` in ``_extract_text_from_parts``), we
    rebuild a SINGLE text part (the first one in source order) carrying
    the full compressed text. The remaining text parts are dropped to
    avoid the doubling bug where compressed-content + original-content
    both ended up on the wire (P0-7). Non-text parts (images, files,
    audio) are passed through verbatim by index. If a future
    pipeline emits multiple compressed strings per item, this
    contract must be revisited.
    """
    compressed_text = compressed_msg.get("content", "")
    original_content = original.get("content")

    # If original had a content-part array, reconstruct it.
    if isinstance(original_content, list) and original_content:
        new_content: list[dict[str, Any]] = []
        text_replaced = False
        for part in original_content:
            ptype = part.get("type", "")
            if ptype in ("input_text", "output_text", "text"):
                if not text_replaced:
                    # Replace the FIRST text part's text in place;
                    # preserve any other fields on the part (e.g.
                    # `annotations`, future extensions).
                    new_content.append({**part, "text": compressed_text})
                    text_replaced = True
                # else: drop this trailing text part — its content
                # was already absorbed into the compressed string
                # via ``_extract_text_from_parts``. Keeping it would
                # double the content on the wire (the P0-7 bug).
            else:
                # Non-text parts (images, files, audio) pass through
                # by index; their position relative to the text
                # parts is preserved.
                new_content.append(part)
        rebuilt = copy.copy(original)
        rebuilt["content"] = new_content
        # Explicit Codex `phase` preservation (PR-A8 / P0-7 + P4-44):
        # ``copy.copy`` already shallow-copies the field, but make the
        # contract loud so future refactors don't silently lose it.
        if "phase" in original:
            rebuilt["phase"] = original["phase"]
        return rebuilt

    # String content or missing — just replace
    rebuilt = copy.copy(original)
    rebuilt["content"] = compressed_text if compressed_text is not None else ""
    if "phase" in original:
        rebuilt["phase"] = original["phase"]
    return rebuilt
