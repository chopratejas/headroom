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
from typing import Any

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


def responses_items_to_messages(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Convert Responses API input items to Chat Completions messages.

    Args:
        items: The ``input`` array from a ``/v1/responses`` request.
            Contains a mix of message items, function_call items,
            function_call_output items, reasoning items, etc.

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

        # --- Unknown item type: preserve ---
        _flush_pending(messages, pending_tool_calls)
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
    """
    compressed_text = compressed_msg.get("content", "")
    original_content = original.get("content")

    # If original had a content-part array, reconstruct it
    if isinstance(original_content, list) and original_content:
        new_content = []
        text_replaced = False
        for part in original_content:
            ptype = part.get("type", "")
            if ptype in ("input_text", "output_text", "text") and not text_replaced:
                new_content.append({**part, "text": compressed_text})
                text_replaced = True
            else:
                new_content.append(part)
        rebuilt = copy.copy(original)
        rebuilt["content"] = new_content
        return rebuilt

    # String content or missing — just replace
    rebuilt = copy.copy(original)
    rebuilt["content"] = compressed_text if compressed_text is not None else ""
    return rebuilt
