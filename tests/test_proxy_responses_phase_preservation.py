"""PR-A8 / P0-7 + P4-44 + P4-47: Responses API converter contract.

These tests pin:

- The Codex-emitted ``phase`` field (e.g. ``commentary`` /
  ``final_answer``) survives a forward + reverse trip through the
  Chat-Completions normalizer. Pre-A8 the field was preserved only
  *accidentally* via ``copy.copy(original)``; one refactor away from
  silent data loss.
- Multi-text-part rebuild collapses to a single text part on the way
  back instead of doubling content (the P0-7 bug).
- Unknown item types log a structured warning (P4-47) so operators see
  new Codex item types in flight.
"""

from __future__ import annotations

import json
import logging

from headroom.proxy.responses_converter import (
    messages_to_responses_items,
    responses_items_to_messages,
)


def test_codex_phase_commentary_preserved_through_compression() -> None:
    """`phase: commentary` survives forward + reverse conversion."""
    items = [
        {
            "type": "message",
            "id": "msg_codex_1",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Here's my reasoning."}],
        },
    ]
    messages, preserved = responses_items_to_messages(items)
    assert preserved == []
    assert len(messages) == 1
    # Pretend the pipeline compressed nothing — feed messages back.
    rebuilt = messages_to_responses_items(messages, items, preserved)
    assert len(rebuilt) == 1
    out = rebuilt[0]
    assert out["phase"] == "commentary"
    assert out["id"] == "msg_codex_1"
    assert out["role"] == "assistant"


def test_codex_phase_final_answer_preserved() -> None:
    """`phase: final_answer` survives even when content is compressed shorter."""
    items = [
        {
            "type": "message",
            "id": "msg_codex_2",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Long original answer ..."}],
        },
    ]
    messages, preserved = responses_items_to_messages(items)
    # Simulate compression: replace the message content with a shorter string.
    messages[0]["content"] = "Short."
    rebuilt = messages_to_responses_items(messages, items, preserved)
    assert rebuilt[0]["phase"] == "final_answer"
    # Content was replaced inside the original part structure.
    assert rebuilt[0]["content"][0]["text"] == "Short."


def test_unknown_item_type_logs_warning_byte_equal() -> None:
    """Unknown item types preserve the item AND log a structured warning.

    Capture is done by attaching a handler directly to the named logger
    rather than relying on `caplog`. Other tests in the suite (proxy
    file-logging setup) flip `headroom.*.propagate = False`, which breaks
    pytest's root-attached caplog handler in unrelated test runs. A
    direct handler is order-independent.
    """
    items = [
        {
            "type": "apply_patch_v4a",
            "id": "patch_1",
            "patch": "--- a\n+++ b\n",
        },
    ]
    target = logging.getLogger("headroom.proxy.responses_converter")
    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _CaptureHandler(level=logging.WARNING)
    prev_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:
        messages, preserved = responses_items_to_messages(items, request_id="req-xyz")
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)

    # Item is preserved (byte-equal) on the rebuild side.
    assert preserved == [0]
    rebuilt = messages_to_responses_items(messages, items, preserved)
    assert rebuilt == items
    # A structured warning fired with the unknown type.
    matched = [r for r in captured if "unknown_responses_item_type" in r.getMessage()]
    assert matched, "expected unknown_responses_item_type warning log line"
    msg = matched[0].getMessage()
    assert "apply_patch_v4a" in msg
    assert "patch_1" in msg
    assert "req-xyz" in msg


def test_multi_text_part_rebuild_no_doubling() -> None:
    """Two-text-part input is collapsed to a single part on rebuild — no doubling."""
    items = [
        {
            "type": "message",
            "id": "msg_multi",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "First paragraph."},
                {"type": "input_text", "text": "Second paragraph."},
            ],
        },
    ]
    messages, preserved = responses_items_to_messages(items)
    # The two text parts join with `\n`.
    assert messages[0]["content"] == "First paragraph.\nSecond paragraph."
    # Pretend compression replaced with a shortened single string.
    messages[0]["content"] = "Para one. Para two."
    rebuilt = messages_to_responses_items(messages, items, preserved)
    out_content = rebuilt[0]["content"]
    # Critical: there is now exactly ONE text part on the rebuild,
    # carrying the compressed string. The previous bug left a
    # second part with the ORIGINAL "Second paragraph." text and
    # the compressed string in part 0 — content doubled on the wire.
    assert isinstance(out_content, list)
    text_parts = [p for p in out_content if p.get("type") == "input_text"]
    assert len(text_parts) == 1, (
        f"expected exactly 1 text part on rebuild, got {len(text_parts)} — "
        f"this is the P0-7 doubling bug regression"
    )
    assert text_parts[0]["text"] == "Para one. Para two."
    # And the round-trip JSON is well-formed.
    assert json.dumps(rebuilt)
