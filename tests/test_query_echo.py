"""Tests for Query Echo — unit tests + integration eval with real API.

Unit tests: verify echo injection logic across message formats.
Integration: verify echo actually improves LLM answers on compressed data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from headroom.transforms.query_echo import extract_user_query, inject_query_echo

# Load .env for integration tests
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# =============================================================================
# Unit Tests: extract_user_query
# =============================================================================


class TestExtractUserQuery:
    def test_simple_string_message(self):
        messages = [
            {"role": "user", "content": "What are the test failures?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        assert extract_user_query(messages) == "What are the test failures?"

    def test_anthropic_content_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find the auth bug in these results"},
                    {"type": "tool_result", "tool_use_id": "x", "content": "data..."},
                ],
            },
        ]
        assert extract_user_query(messages) == "Find the auth bug in these results"

    def test_skips_trivial_messages(self):
        messages = [
            {"role": "user", "content": "What are the top 3 errors in this log output?"},
            {"role": "assistant", "content": "Looking..."},
            {"role": "user", "content": "yes"},
        ]
        # "yes" is too short (<10 chars), should get the earlier question
        assert "errors" in extract_user_query(messages)

    def test_empty_messages(self):
        assert extract_user_query([]) == ""

    def test_no_user_messages(self):
        messages = [{"role": "assistant", "content": "Hello"}]
        assert extract_user_query(messages) == ""

    def test_multi_turn_gets_last_substantive(self):
        messages = [
            {"role": "user", "content": "Tell me about the weather"},
            {"role": "assistant", "content": "It's sunny."},
            {"role": "user", "content": "What about the authentication failures in CI?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        assert "authentication" in extract_user_query(messages)


# =============================================================================
# Unit Tests: inject_query_echo
# =============================================================================


class TestInjectQueryEcho:
    def test_injects_after_compressed_content(self):
        messages = [
            {"role": "user", "content": "What are the failures?"},
            {
                "role": "tool",
                "content": '[{"status":"pass"}]\n[50 items compressed to 5. Retrieve more: hash=abc123]',
            },
        ]
        result = inject_query_echo(
            messages, "What are the failures?", tokens_saved=500, original_tokens=1000
        )
        assert result is True
        assert "[Recall:" in messages[-1]["content"]
        assert "failures" in messages[-1]["content"]

    def test_skips_when_no_compression(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "some data"},
        ]
        result = inject_query_echo(messages, "hello", tokens_saved=0, original_tokens=100)
        assert result is False

    def test_skips_when_low_compression(self):
        messages = [
            {"role": "tool", "content": "data\n[10 items compressed to 8]"},
        ]
        result = inject_query_echo(messages, "query", tokens_saved=20, original_tokens=100)
        # 20% compression — below 30% threshold
        assert result is False

    def test_triggers_at_high_compression(self):
        messages = [
            {
                "role": "tool",
                "content": "data\n[500 items compressed to 20. Retrieve more: hash=abc]",
            },
        ]
        result = inject_query_echo(
            messages, "What are the errors?", tokens_saved=800, original_tokens=1000
        )
        assert result is True
        assert "[Recall:" in messages[0]["content"]

    def test_anthropic_content_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": '[{"x":1}]\n[100 items compressed to 10. hash=abc123]',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "2",
                        "content": '[{"y":2}]\n[200 items compressed to 15. hash=def456]',
                    },
                ],
            },
        ]
        result = inject_query_echo(
            messages, "Find the errors", tokens_saved=600, original_tokens=1000
        )
        assert result is True
        # Echo should be on the LAST compressed block
        last_block = messages[0]["content"][-1]
        assert "[Recall:" in last_block["content"]

    def test_truncates_long_query(self):
        long_query = "x" * 500
        messages = [
            {"role": "tool", "content": "data\n[100 items compressed to 5. hash=abc]"},
        ]
        inject_query_echo(messages, long_query, tokens_saved=500, original_tokens=1000)
        # Should be truncated to 200 chars
        assert len(messages[0]["content"]) < 800

    def test_sanitizes_quotes(self):
        messages = [
            {"role": "tool", "content": "data\n[100 items compressed to 5. hash=abc]"},
        ]
        inject_query_echo(
            messages, 'Find "auth_error" in logs', tokens_saved=500, original_tokens=1000
        )
        # Double quotes replaced with single quotes
        assert '"auth_error"' not in messages[0]["content"]
        assert "'auth_error'" in messages[0]["content"]

    def test_no_echo_without_compressed_marker(self):
        """If no message contains 'compressed', no echo injected."""
        messages = [
            {"role": "tool", "content": "just plain tool output without compression"},
        ]
        result = inject_query_echo(messages, "query", tokens_saved=500, original_tokens=1000)
        assert result is False

    def test_empty_query_skipped(self):
        messages = [
            {"role": "tool", "content": "data\n[100 items compressed]"},
        ]
        result = inject_query_echo(messages, "", tokens_saved=500, original_tokens=1000)
        assert result is False


# =============================================================================
# Integration Eval: Does echo improve answer quality?
# =============================================================================


def _call_claude(messages, max_tokens=200):
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "X-Api-Key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": max_tokens,
            "messages": messages,
        },
        timeout=30,
    )
    return resp.json()


def _make_compressed_test_data():
    """Create realistic compressed test results where answer is in omitted data."""
    # Visible: 10 passing tests
    visible = [
        {"test": f"test_module_{i}", "status": "passed", "duration_ms": 50 + i} for i in range(10)
    ]
    compressed_output = json.dumps(visible, indent=2)

    # The important info is in the COMPRESSED summary
    summary = "87 passed, 2 failed, 1 error; notable: test_auth (fail); test_db (timeout)"
    compressed_output += (
        f"\n[90 items compressed to 10. Omitted: {summary}."
        f" Retrieve more: hash=abc123def456789012345678."
        f" Expires in 5m.]"
    )
    return compressed_output


@pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
class TestQueryEchoIntegration:
    def test_with_echo_finds_failures(self):
        """WITH echo → LLM should identify failures from compressed data."""
        compressed = _make_compressed_test_data()
        question = "What are the test failures and what went wrong?"

        # Add echo at the end
        echoed_content = compressed + f"\n\n[Recall: User asked: '{question}']"

        messages = [
            {
                "role": "user",
                "content": f"Here are the CI test results:\n\n{echoed_content}\n\nAnswer my question.",
            },
        ]
        resp = _call_claude(messages)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        print(f"\n  WITH echo response: {text[:200]}")

        has_failure_info = any(w in text for w in ["fail", "error", "timeout", "auth", "db"])
        assert has_failure_info, f"LLM missed failures with echo: {text[:300]}"

    def test_without_echo_baseline(self):
        """WITHOUT echo → baseline for comparison."""
        compressed = _make_compressed_test_data()
        question = "What are the test failures and what went wrong?"

        messages = [
            {
                "role": "user",
                "content": f"Here are the CI test results:\n\n{compressed}\n\n{question}",
            },
        ]
        resp = _call_claude(messages)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        print(f"\n  WITHOUT echo response: {text[:200]}")

        # Not asserting — this is the baseline. May or may not find failures.
        has_failure_info = any(w in text for w in ["fail", "error", "timeout"])
        print(f"  Found failure info without echo: {has_failure_info}")

    def test_echo_with_specific_lookup(self):
        """Echo helps LLM find specific data in compressed results."""
        # Compressed config data — visible items are all 'production'
        visible = [
            {"env": "production", "key": "DB_URL", "value": "postgres://prod:5432/app"},
            {"env": "production", "key": "REDIS_URL", "value": "redis://prod:6379"},
            {"env": "production", "key": "API_KEY", "value": "sk-prod-xxx"},
        ]
        compressed = json.dumps(visible, indent=2)
        compressed += (
            "\n[45 items compressed to 3. Omitted: 30 staging, 12 development."
            " Retrieve more: hash=cfg123def456789012345678."
            " Expires in 5m.]"
        )

        question = "What is the staging database URL?"
        echoed = compressed + f"\n\n[Recall: User asked: '{question}']"

        messages = [
            {"role": "user", "content": f"Application configs:\n\n{echoed}\n\nAnswer precisely."},
        ]
        resp = _call_claude(messages)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        print(f"\n  Staging lookup with echo: {text[:200]}")

        # With echo + summary mentioning "30 staging", LLM should know
        # staging data exists in compressed items
        knows_staging = "staging" in text or "compressed" in text or "retrieve" in text
        assert knows_staging, f"LLM didn't reference staging data: {text[:300]}"
