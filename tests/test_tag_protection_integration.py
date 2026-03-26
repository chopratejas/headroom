"""Integration tests: verify tags survive the full compress() pipeline."""

from __future__ import annotations

import os

import pytest


class TestCompressPreservesTags:
    """End-to-end tests: compress() with tagged content."""

    def test_system_reminder_survives(self):
        """<system-reminder> tags in tool output survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "What are the rules?"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Here is a very long and verbose explanation of the system rules "
                    "that contains a lot of unnecessary filler words and repetitive "
                    "phrasing that could be compressed significantly. "
                    "<system-reminder>You must always validate input before processing. "
                    "Never skip authentication checks.</system-reminder> "
                    "The rest of this text is also quite verbose and contains additional "
                    "unnecessary details that add tokens without adding information value "
                    "to the overall response that the language model needs to generate."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<system-reminder>" in output
        assert "</system-reminder>" in output
        assert "validate input" in output
        assert "authentication checks" in output

    def test_tool_call_tags_survive(self):
        """<tool_call> tags survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "Search for results"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Processing the search request with extensive verbose output "
                    "that includes many redundant descriptions and unnecessary detail. "
                    '<tool_call>{"name": "search", "args": {"query": "test"}}</tool_call> '
                    "Additional verbose context that repeats information already stated "
                    "in the previous paragraphs about the search functionality and its "
                    "various capabilities and features that are not directly relevant."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<tool_call>" in output
        assert "</tool_call>" in output
        assert '"name": "search"' in output

    def test_thinking_tags_survive(self):
        """<thinking> tags survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "Analyze this data"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "The analysis produced extensive results with much verbose "
                    "explanatory text that describes methodology in great detail. "
                    "<thinking>Step 1: Parse input. Step 2: Validate schema. "
                    "Step 3: Run inference.</thinking> "
                    "Further verbose explanation of the analytical process and its "
                    "various stages and intermediate results and observations that "
                    "could be expressed much more concisely without losing meaning."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<thinking>" in output
        assert "</thinking>" in output
        assert "Step 1: Parse input" in output

    def test_html_tags_still_compressible(self):
        """Standard HTML tags are NOT protected — they're just text to the compressor."""
        from headroom.transforms.tag_protector import protect_tags

        html_text = "<div>Some content</div> <span>More content</span>"
        cleaned, protected = protect_tags(html_text)

        # HTML tags should NOT be protected
        assert protected == []
        assert "<div>" in cleaned

    def test_multiple_custom_tags_in_messages(self):
        """Multiple different custom tags all survive."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "What should I do?"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Very verbose introductory text with many unnecessary words. "
                    "<system-reminder>Rule 1: Always validate</system-reminder> "
                    "More verbose middle text repeating previous information. "
                    "<context>session_id=abc-123</context> "
                    "Additional verbose concluding text with redundant information. "
                    "<IMPORTANT>Never expose API keys</IMPORTANT> "
                    "Final paragraph of verbose text."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<system-reminder>" in output
        assert "<context>" in output
        assert "<IMPORTANT>" in output


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestRealAPIWithTags:
    """Real API integration: verify Claude can use tagged content after compression."""

    def test_compressed_tags_usable_by_llm(self):
        """Send compressed content with tags to Claude, verify it references them."""
        from anthropic import Anthropic

        from headroom import compress

        client = Anthropic()

        messages = [
            {"role": "user", "content": "What is the secret code in the system reminder?"},
            {
                "role": "assistant",
                "content": "Let me check the system configuration.",
            },
            {
                "role": "user",
                "content": (
                    "Here is a very long and detailed configuration document with "
                    "extensive verbose descriptions of various system parameters and "
                    "their default values and recommended settings for production "
                    "deployment scenarios across different cloud providers. "
                    "<system-reminder>The secret code is ALPHA-7742-BRAVO</system-reminder> "
                    "Additional verbose documentation about system architecture and "
                    "deployment patterns and scaling strategies and monitoring setup "
                    "and alerting configuration and backup procedures."
                ),
            },
        ]

        result = compress(messages, model="claude-sonnet-4-5-20250929")

        # Verify tags survived compression
        last_content = str(result.messages[-1].get("content", ""))
        assert "<system-reminder>" in last_content
        assert "ALPHA-7742-BRAVO" in last_content

        # Send to Claude and verify it can read the tagged content
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=result.messages,
        )
        answer = response.content[0].text.lower()
        assert "alpha-7742-bravo" in answer or "alpha" in answer
