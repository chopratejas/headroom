"""Tests for inline memory extraction (Letta-style)."""

from __future__ import annotations

from headroom.memory.inline_extractor import (
    MEMORY_INSTRUCTION,
    MEMORY_INSTRUCTION_SHORT,
    ParsedResponse,
    inject_memory_instruction,
    parse_response_with_memory,
)


class TestParseResponseWithMemory:
    """Test response parsing to extract memories."""

    def test_extracts_single_memory(self):
        """Parse response with one memory."""
        response = """Great choice! Python is excellent for backend development.

<memory>{"memories": [{"content": "User prefers Python", "category": "preference"}]}</memory>"""

        parsed = parse_response_with_memory(response)

        assert (
            parsed.content.strip() == "Great choice! Python is excellent for backend development."
        )
        assert len(parsed.memories) == 1
        assert parsed.memories[0]["content"] == "User prefers Python"
        assert parsed.memories[0]["category"] == "preference"
        assert parsed.raw == response

    def test_extracts_multiple_memories(self):
        """Parse response with multiple memories."""
        response = """That's interesting background!

<memory>{"memories": [
    {"content": "Works at fintech startup", "category": "fact"},
    {"content": "Uses PostgreSQL", "category": "preference"}
]}</memory>"""

        parsed = parse_response_with_memory(response)

        assert len(parsed.memories) == 2
        assert parsed.memories[0]["content"] == "Works at fintech startup"
        assert parsed.memories[1]["content"] == "Uses PostgreSQL"

    def test_handles_empty_memories(self):
        """Parse response with no memories."""
        response = """Hello! How can I help?

<memory>{"memories": []}</memory>"""

        parsed = parse_response_with_memory(response)

        assert "Hello! How can I help?" in parsed.content
        assert len(parsed.memories) == 0

    def test_handles_no_memory_block(self):
        """Parse response without memory block."""
        response = "Just a normal response without memory."

        parsed = parse_response_with_memory(response)

        assert parsed.content == response
        assert len(parsed.memories) == 0

    def test_handles_malformed_json(self):
        """Parse response with invalid JSON in memory block."""
        response = """Some response.

<memory>this is not valid json</memory>"""

        parsed = parse_response_with_memory(response)

        assert "Some response" in parsed.content
        assert len(parsed.memories) == 0  # Gracefully handles error

    def test_case_insensitive_tags(self):
        """Memory tags should be case-insensitive."""
        response = """Response here.

<MEMORY>{"memories": [{"content": "Test", "category": "fact"}]}</MEMORY>"""

        parsed = parse_response_with_memory(response)

        assert len(parsed.memories) == 1

    def test_memory_block_in_middle(self):
        """Memory block can appear anywhere in response."""
        response = """First part.

<memory>{"memories": [{"content": "Test", "category": "fact"}]}</memory>

More content after."""

        parsed = parse_response_with_memory(response)

        assert len(parsed.memories) == 1
        assert "First part" in parsed.content
        assert "More content after" in parsed.content


class TestInjectMemoryInstruction:
    """Test injection of memory instruction into messages."""

    def test_appends_to_existing_system_prompt(self):
        """Instruction appended to existing system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = inject_memory_instruction(messages, short=True)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "memory" in result[0]["content"].lower()

    def test_creates_system_prompt_if_missing(self):
        """Creates system message if none exists."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        result = inject_memory_instruction(messages, short=True)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "memory" in result[0]["content"].lower()

    def test_does_not_modify_original(self):
        """Original messages list is not modified."""
        messages = [
            {"role": "system", "content": "Original prompt."},
            {"role": "user", "content": "Hello"},
        ]
        original_content = messages[0]["content"]

        inject_memory_instruction(messages, short=True)

        assert messages[0]["content"] == original_content

    def test_short_vs_long_instruction(self):
        """Short instruction is shorter than full instruction."""
        messages = [{"role": "user", "content": "Hello"}]

        short = inject_memory_instruction(messages, short=True)
        long = inject_memory_instruction(messages, short=False)

        assert len(short[0]["content"]) < len(long[0]["content"])

    def test_instruction_contains_required_format(self):
        """Instruction explains the memory format."""
        messages = [{"role": "user", "content": "Hello"}]

        result = inject_memory_instruction(messages, short=False)
        content = result[0]["content"]

        assert "<memory>" in content
        assert "memories" in content
        assert "category" in content


class TestParsedResponse:
    """Test ParsedResponse dataclass."""

    def test_dataclass_fields(self):
        """ParsedResponse has expected fields."""
        parsed = ParsedResponse(
            content="Hello",
            memories=[{"content": "Test", "category": "fact"}],
            raw="Hello\n<memory>...</memory>",
        )

        assert parsed.content == "Hello"
        assert len(parsed.memories) == 1
        assert parsed.raw == "Hello\n<memory>...</memory>"


class TestMemoryInstructions:
    """Test memory instruction prompts."""

    def test_short_instruction_contains_essentials(self):
        """Short instruction has minimum required info."""
        assert "<memory>" in MEMORY_INSTRUCTION_SHORT
        assert "memories" in MEMORY_INSTRUCTION_SHORT
        assert "category" in MEMORY_INSTRUCTION_SHORT

    def test_full_instruction_more_detailed(self):
        """Full instruction has categories explained."""
        assert "preference" in MEMORY_INSTRUCTION
        assert "fact" in MEMORY_INSTRUCTION
        assert "context" in MEMORY_INSTRUCTION
        assert "Greetings" in MEMORY_INSTRUCTION or "greeting" in MEMORY_INSTRUCTION.lower()
