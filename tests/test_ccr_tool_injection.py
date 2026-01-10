"""Tests for CCR tool injection and MCP integration."""

import json

from headroom.ccr import (
    CCR_TOOL_NAME,
    CCRToolInjector,
    create_ccr_tool_definition,
    create_system_instructions,
    parse_tool_call,
)


class TestCCRToolDefinition:
    """Test tool definition creation for different providers."""

    def test_anthropic_format(self):
        """Anthropic tool definition has correct format."""
        tool = create_ccr_tool_definition("anthropic")

        assert tool["name"] == CCR_TOOL_NAME
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "hash" in tool["input_schema"]["properties"]
        assert "query" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["hash"]

    def test_openai_format(self):
        """OpenAI tool definition has correct format."""
        tool = create_ccr_tool_definition("openai")

        assert tool["type"] == "function"
        assert tool["function"]["name"] == CCR_TOOL_NAME
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["required"] == ["hash"]

    def test_google_format(self):
        """Google tool definition has correct format."""
        tool = create_ccr_tool_definition("google")

        assert tool["name"] == CCR_TOOL_NAME
        assert "parameters" in tool
        assert tool["parameters"]["required"] == ["hash"]


class TestCCRToolInjector:
    """Test CCRToolInjector functionality."""

    def test_scan_for_markers_finds_hash(self):
        """Scanner detects compression markers in messages."""
        messages = [
            {"role": "user", "content": "Find errors"},
            {
                "role": "tool",
                "content": '[{"id": 1}]\n[100 items compressed to 10. Retrieve more: hash=abc123def456]',
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1
        assert "abc123def456" in hashes
        assert injector.has_compressed_content

    def test_scan_for_markers_multiple_hashes(self):
        """Scanner finds multiple distinct hashes."""
        messages = [
            {
                "role": "tool",
                "content": "[50 items compressed to 5. Retrieve more: hash=aaa111111111]",
            },
            {
                "role": "tool",
                "content": "[200 items compressed to 20. Retrieve more: hash=bbb222222222]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 2
        assert "aaa111111111" in hashes
        assert "bbb222222222" in hashes

    def test_scan_no_duplicates(self):
        """Scanner deduplicates repeated hashes."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=aabbcc123456]",
            },
            {
                "role": "assistant",
                "content": "I see [100 items compressed to 10. Retrieve more: hash=aabbcc123456]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1

    def test_scan_anthropic_content_blocks(self):
        """Scanner handles Anthropic's content block format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find errors"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "[100 items compressed to 10. Retrieve more: hash=b10cf0a2b3c4]",
                    },
                ],
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert "b10cf0a2b3c4" in hashes

    def test_inject_tool_when_compression_detected(self):
        """Tool is injected when compression markers are found."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456]",
            },
        ]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(None)

        assert was_injected
        assert len(tools) == 1
        assert tools[0]["name"] == CCR_TOOL_NAME

    def test_inject_tool_adds_to_existing(self):
        """CCR tool is added to existing tools list."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=e1e2e3f4f5f6]",
            },
        ]
        existing_tools = [{"name": "other_tool", "input_schema": {}}]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert was_injected
        assert len(tools) == 2
        assert tools[0]["name"] == "other_tool"
        assert tools[1]["name"] == CCR_TOOL_NAME

    def test_skip_injection_if_tool_present_anthropic(self):
        """Injection skipped if tool already present (Anthropic format)."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=aac123456789]",
            },
        ]
        # Tool already present (e.g., from MCP)
        existing_tools = [{"name": CCR_TOOL_NAME, "input_schema": {}}]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert not was_injected
        assert len(tools) == 1  # Not duplicated

    def test_skip_injection_if_tool_present_openai(self):
        """Injection skipped if tool already present (OpenAI format)."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=bbc456789012]",
            },
        ]
        # OpenAI format tool already present
        existing_tools = [
            {"type": "function", "function": {"name": CCR_TOOL_NAME, "parameters": {}}}
        ]

        injector = CCRToolInjector(provider="openai")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert not was_injected
        assert len(tools) == 1

    def test_no_injection_without_compression(self):
        """No injection when no compression markers found."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": '{"result": "ok"}'},
        ]

        injector = CCRToolInjector()
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(None)

        assert not was_injected
        assert tools == []

    def test_inject_system_instructions(self):
        """System instructions are injected when compression detected."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456]",
            },
        ]

        injector = CCRToolInjector(inject_system_instructions=True)
        injector.scan_for_markers(messages)
        updated = injector.inject_into_system_message(messages)

        assert "Compressed Context Available" in updated[0]["content"]
        assert "abc123def456" in updated[0]["content"]

    def test_process_request_full_flow(self):
        """process_request handles complete injection flow."""
        messages = [
            {"role": "system", "content": "Assistant"},
            {"role": "user", "content": "Search for errors"},
            {
                "role": "tool",
                "content": "[500 items compressed to 25. Retrieve more: hash=f011f10abcde]",
            },
        ]

        injector = CCRToolInjector(
            provider="anthropic",
            inject_tool=True,
            inject_system_instructions=True,
        )
        updated_messages, updated_tools, was_injected = injector.process_request(messages, None)

        assert was_injected
        assert updated_tools is not None
        assert len(updated_tools) == 1
        assert updated_tools[0]["name"] == CCR_TOOL_NAME
        assert "Compressed Context Available" in updated_messages[0]["content"]


class TestParseToolCall:
    """Test parsing of tool calls from LLM responses."""

    def test_parse_anthropic_format(self):
        """Parse Anthropic tool call format."""
        tool_call = {
            "id": "toolu_123",
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123", "query": "errors"},
        }

        hash_key, query = parse_tool_call(tool_call, "anthropic")

        assert hash_key == "abc123"
        assert query == "errors"

    def test_parse_openai_format(self):
        """Parse OpenAI tool call format."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": CCR_TOOL_NAME,
                "arguments": json.dumps({"hash": "def456", "query": None}),
            },
        }

        hash_key, query = parse_tool_call(tool_call, "openai")

        assert hash_key == "def456"
        assert query is None

    def test_parse_non_ccr_tool(self):
        """Returns None for non-CCR tool calls."""
        tool_call = {
            "name": "other_tool",
            "input": {"param": "value"},
        }

        hash_key, query = parse_tool_call(tool_call, "anthropic")

        assert hash_key is None
        assert query is None

    def test_parse_malformed_openai_args(self):
        """Handles malformed JSON in OpenAI arguments."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": CCR_TOOL_NAME,
                "arguments": "not valid json",
            },
        }

        hash_key, query = parse_tool_call(tool_call, "openai")

        assert hash_key is None


class TestSystemInstructions:
    """Test system instruction generation."""

    def test_create_instructions_single_hash(self):
        """Instructions include single hash."""
        instructions = create_system_instructions(["hash123"])

        assert "hash123" in instructions
        assert CCR_TOOL_NAME in instructions
        assert "Compressed Context Available" in instructions

    def test_create_instructions_multiple_hashes(self):
        """Instructions include multiple hashes."""
        hashes = ["hash1", "hash2", "hash3"]
        instructions = create_system_instructions(hashes)

        for h in hashes:
            assert h in instructions

    def test_create_instructions_truncates_many_hashes(self):
        """Instructions truncate when many hashes present."""
        hashes = [f"hash{i}" for i in range(10)]
        instructions = create_system_instructions(hashes)

        # First 5 should be present, rest truncated
        assert "hash0" in instructions
        assert "hash4" in instructions
        assert "..." in instructions
