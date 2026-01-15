"""Tests for memory extractor.

Mocks LLM HTTP responses to test extraction logic without external calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from headroom.memory.extractor import (
    CHEAP_MODELS,
    MemoryExtractor,
    detect_provider,
    get_cheap_model,
)


class TestProviderDetection:
    """Test provider detection from client class."""

    def test_detect_openai(self):
        """Detect OpenAI from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "openai.resources.chat"

        result = detect_provider(mock_client)

        assert result == "openai"

    def test_detect_anthropic(self):
        """Detect Anthropic from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "anthropic.resources"

        result = detect_provider(mock_client)

        assert result == "anthropic"

    def test_detect_groq(self):
        """Detect Groq from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "groq.resources"

        result = detect_provider(mock_client)

        assert result == "groq"

    def test_detect_together(self):
        """Detect Together from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "together.client"

        result = detect_provider(mock_client)

        assert result == "together"

    def test_detect_fireworks(self):
        """Detect Fireworks from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "fireworks.client"

        result = detect_provider(mock_client)

        assert result == "fireworks"

    def test_detect_mistralai(self):
        """Detect Mistral from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "mistralai.client"

        result = detect_provider(mock_client)

        assert result == "mistralai"

    def test_detect_cohere(self):
        """Detect Cohere from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "cohere.client"

        result = detect_provider(mock_client)

        assert result == "cohere"

    def test_detect_google(self):
        """Detect Google from module path."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "google.generativeai"

        result = detect_provider(mock_client)

        assert result == "google"

    def test_detect_unknown_returns_none(self):
        """Unknown provider returns None."""
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "some.unknown.provider"

        result = detect_provider(mock_client)

        assert result is None


class TestCheapModelMapping:
    """Test cheap model selection."""

    def test_all_providers_have_models(self):
        """All expected providers have cheap models defined."""
        expected_providers = [
            "openai",
            "anthropic",
            "mistralai",
            "groq",
            "together",
            "fireworks",
            "google",
            "cohere",
        ]

        for provider in expected_providers:
            assert provider in CHEAP_MODELS, f"Missing model for {provider}"
            assert CHEAP_MODELS[provider], f"Empty model for {provider}"

    def test_get_cheap_model_returns_correct_model(self):
        """get_cheap_model returns correct model for provider."""
        assert get_cheap_model("openai") == "gpt-4o-mini"
        assert get_cheap_model("anthropic") == "claude-3-5-haiku-latest"
        assert get_cheap_model("groq") == "llama-3.3-70b-versatile"

    def test_get_cheap_model_unknown_returns_none(self):
        """Unknown provider returns None."""
        assert get_cheap_model("unknown") is None


class TestMemoryExtractorInit:
    """Test extractor initialization."""

    def test_auto_detects_provider_and_model(self, mock_openai_client):
        """Extractor auto-detects provider and selects cheap model."""
        # Mock the module path
        mock_openai_client.__class__.__module__ = "openai.resources"

        extractor = MemoryExtractor(mock_openai_client)

        assert extractor.provider == "openai"
        assert extractor.model == "gpt-4o-mini"

    def test_explicit_model_overrides_auto(self, mock_openai_client):
        """Explicit model parameter overrides auto-detection."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        extractor = MemoryExtractor(mock_openai_client, model="gpt-4-turbo")

        assert extractor.model == "gpt-4-turbo"

    def test_unknown_provider_warns(self, mock_openai_client, caplog):
        """Unknown provider logs warning."""
        mock_openai_client.__class__.__module__ = "unknown.provider"

        with caplog.at_level("WARNING"):
            extractor = MemoryExtractor(mock_openai_client)

        assert extractor.model is None
        assert "Could not detect cheap model" in caplog.text


class TestMemoryExtraction:
    """Test memory extraction from conversations."""

    def test_extracts_preference(self, mock_openai_client):
        """Extracts preference from conversation."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        # Mock the LLM response
        mock_openai_client.chat.completions.set_response(
            '{"memories": [{"content": "Prefers Python", "category": "preference", "importance": 0.8}], "should_remember": true}'
        )

        extractor = MemoryExtractor(mock_openai_client)
        memories = extractor.extract(
            "I really prefer Python for data science",
            "Great choice! Python is excellent for data science.",
        )

        assert len(memories) == 1
        assert memories[0].content == "Prefers Python"
        assert memories[0].category == "preference"
        assert memories[0].importance == 0.8

    def test_extracts_multiple_memories(self, mock_openai_client):
        """Extracts multiple memories from one conversation."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response(
            '{"memories": ['
            '{"content": "Works at a startup", "category": "fact", "importance": 0.7},'
            '{"content": "Building an AI agent", "category": "context", "importance": 0.6}'
            '], "should_remember": true}'
        )

        extractor = MemoryExtractor(mock_openai_client)
        memories = extractor.extract(
            "I work at a startup building an AI agent",
            "That sounds exciting!",
        )

        assert len(memories) == 2
        assert memories[0].content == "Works at a startup"
        assert memories[1].content == "Building an AI agent"

    def test_skips_trivial_conversation(self, mock_openai_client):
        """Returns empty for trivial conversations."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response(
            '{"memories": [], "should_remember": false}'
        )

        extractor = MemoryExtractor(mock_openai_client)
        memories = extractor.extract("Hello", "Hi there!")

        assert len(memories) == 0

    def test_handles_json_in_code_block(self, mock_openai_client):
        """Parses JSON wrapped in markdown code block."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response(
            '```json\n{"memories": [{"content": "Likes vim", "category": "preference"}], "should_remember": true}\n```'
        )

        extractor = MemoryExtractor(mock_openai_client)
        memories = extractor.extract("I use vim", "Nice!")

        assert len(memories) == 1
        assert memories[0].content == "Likes vim"

    def test_handles_malformed_json(self, mock_openai_client, caplog):
        """Gracefully handles malformed JSON."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response("not valid json")

        with caplog.at_level("WARNING"):
            extractor = MemoryExtractor(mock_openai_client)
            memories = extractor.extract("test", "test")

        assert len(memories) == 0
        assert "Failed to parse" in caplog.text

    def test_no_extraction_without_model(self, mock_openai_client, caplog):
        """Skips extraction if no model configured."""
        mock_openai_client.__class__.__module__ = "unknown.provider"

        with caplog.at_level("WARNING"):
            extractor = MemoryExtractor(mock_openai_client)
            memories = extractor.extract("test", "test")

        assert len(memories) == 0
        assert "No extraction model" in caplog.text


class TestBatchExtraction:
    """Test batch extraction."""

    def test_batch_extracts_for_multiple_users(self, mock_openai_client):
        """Batch extraction returns memories per user."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response(
            '{"alice": {"memories": [{"content": "Likes Python", "category": "preference"}], "should_remember": true},'
            '"bob": {"memories": [{"content": "Likes Java", "category": "preference"}], "should_remember": true}}'
        )

        extractor = MemoryExtractor(mock_openai_client)
        result = extractor.extract_batch(
            [
                ("alice", "I like Python", "Great!"),
                ("bob", "I like Java", "Nice!"),
            ]
        )

        assert "alice" in result
        assert "bob" in result
        assert result["alice"][0].content == "Likes Python"
        assert result["bob"][0].content == "Likes Java"

    def test_batch_empty_input_returns_empty(self, mock_openai_client):
        """Empty batch returns empty dict."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        extractor = MemoryExtractor(mock_openai_client)
        result = extractor.extract_batch([])

        assert result == {}

    def test_batch_handles_partial_results(self, mock_openai_client):
        """Batch handles some users with no memories."""
        mock_openai_client.__class__.__module__ = "openai.resources"

        mock_openai_client.chat.completions.set_response(
            '{"alice": {"memories": [{"content": "Fact", "category": "fact"}], "should_remember": true},'
            '"bob": {"memories": [], "should_remember": false}}'
        )

        extractor = MemoryExtractor(mock_openai_client)
        result = extractor.extract_batch(
            [
                ("alice", "Important info", "Noted!"),
                ("bob", "Hello", "Hi!"),
            ]
        )

        assert "alice" in result
        assert "bob" not in result  # No memories to remember


class TestAnthropicProvider:
    """Test Anthropic-specific API handling."""

    def test_anthropic_uses_messages_api(self, mock_anthropic_client):
        """Anthropic uses messages.create API."""
        mock_anthropic_client.__class__.__module__ = "anthropic.resources"

        mock_anthropic_client.messages.set_response(
            '{"memories": [{"content": "Test", "category": "fact"}], "should_remember": true}'
        )

        extractor = MemoryExtractor(mock_anthropic_client)
        memories = extractor.extract("test query", "test response")

        assert len(memories) == 1
        assert memories[0].content == "Test"

        # Verify Anthropic API was called
        assert len(mock_anthropic_client.messages.calls) == 1
        call = mock_anthropic_client.messages.calls[0]
        assert call["model"] == "claude-3-5-haiku-latest"
