"""Tests for HeadroomClient cache optimizer integration."""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from headroom import (
    HeadroomClient,
    AnthropicCacheOptimizer,
    CacheOptimizerRegistry,
)
from headroom.providers import AnthropicProvider, OpenAIProvider


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield f"sqlite:///{path}"
    if os.path.exists(path):
        os.unlink(path)


class MockTokenCounter:
    """Mock token counter for testing."""

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_messages(self, messages: list) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(block.get("text", "")) // 4
        return total


class MockAnthropicProvider:
    """Mock Anthropic provider for testing."""

    name = "anthropic"

    def get_token_counter(self, model: str):
        return MockTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 200000


class MockOpenAIProvider:
    """Mock OpenAI provider for testing."""

    name = "openai"

    def get_token_counter(self, model: str):
        return MockTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 128000


class TestHeadroomClientCacheIntegration:
    """Test HeadroomClient cache optimizer integration."""

    def test_auto_detect_anthropic_optimizer(self, temp_db):
        """Test that Anthropic optimizer is auto-detected."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
        )

        assert client._cache_optimizer is not None
        assert client._cache_optimizer.name == "anthropic-cache-optimizer"

    def test_auto_detect_openai_optimizer(self, temp_db):
        """Test that OpenAI optimizer is auto-detected."""
        mock_client = MagicMock()
        provider = MockOpenAIProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
        )

        assert client._cache_optimizer is not None
        assert client._cache_optimizer.name == "openai-prefix-stabilizer"

    def test_custom_optimizer(self, temp_db):
        """Test using a custom optimizer."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()
        custom_optimizer = AnthropicCacheOptimizer()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            cache_optimizer=custom_optimizer,
        )

        assert client._cache_optimizer is custom_optimizer

    def test_disable_cache_optimizer(self, temp_db):
        """Test disabling cache optimizer."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=False,
        )

        assert client._cache_optimizer is None

    def test_semantic_cache_layer_creation(self, temp_db):
        """Test semantic cache layer is created when enabled."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
            enable_semantic_cache=True,
        )

        assert client._semantic_cache_layer is not None
        assert client._cache_optimizer is not None

    def test_extract_query_from_string_content(self, temp_db):
        """Test query extraction from string content."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        query = client._extract_query(messages)
        assert query == "What is 2+2?"

    def test_extract_query_from_content_blocks(self, temp_db):
        """Test query extraction from content block format."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            },
        ]

        query = client._extract_query(messages)
        assert query == "What is 2+2?"

    def test_extract_query_last_user_message(self, temp_db):
        """Test that query extraction uses last user message."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        query = client._extract_query(messages)
        assert query == "Second question"

    def test_config_propagation(self, temp_db):
        """Test that config is properly propagated."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
            enable_semantic_cache=True,
        )

        assert client._config.cache_optimizer.enabled is True
        assert client._config.cache_optimizer.enable_semantic_cache is True
