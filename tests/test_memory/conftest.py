"""Test fixtures for Headroom Memory.

Philosophy: Mock at boundaries, not internals.
- SQLite: REAL (local, fast, no side effects)
- LLM clients: MOCKED (external dependency)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def temp_db():
    """Fresh SQLite DB for each test - REAL database, auto-cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        yield db_path


@pytest.fixture
def memory_store(temp_db):
    """Real SQLite memory store."""
    from headroom.memory.store import SQLiteMemoryStore

    return SQLiteMemoryStore(temp_db)


@pytest.fixture
def mock_extractor():
    """Extractor with controllable responses - for testing worker/wrapper."""
    from headroom.memory.store import Memory

    class MockExtractor:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self.batch_calls: list[list[tuple[str, str, str]]] = []
            self._response: list[Memory] = []
            self._batch_response: dict[str, list[Memory]] = {}

        def set_response(self, memories: list[Memory]) -> None:
            self._response = memories

        def set_batch_response(self, response: dict[str, list[Memory]]) -> None:
            self._batch_response = response

        def extract(self, query: str, response: str) -> list[Memory]:
            self.calls.append((query, response))
            return self._response

        def extract_batch(
            self, conversations: list[tuple[str, str, str]]
        ) -> dict[str, list[Memory]]:
            self.batch_calls.append(conversations)
            return self._batch_response

    return MockExtractor()


@pytest.fixture
def mock_openai_client():
    """Fake OpenAI client - for testing wrapper without API calls."""

    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockChoice:
        def __init__(self, content: str):
            self.message = MockMessage(content)

    class MockResponse:
        def __init__(self, content: str = "Hello!"):
            self.choices = [MockChoice(content)]

    class MockCompletions:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self._response = MockResponse()

        def set_response(self, content: str) -> None:
            self._response = MockResponse(content)

        def create(self, **kwargs: Any) -> MockResponse:
            self.calls.append(kwargs)
            return self._response

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

    class MockClient:
        """Mock OpenAI client."""

        def __init__(self):
            self.chat = MockChat()

    return MockClient()


@pytest.fixture
def mock_anthropic_client():
    """Fake Anthropic client - for testing wrapper without API calls."""

    class MockTextBlock:
        def __init__(self, text: str):
            self.text = text

    class MockResponse:
        def __init__(self, content: str = "Hello!"):
            self.content = [MockTextBlock(content)]

    class MockMessages:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self._response = MockResponse()

        def set_response(self, content: str) -> None:
            self._response = MockResponse(content)

        def create(self, **kwargs: Any) -> MockResponse:
            self.calls.append(kwargs)
            return self._response

    class MockClient:
        """Mock Anthropic client."""

        def __init__(self):
            self.messages = MockMessages()

    return MockClient()


@pytest.fixture
def sample_memories():
    """Sample memories for testing."""
    from headroom.memory.store import Memory

    return [
        Memory(content="User prefers Python", category="preference", importance=0.8),
        Memory(content="User works at a startup", category="fact", importance=0.7),
        Memory(content="User is building an AI agent", category="context", importance=0.6),
    ]
