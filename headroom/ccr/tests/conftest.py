"""Shared fixtures for MCP server tests."""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# --- Mock MCP types so tests work even if mcp package isn't installed ---

@dataclass
class MockTextContent:
    type: str
    text: str


@dataclass
class MockTool:
    name: str
    description: str
    inputSchema: dict


@pytest.fixture
def large_json_content():
    """Large JSON content that triggers compression."""
    items = [
        {
            "id": i,
            "name": f"server-{i:03d}",
            "status": "error" if i == 42 else "running",
            "cpu": i * 3 % 100,
            "memory": i * 512,
            "region": ["us-east-1", "eu-west-1", "ap-southeast-1"][i % 3],
        }
        for i in range(80)
    ]
    return json.dumps(items)


@pytest.fixture
def small_content():
    """Small content that likely passes through without compression."""
    return "Hello, world!"


@pytest.fixture
def mock_compress_result():
    """Mock CompressResult for patching headroom.compress.compress."""
    result = MagicMock()
    result.messages = [{"role": "tool", "content": "[compressed content]"}]
    result.tokens_before = 5000
    result.tokens_after = 1000
    result.compression_ratio = 0.2
    result.transforms_applied = ["smart_crusher"]
    return result


@pytest.fixture
def mock_store():
    """Mock CompressionStore."""
    store = MagicMock()
    store.store.return_value = "abc123hash"
    store.get_stats.return_value = {"entry_count": 5, "max_entries": 500}

    # Mock retrieve
    entry = MagicMock()
    entry.original_content = "full original content here"
    entry.original_item_count = 80
    entry.compressed_item_count = 10
    entry.retrieval_count = 1
    store.retrieve.return_value = entry

    # Mock search
    store.search.return_value = [{"text": "matched item", "score": 0.9}]

    return store
