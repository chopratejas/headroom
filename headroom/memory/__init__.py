"""Headroom Memory - Simple, LLM-driven memory for AI applications.

Two approaches available:

1. Background extraction (original):
    from headroom import with_memory
    client = with_memory(OpenAI(), user_id="alice")

2. Zero-latency inline extraction (recommended):
    from headroom.memory import with_fast_memory
    client = with_fast_memory(OpenAI(), user_id="alice")
"""

from headroom.memory.fast_store import FastMemoryStore, MemoryChunk
from headroom.memory.fast_wrapper import with_fast_memory
from headroom.memory.inline_extractor import (
    InlineMemoryWrapper,
    inject_memory_instruction,
    parse_response_with_memory,
)
from headroom.memory.store import Memory, SQLiteMemoryStore
from headroom.memory.wrapper import with_memory

__all__ = [
    # Original approach (background extraction)
    "with_memory",
    "Memory",
    "SQLiteMemoryStore",
    # Fast approach (inline extraction - recommended)
    "with_fast_memory",
    "FastMemoryStore",
    "MemoryChunk",
    # Low-level inline extraction
    "InlineMemoryWrapper",
    "inject_memory_instruction",
    "parse_response_with_memory",
]
