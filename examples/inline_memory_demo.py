#!/usr/bin/env python3
"""Demo: Zero-Latency Inline Memory Extraction (Letta-style).

This demonstrates the Letta/MemGPT approach where the LLM outputs
memories as part of its response - ZERO extra latency!

Comparison:
- OLD: Main LLM call (500ms) + Extraction LLM call (500ms) = 1000ms total
- NEW: Main LLM call with inline extraction (500ms) = 500ms total

The memory is extracted from the SAME tokens the LLM is already generating.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/inline_memory_demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from headroom.memory.inline_extractor import (
    InlineMemoryWrapper,
)


def demo_inline_extraction():
    """Demonstrate inline memory extraction."""
    print("=" * 60)
    print("ZERO-LATENCY INLINE MEMORY EXTRACTION")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    wrapper = InlineMemoryWrapper(client)

    # Test conversations with memory-worthy content
    test_conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "I'm a Python developer working on a fintech startup. We use PostgreSQL for our database.",
                },
            ],
            "description": "User shares background info",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you today?"},
            ],
            "description": "Simple greeting (should have no memories)",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "I prefer async/await over callbacks, and I always use type hints in my code.",
                },
            ],
            "description": "User shares preferences",
        },
    ]

    total_latency = 0
    total_memories = 0

    for i, test in enumerate(test_conversations, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {test['description']}")
        print(f"{'─' * 60}")

        user_msg = test["messages"][-1]["content"]
        print(f"User: {user_msg[:80]}...")

        start = time.perf_counter()
        response, memories = wrapper.chat(
            messages=test["messages"],
            model="gpt-4o-mini",
        )
        elapsed = time.perf_counter() - start

        total_latency += elapsed
        total_memories += len(memories)

        print(f"\nAssistant: {response[:150]}...")
        print(f"\nLatency: {elapsed * 1000:.0f}ms")
        print(f"Memories extracted: {len(memories)}")

        if memories:
            for mem in memories:
                print(f"  - [{mem.get('category', 'unknown')}] {mem.get('content', '')}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total conversations: {len(test_conversations)}")
    print(f"Total memories extracted: {total_memories}")
    print(f"Average latency: {total_latency / len(test_conversations) * 1000:.0f}ms")
    print("\n✓ ZERO extra latency - memories extracted from same response!")


def benchmark_vs_separate_extraction():
    """Compare inline vs separate LLM extraction."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Inline vs Separate Extraction")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    wrapper = InlineMemoryWrapper(client)

    test_message = "I'm a senior backend engineer at Netflix. I prefer Go for microservices but Python for ML. I always use Docker and Kubernetes."

    # Measure inline extraction
    print("\n1. INLINE EXTRACTION (Letta-style)")
    print("   Single LLM call with memory instruction")

    inline_latencies = []
    for i in range(3):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_message},
        ]

        start = time.perf_counter()
        response, memories = wrapper.chat(messages, model="gpt-4o-mini")
        elapsed = time.perf_counter() - start

        inline_latencies.append(elapsed * 1000)
        print(f"   Run {i + 1}: {elapsed * 1000:.0f}ms ({len(memories)} memories)")

    # Measure separate extraction (simulated)
    print("\n2. SEPARATE EXTRACTION (Traditional)")
    print("   Main LLM call + Extraction LLM call")

    separate_latencies = []
    for i in range(3):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_message},
        ]

        start = time.perf_counter()

        # First call: Main response
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        main_response = response1.choices[0].message.content

        # Second call: Extract memories
        extraction_prompt = f"""Extract memories from this conversation:
User: {test_message}
Assistant: {main_response}

Return JSON: {{"memories": [{{"content": "...", "category": "preference|fact|context"}}]}}"""

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
        )

        elapsed = time.perf_counter() - start
        separate_latencies.append(elapsed * 1000)
        print(f"   Run {i + 1}: {elapsed * 1000:.0f}ms")

    # Summary
    inline_avg = sum(inline_latencies) / len(inline_latencies)
    separate_avg = sum(separate_latencies) / len(separate_latencies)

    print(f"\n{'─' * 60}")
    print(f"{'Approach':<30} {'Avg Latency':<15} {'Savings':<15}")
    print(f"{'─' * 60}")
    print(f"{'Inline (Letta-style)':<30} {inline_avg:>10.0f}ms {'baseline':>15}")
    print(
        f"{'Separate extraction':<30} {separate_avg:>10.0f}ms {f'+{separate_avg - inline_avg:.0f}ms':>15}"
    )
    print(f"{'─' * 60}")

    savings = separate_avg - inline_avg
    print(
        f"\n✓ Inline extraction saves {savings:.0f}ms ({savings / separate_avg * 100:.0f}% faster)"
    )
    print("✓ This is the latency of an ENTIRE extra LLM call - now FREE!")


if __name__ == "__main__":
    demo_inline_extraction()
    benchmark_vs_separate_extraction()
