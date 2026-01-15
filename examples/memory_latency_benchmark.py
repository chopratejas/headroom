#!/usr/bin/env python3
"""Benchmark: LLM Extraction vs Embedding-Only Memory.

Demonstrates the massive latency difference between:
1. OLD: LLM-based extraction (2-3 seconds)
2. NEW: Embedding-only storage (sub-100ms)

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/memory_latency_benchmark.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from headroom.memory.extractor import MemoryExtractor
from headroom.memory.fast_store import (
    FastMemoryStore,
    create_local_embed_fn,
    create_openai_batch_embed_fn,
    create_openai_embed_fn,
)

# Test messages with memory-worthy content
TEST_MESSAGES = [
    ("I prefer Python over JavaScript for backend development", "Great choice!"),
    ("I work at a fintech startup handling payment processing", "Interesting domain!"),
    ("Always use PostgreSQL for relational data, never MongoDB", "Solid preference!"),
    ("I'm migrating from monolith to microservices architecture", "Good luck!"),
    ("My email is test@example.com, contact me there only", "Noted!"),
]


def benchmark_llm_extraction(client: OpenAI, num_runs: int = 5) -> list[float]:
    """Benchmark the OLD LLM-based extraction approach."""
    print("\n" + "=" * 60)
    print("BENCHMARK: LLM-Based Extraction (OLD)")
    print("=" * 60)

    extractor = MemoryExtractor(client)
    latencies = []

    for i, (query, response) in enumerate(TEST_MESSAGES[:num_runs]):
        start = time.perf_counter()
        memories = extractor.extract(query, response)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)  # Convert to ms
        print(f"  Run {i + 1}: {elapsed * 1000:.0f}ms - extracted {len(memories)} memories")

    return latencies


def benchmark_embedding_store(client: OpenAI, num_runs: int = 5) -> list[float]:
    """Benchmark embedding-only approach with INDIVIDUAL API calls."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Embedding-Only, Individual Calls")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fast_memory.db"
        embed_fn = create_openai_embed_fn(client)
        store = FastMemoryStore(db_path, embed_fn=embed_fn)

        latencies = []

        for i, (query, response) in enumerate(TEST_MESSAGES[:num_runs]):
            start = time.perf_counter()
            # Store both messages (2 separate API calls)
            store.add("test_user", query, role="user")
            store.add("test_user", response, role="assistant")
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)  # Convert to ms
            print(f"  Run {i + 1}: {elapsed * 1000:.0f}ms - stored 2 chunks (2 API calls)")

    return latencies


def benchmark_batched_embedding(client: OpenAI, num_runs: int = 5) -> list[float]:
    """Benchmark embedding-only approach with BATCHED API calls."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Embedding-Only, BATCHED Calls")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fast_memory.db"
        embed_fn = create_openai_embed_fn(client)
        batch_embed_fn = create_openai_batch_embed_fn(client)
        store = FastMemoryStore(db_path, embed_fn=embed_fn)

        latencies = []

        for i, (query, response) in enumerate(TEST_MESSAGES[:num_runs]):
            start = time.perf_counter()
            # Store both messages in ONE API call
            store.add_turn_batched("test_user", query, response, batch_embed_fn)
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)  # Convert to ms
            print(f"  Run {i + 1}: {elapsed * 1000:.0f}ms - stored 2 chunks (1 API call)")

    return latencies


def benchmark_local_embedding(num_runs: int = 5) -> list[float]:
    """Benchmark embedding-only approach with LOCAL model (FASTEST)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: LOCAL Embeddings (FASTEST - No API!)")
    print("=" * 60)

    # Load model once (this is slow, but only happens once)
    print("  Loading local model (one-time cost)...")
    start_load = time.perf_counter()
    embed_fn = create_local_embed_fn("all-MiniLM-L6-v2")
    load_time = time.perf_counter() - start_load
    print(f"  Model loaded in {load_time:.1f}s")

    # Warmup runs to trigger JIT compilation
    print("  Warming up (JIT compilation)...")
    for _ in range(3):
        embed_fn("warmup text for compilation")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fast_memory.db"
        store = FastMemoryStore(db_path, embed_fn=embed_fn, embedding_dim=384)

        latencies = []

        for i, (query, response) in enumerate(TEST_MESSAGES[:num_runs]):
            start = time.perf_counter()
            # Store both messages
            store.add("test_user", query, role="user")
            store.add("test_user", response, role="assistant")
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)  # Convert to ms
            print(f"  Run {i + 1}: {elapsed * 1000:.1f}ms - stored 2 chunks (LOCAL)")

    return latencies


def benchmark_search_comparison(client: OpenAI) -> None:
    """Compare search latency: FTS5 vs Vector Similarity."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Search Latency")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "fast_memory.db"
        embed_fn = create_openai_embed_fn(client)
        store = FastMemoryStore(db_path, embed_fn=embed_fn)

        # Populate with test data
        print("  Populating store with 20 memories...")
        for query, response in TEST_MESSAGES * 4:
            store.add("test_user", query, role="user")
            store.add("test_user", response, role="assistant")

        # Benchmark searches
        search_queries = [
            "What programming language?",
            "database recommendations",
            "architecture patterns",
            "contact information",
        ]

        print("\n  Search latencies:")
        for query in search_queries:
            start = time.perf_counter()
            results = store.search("test_user", query, top_k=3)
            elapsed = time.perf_counter() - start

            top_match = results[0][0].text[:40] if results else "None"
            print(f"    '{query}' -> {elapsed * 1000:.0f}ms ({len(results)} results)")
            print(f"      Top match: '{top_match}...'")


def main():
    """Run all benchmarks."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("MEMORY LATENCY BENCHMARK")
    print("Comparing LLM Extraction vs Embedding-Only")
    print("=" * 60)

    # Run benchmarks
    llm_latencies = benchmark_llm_extraction(client, num_runs=3)
    embed_latencies = benchmark_embedding_store(client, num_runs=3)
    batched_latencies = benchmark_batched_embedding(client, num_runs=5)
    local_latencies = benchmark_local_embedding(num_runs=5)
    benchmark_search_comparison(client)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    llm_avg = mean(llm_latencies)
    embed_avg = mean(embed_latencies)
    batched_avg = mean(batched_latencies)
    local_avg = mean(local_latencies)

    print(f"\n{'Approach':<35} {'Avg Latency':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'LLM Extraction (OLD)':<35} {llm_avg:>10.0f}ms {'1.0x':>10}")
    print(f"{'Embedding (2 API calls)':<35} {embed_avg:>10.0f}ms {llm_avg / embed_avg:>9.1f}x")
    print(
        f"{'Embedding BATCHED (1 API call)':<35} {batched_avg:>10.0f}ms {llm_avg / batched_avg:>9.1f}x"
    )
    print(f"{'LOCAL Embeddings (no API!)':<35} {local_avg:>10.1f}ms {llm_avg / local_avg:>9.0f}x")

    print(f"\n{'=' * 60}")
    print(f"BEST SPEEDUP: {llm_avg / local_avg:.0f}x FASTER with local embeddings!")
    print(f"{'=' * 60}")

    if local_avg < 100:
        print("\n✓ SUB-100ms ACHIEVED with local embeddings!")
    if local_avg < 50:
        print("✓ SUB-50ms ACHIEVED!")
    if local_avg < 20:
        print("✓ SUB-20ms ACHIEVED - GOAL MET!")


if __name__ == "__main__":
    main()
