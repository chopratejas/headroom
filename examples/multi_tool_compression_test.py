#!/usr/bin/env python3
"""
Multi-Tool Compression Test: Diverse Data Types

This test shows how Headroom compresses different types of tool outputs:
- GitHub: Issues, PRs, repo metadata
- ArXiv: Paper abstracts and citations
- Code Search: Source code snippets
- Database: JSON records

We compare WITHOUT Headroom (raw data) vs WITH Headroom (compressed).
"""

import json

from headroom.config import SmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher

# =============================================================================
# MOCK TOOL DATA - Realistic responses from various sources
# =============================================================================

# Critical items are at specific positions to test needle preservation
GITHUB_ISSUES = [
    {
        "number": i,
        "title": f"Issue #{i}: {'CRITICAL: Memory leak in worker pool causing OOM' if i == 42 else 'Feature request: ' + ['dark mode', 'API pagination', 'webhook support', 'rate limiting'][i % 4]}",
        "state": "open" if i % 3 != 0 else "closed",
        "author": f"user{i % 20}",
        "labels": ["bug", "priority:critical", "memory-leak"] if i == 42 else ["enhancement"],
        "created_at": f"2024-12-{(i % 28) + 1:02d}T10:00:00Z",
        "updated_at": f"2024-12-{(i % 28) + 1:02d}T15:00:00Z",
        "comments": 47 if i == 42 else i % 10,
        "body": "Worker threads are not being released after task completion, causing memory to grow unboundedly. Stack trace attached. FIX: Call thread_pool.shutdown() in cleanup_worker()."
        if i == 42
        else f"Please add support for {['dark mode', 'API pagination', 'webhook support', 'rate limiting'][i % 4]}.",
        "assignees": ["maintainer1", "memory-team"] if i == 42 else [],
    }
    for i in range(50)
]

ARXIV_PAPERS = [
    {
        "id": f"2401.{i:05d}",
        "title": "Memory-Efficient Worker Pool Management: A Practical Guide"
        if i == 15
        else ["Deep Learning for Code", "Efficient Transformers", "Neural Search", "LLM Scaling"][
            i % 4
        ],
        "authors": [f"Author{j}" for j in range(3 + i % 3)],
        "abstract": "We present techniques for managing memory in worker pools, including automatic cleanup, connection pooling limits, and garbage collection strategies. Key finding: setting max_connections=500 and implementing periodic cleanup reduces memory by 73%."
        if i == 15
        else f"This paper presents approaches to {['code generation', 'transformer efficiency', 'neural search', 'model scaling'][i % 4]}.",
        "categories": ["cs.SE", "cs.DC"] if i == 15 else ["cs.LG"],
        "citations": 1247 if i == 15 else i * 3,
    }
    for i in range(30)
]

CODE_SEARCH_RESULTS = [
    {
        "file": f"src/{'worker.py' if i == 23 else ['utils.py', 'api.py', 'models.py'][i % 3]}",
        "line": 100 + i * 10,
        "content": """def cleanup_worker(self):
    '''Release worker resources - FIXES MEMORY LEAK'''
    self.thread_pool.shutdown(wait=True)
    self.connections.clear()
    gc.collect()  # Force garbage collection
    logger.info("Worker cleaned up, memory released")"""
        if i == 23
        else f"""def process_{["data", "request", "model"][i % 3]}(self, input):
    result = self.transform(input)
    return self.validate(result)""",
        "language": "python",
        "match_score": 0.99 if i == 23 else 0.5 - (i * 0.01),
    }
    for i in range(40)
]

DATABASE_RECORDS = [
    {
        "id": f"rec_{i:06d}",
        "level": "ERROR" if i == 17 else "INFO",
        "timestamp": f"2024-12-15T{(i % 24):02d}:{(i % 60):02d}:00Z",
        "service": "worker-pool" if i == 17 else ["api", "auth", "db", "cache"][i % 4],
        "message": "OutOfMemoryError: Java heap space exhausted in WorkerPool.execute() - SOLUTION: increase max_connections to 500"
        if i == 17
        else f"Operation completed: {['request processed', 'authenticated', 'query done', 'cache hit'][i % 4]}",
        "stack_trace": "java.lang.OutOfMemoryError\n\tat WorkerPool.execute(WorkerPool.java:234)"
        if i == 17
        else None,
    }
    for i in range(60)
]


def compress_and_show(name: str, data: list, query: str, needle_check: callable) -> dict:
    """Compress data and show before/after with needle verification."""
    config = SmartCrusherConfig()
    crusher = SmartCrusher(config)

    original_json = json.dumps(data, indent=2)
    result = crusher.crush(original_json, query=query)
    compressed_data = json.loads(result.compressed)

    # Check if needle was preserved
    needle_found = needle_check(compressed_data)

    reduction = (1 - len(result.compressed) / len(original_json)) * 100

    return {
        "name": name,
        "items_before": len(data),
        "items_after": len(compressed_data),
        "chars_before": len(original_json),
        "chars_after": len(result.compressed),
        "reduction_percent": reduction,
        "needle_preserved": needle_found,
        "compressed_data": compressed_data,
    }


def main():
    print("\n" + "=" * 70)
    print("MULTI-TOOL COMPRESSION TEST")
    print("Testing Headroom on diverse data types")
    print("=" * 70)

    query = "memory leak worker pool OutOfMemory fix"

    results = []

    # Test each data source
    print("\n" + "-" * 70)
    print("1. GITHUB ISSUES")
    print("-" * 70)
    gh_result = compress_and_show(
        "GitHub Issues",
        GITHUB_ISSUES,
        query,
        lambda data: any("memory leak" in str(item).lower() for item in data),
    )
    results.append(gh_result)
    print(f"   Items: {gh_result['items_before']} → {gh_result['items_after']}")
    print(f"   Chars: {gh_result['chars_before']:,} → {gh_result['chars_after']:,}")
    print(f"   Reduction: {gh_result['reduction_percent']:.1f}%")
    print(f"   Critical issue #42 preserved: {gh_result['needle_preserved']}")

    print("\n" + "-" * 70)
    print("2. ARXIV PAPERS")
    print("-" * 70)
    arxiv_result = compress_and_show(
        "ArXiv Papers",
        ARXIV_PAPERS,
        query,
        lambda data: any("worker pool" in str(item).lower() for item in data),
    )
    results.append(arxiv_result)
    print(f"   Items: {arxiv_result['items_before']} → {arxiv_result['items_after']}")
    print(f"   Chars: {arxiv_result['chars_before']:,} → {arxiv_result['chars_after']:,}")
    print(f"   Reduction: {arxiv_result['reduction_percent']:.1f}%")
    print(f"   Memory paper #15 preserved: {arxiv_result['needle_preserved']}")

    print("\n" + "-" * 70)
    print("3. CODE SEARCH")
    print("-" * 70)
    code_result = compress_and_show(
        "Code Search",
        CODE_SEARCH_RESULTS,
        query,
        lambda data: any("cleanup_worker" in str(item) for item in data),
    )
    results.append(code_result)
    print(f"   Items: {code_result['items_before']} → {code_result['items_after']}")
    print(f"   Chars: {code_result['chars_before']:,} → {code_result['chars_after']:,}")
    print(f"   Reduction: {code_result['reduction_percent']:.1f}%")
    print(f"   Fix code #23 preserved: {code_result['needle_preserved']}")

    print("\n" + "-" * 70)
    print("4. DATABASE LOGS")
    print("-" * 70)
    db_result = compress_and_show(
        "Database Logs",
        DATABASE_RECORDS,
        query,
        lambda data: any("OutOfMemoryError" in str(item) for item in data),
    )
    results.append(db_result)
    print(f"   Items: {db_result['items_before']} → {db_result['items_after']}")
    print(f"   Chars: {db_result['chars_before']:,} → {db_result['chars_after']:,}")
    print(f"   Reduction: {db_result['reduction_percent']:.1f}%")
    print(f"   Error log #17 preserved: {db_result['needle_preserved']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_before = sum(r["chars_before"] for r in results)
    total_after = sum(r["chars_after"] for r in results)
    total_reduction = (1 - total_after / total_before) * 100
    all_needles = all(r["needle_preserved"] for r in results)

    print("""
    Data Source      Before      After     Reduction   Needle OK
    ─────────────────────────────────────────────────────────────""")
    for r in results:
        print(
            f"    {r['name']:<16} {r['chars_before']:>6,}  →  {r['chars_after']:>5,}     {r['reduction_percent']:>5.1f}%      {'Yes' if r['needle_preserved'] else 'NO!'}"
        )
    print("    ─────────────────────────────────────────────────────────────")
    print(
        f"    TOTAL            {total_before:>6,}  →  {total_after:>5,}     {total_reduction:>5.1f}%      {'All' if all_needles else 'FAIL'}"
    )

    print(f"""
    TOKENS (estimated):
      Before: ~{total_before // 4:,} tokens
      After:  ~{total_after // 4:,} tokens
      Saved:  ~{(total_before - total_after) // 4:,} tokens ({total_reduction:.1f}%)

    CRITICAL INFO PRESERVED: {all_needles}
      - GitHub Issue #42 (memory leak bug): {"Found" if results[0]["needle_preserved"] else "MISSING"}
      - ArXiv Paper #15 (worker pool memory): {"Found" if results[1]["needle_preserved"] else "MISSING"}
      - Code file #23 (cleanup_worker fix): {"Found" if results[2]["needle_preserved"] else "MISSING"}
      - DB Log #17 (OutOfMemoryError): {"Found" if results[3]["needle_preserved"] else "MISSING"}
    """)

    # Show what was kept for one example
    print("=" * 70)
    print("EXAMPLE: What Headroom kept from GitHub Issues")
    print("=" * 70)
    for i, item in enumerate(gh_result["compressed_data"][:5]):
        title = item.get("title", "")[:60]
        labels = item.get("labels", [])
        print(f"  {i + 1}. #{item.get('number')}: {title}...")
        if labels:
            print(f"     Labels: {labels}")
    if len(gh_result["compressed_data"]) > 5:
        print(f"  ... and {len(gh_result['compressed_data']) - 5} more items")


if __name__ == "__main__":
    main()
