#!/usr/bin/env python3
"""
Multi-Tool Agent Test: Diverse Data Types with Claude API

This test creates an agent with multiple tools returning different data types:
- GitHub: Issues, PRs, repo metadata
- ArXiv: Paper abstracts and citations
- Code Search: Source code snippets
- Database: JSON records

We run it WITHOUT Headroom and WITH Headroom to compare token usage.
Uses Claude API for real function calling.
"""

import json
import os
import time
from dataclasses import dataclass

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable required")

# =============================================================================
# MOCK TOOL DATA - Realistic responses from various sources
# =============================================================================

GITHUB_ISSUES = [
    {
        "number": i,
        "title": f"Issue #{i}: {'Memory leak in worker pool' if i == 42 else 'Feature request: ' + ['dark mode', 'API pagination', 'webhook support', 'rate limiting'][i % 4]}",
        "state": "open" if i % 3 != 0 else "closed",
        "author": f"user{i % 20}",
        "labels": ["bug", "priority:high"] if i == 42 else ["enhancement"],
        "created_at": f"2024-12-{(i % 28) + 1:02d}T10:00:00Z",
        "updated_at": f"2024-12-{(i % 28) + 1:02d}T15:00:00Z",
        "comments": i % 10,
        "body": "Worker threads are not being released after task completion, causing memory to grow unboundedly. Stack trace attached."
        if i == 42
        else f"Please add support for {['dark mode', 'API pagination', 'webhook support', 'rate limiting'][i % 4]}. This would greatly improve the user experience.",
        "assignees": ["maintainer1"] if i == 42 else [],
        "milestone": "v2.0" if i < 20 else None,
        "reactions": {"thumbs_up": 47 if i == 42 else i % 5, "thumbs_down": 0},
    }
    for i in range(50)
]

ARXIV_PAPERS = [
    {
        "id": f"2401.{i:05d}",
        "title": f"{'Attention Is All You Need: Revisited' if i == 15 else ['Deep Learning for Code Generation', 'Efficient Transformers', 'Neural Architecture Search', 'Language Model Scaling'][i % 4]}",
        "authors": [f"Author{j}" for j in range(3 + i % 3)],
        "abstract": "We revisit the transformer architecture and propose key optimizations that reduce memory usage by 40% while maintaining accuracy. Our method introduces sparse attention patterns..."
        if i == 15
        else f"This paper presents a novel approach to {['code generation', 'transformer efficiency', 'neural architecture', 'model scaling'][i % 4]}. We demonstrate state-of-the-art results on benchmark datasets.",
        "categories": ["cs.LG", "cs.CL"] if i == 15 else ["cs.LG"],
        "published": f"2024-01-{(i % 28) + 1:02d}",
        "citations": 1247 if i == 15 else i * 3,
        "pdf_url": f"https://arxiv.org/pdf/2401.{i:05d}.pdf",
        "comment": "Accepted at NeurIPS 2024" if i == 15 else None,
    }
    for i in range(30)
]

CODE_SEARCH_RESULTS = [
    {
        "file": f"src/{'worker.py' if i == 23 else ['utils.py', 'api.py', 'models.py', 'handlers.py'][i % 4]}",
        "line": 100 + i * 10,
        "content": '''def cleanup_worker(self):
    """Release worker resources - MEMORY LEAK FIX"""
    self.thread_pool.shutdown(wait=True)
    self.connections.clear()
    gc.collect()  # Force garbage collection'''
        if i == 23
        else f'''def process_{["data", "request", "model", "event"][i % 4]}(self, input):
    """Process incoming {["data", "request", "model", "event"][i % 4]}"""
    result = self.transform(input)
    return self.validate(result)''',
        "language": "python",
        "repository": "main-app",
        "relevance_score": 0.98 if i == 23 else 0.7 - (i * 0.01),
        "context_before": ["    # Worker management", "    "],
        "context_after": ["", "    def start_worker(self):"],
    }
    for i in range(40)
]

DATABASE_RECORDS = [
    {
        "id": f"rec_{i:06d}",
        "type": "error" if i == 17 else "info",
        "timestamp": f"2024-12-15T{(i % 24):02d}:{(i % 60):02d}:00Z",
        "service": "worker-pool" if i == 17 else ["api", "auth", "db", "cache"][i % 4],
        "message": "OutOfMemoryError: heap space exhausted in WorkerPool.execute()"
        if i == 17
        else f"Operation completed: {['request processed', 'user authenticated', 'query executed', 'cache updated'][i % 4]}",
        "metadata": {
            "heap_used": "7.8GB" if i == 17 else f"{1 + i % 3}GB",
            "heap_max": "8GB",
            "thread_count": 847 if i == 17 else 50 + i % 50,
        },
        "stack_trace": "java.lang.OutOfMemoryError: Java heap space\n\tat WorkerPool.execute(WorkerPool.java:234)\n\tat TaskRunner.run(TaskRunner.java:89)"
        if i == 17
        else None,
    }
    for i in range(60)
]


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@tool(name="search_github_issues")
def search_github_issues(query: str, repo: str = "main-app") -> str:
    """Search GitHub issues in a repository.

    Args:
        query: Search query for issues
        repo: Repository name

    Returns:
        JSON array of matching issues
    """
    return json.dumps(GITHUB_ISSUES, indent=2)


@tool(name="search_arxiv_papers")
def search_arxiv_papers(query: str, max_results: int = 30) -> str:
    """Search ArXiv for academic papers.

    Args:
        query: Search query for papers
        max_results: Maximum number of results

    Returns:
        JSON array of matching papers
    """
    return json.dumps(ARXIV_PAPERS, indent=2)


@tool(name="search_code")
def search_code(query: str, language: str = "python") -> str:
    """Search codebase for matching code.

    Args:
        query: Code search query
        language: Programming language filter

    Returns:
        JSON array of code search results
    """
    return json.dumps(CODE_SEARCH_RESULTS, indent=2)


@tool(name="query_database")
def query_database(query: str, table: str = "logs") -> str:
    """Query the database for records.

    Args:
        query: SQL-like query
        table: Table to query

    Returns:
        JSON array of database records
    """
    return json.dumps(DATABASE_RECORDS, indent=2)


# =============================================================================
# TEST RUNNER
# =============================================================================


@dataclass
class TestResult:
    label: str
    input_tokens: int
    output_tokens: int
    response: str
    duration_ms: float
    tool_calls: int


def count_tokens_approx(text: str) -> int:
    """Approximate token count (Ollama doesn't always report tokens)."""
    return len(text) // 4


def run_agent_test(use_headroom: bool) -> TestResult:
    """Run the multi-tool agent test."""

    label = "WITH Headroom" if use_headroom else "WITHOUT Headroom (Baseline)"

    if use_headroom:
        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Claude(id="claude-sonnet-4-20250514")
        model = HeadroomAgnoModel(wrapped_model=base_model)
    else:
        model = Claude(id="claude-sonnet-4-20250514")

    agent = Agent(
        model=model,
        tools=[search_github_issues, search_arxiv_papers, search_code, query_database],
        markdown=True,
    )

    # The question that requires searching multiple sources
    question = """I'm investigating a memory leak in our application. Please:
1. Search GitHub issues for memory-related bugs
2. Search our codebase for memory leak fixes
3. Check the database logs for OutOfMemory errors
4. Find any relevant research papers about memory management in worker pools

Summarize what you find and recommend a fix."""

    print(f"\n{'=' * 70}")
    print(f"Running: {label}")
    print(f"{'=' * 70}")
    print(f"Question: {question[:100]}...")

    start_time = time.time()

    try:
        response = agent.run(question)
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        response_text = f"Error: {e}"

    duration_ms = (time.time() - start_time) * 1000

    # Get token counts
    if use_headroom and hasattr(model, "total_tokens_saved"):
        summary = model.get_savings_summary()
        input_tokens = summary.get("total_tokens_after", 0)  # Actual tokens sent to API
        tokens_before = summary.get("total_tokens_before", 0)
        tokens_saved = model.total_tokens_saved
        savings_pct = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0
        print("\n📊 Headroom Optimization Stats:")
        print(f"   API requests made: {summary.get('total_requests', 0)}")
        print(f"   Tokens BEFORE optimization: {tokens_before:,}")
        print(f"   Tokens AFTER optimization: {input_tokens:,}")
        print(f"   Tokens SAVED: {tokens_saved:,} ({savings_pct:.1f}%)")
    else:
        # Estimate from data size
        total_data = (
            json.dumps(GITHUB_ISSUES)
            + json.dumps(ARXIV_PAPERS)
            + json.dumps(CODE_SEARCH_RESULTS)
            + json.dumps(DATABASE_RECORDS)
        )
        input_tokens = count_tokens_approx(total_data + question)

    print(f"\nResponse preview: {response_text[:500]}...")
    print(f"Duration: {duration_ms:.0f}ms")

    return TestResult(
        label=label,
        input_tokens=input_tokens,
        output_tokens=count_tokens_approx(response_text),
        response=response_text,
        duration_ms=duration_ms,
        tool_calls=4,  # We expect 4 tool calls
    )


def main():
    print("\n" + "=" * 70)
    print("MULTI-TOOL AGENT TEST")
    print("Testing diverse data types: GitHub, ArXiv, Code, Database")
    print("Model: Claude Sonnet (claude-sonnet-4-20250514)")
    print("=" * 70)

    # Show data sizes
    print("\nTool output sizes:")
    print(
        f"  GitHub Issues:  {len(json.dumps(GITHUB_ISSUES)):,} chars ({len(GITHUB_ISSUES)} items)"
    )
    print(f"  ArXiv Papers:   {len(json.dumps(ARXIV_PAPERS)):,} chars ({len(ARXIV_PAPERS)} items)")
    print(
        f"  Code Search:    {len(json.dumps(CODE_SEARCH_RESULTS)):,} chars ({len(CODE_SEARCH_RESULTS)} items)"
    )
    print(
        f"  Database Logs:  {len(json.dumps(DATABASE_RECORDS)):,} chars ({len(DATABASE_RECORDS)} items)"
    )
    total_chars = sum(
        len(json.dumps(d))
        for d in [GITHUB_ISSUES, ARXIV_PAPERS, CODE_SEARCH_RESULTS, DATABASE_RECORDS]
    )
    print(f"  TOTAL:          {total_chars:,} chars (~{total_chars // 4:,} tokens)")

    # Run baseline (no Headroom)
    print("\n" + "-" * 70)
    baseline = run_agent_test(use_headroom=False)

    # Run with Headroom
    print("\n" + "-" * 70)
    optimized = run_agent_test(use_headroom=True)

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"""
                              Baseline        Headroom
    ─────────────────────────────────────────────────────
    Tokens Sent to API:       {baseline.input_tokens:>6,}          {optimized.input_tokens:>6,}
    Duration:                 {baseline.duration_ms:>6,.0f}ms        {optimized.duration_ms:>6,.0f}ms
    Tool Calls:               {baseline.tool_calls:>6}            {optimized.tool_calls:>6}
    """)

    if baseline.input_tokens > optimized.input_tokens:
        saved = baseline.input_tokens - optimized.input_tokens
        percent = (saved / baseline.input_tokens) * 100
        print(f"    ✨ Tokens Saved: {saved:,} ({percent:.1f}% reduction)")
        print(f"    💰 Estimated Cost Savings: {percent:.0f}% on input tokens")

    print("\n" + "=" * 70)
    print("BASELINE RESPONSE (excerpt):")
    print("=" * 70)
    print(baseline.response[:1500] if len(baseline.response) > 1500 else baseline.response)

    print("\n" + "=" * 70)
    print("HEADROOM RESPONSE (excerpt):")
    print("=" * 70)
    print(optimized.response[:1500] if len(optimized.response) > 1500 else optimized.response)


if __name__ == "__main__":
    main()
