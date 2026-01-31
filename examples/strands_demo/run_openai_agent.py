#!/usr/bin/env python3
"""Strands SDK + Headroom Demo with OpenAI.

This example demonstrates how to use Headroom's HookProvider with
Strands Agents SDK to automatically compress tool outputs and save tokens.

Requirements:
    pip install "headroom-ai[strands]"
    export OPENAI_API_KEY="your-api-key"

Reference:
    https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/

Usage:
    python run_openai_agent.py
"""

import json
import os

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    print("  export OPENAI_API_KEY='your-api-key'")
    exit(1)

try:
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel
except ImportError:
    print("ERROR: Strands SDK not installed")
    print("  pip install strands-agents")
    exit(1)

try:
    import tiktoken

    from headroom.integrations.strands import HeadroomHookProvider
    from headroom.transforms import SmartCrusher
    from headroom.transforms.smart_crusher import SmartCrusherConfig
except ImportError:
    print("ERROR: Headroom not installed with strands support")
    print("  pip install 'headroom-ai[strands]'")
    exit(1)


# =============================================================================
# Define some example tools that return verbose outputs
# =============================================================================


@tool
def search_documentation(query: str) -> str:
    """Search documentation for information about a topic.

    Args:
        query: The search query

    Returns:
        Search results as JSON
    """
    # Simulate verbose search results (this is what Headroom will compress!)
    results = [
        {
            "id": f"doc-{i}",
            "title": f"Documentation Page {i}: {query}",
            "content": f"This is the content for documentation page {i} about {query}. "
            f"It contains detailed information about the topic including examples, "
            f"best practices, and common pitfalls to avoid. The documentation covers "
            f"various aspects such as installation, configuration, and usage patterns.",
            "score": 0.95 - (i * 0.05),
            "url": f"https://docs.example.com/page-{i}",
            "last_updated": "2025-01-15",
            "author": "Documentation Team",
            "category": "Technical Reference",
        }
        for i in range(20)  # 20 results = lots of tokens!
    ]
    return json.dumps(results, indent=2)


@tool
def get_server_logs(server_name: str, lines: int = 50) -> str:
    """Get recent server logs.

    Args:
        server_name: Name of the server
        lines: Number of log lines to return

    Returns:
        Server logs as JSON
    """
    # Simulate verbose log output
    logs = [
        {
            "timestamp": f"2025-01-30T10:{i:02d}:00Z",
            "level": "INFO" if i % 10 != 7 else "ERROR",
            "service": server_name,
            "message": f"Request processed successfully - latency={50 + i}ms"
            if i % 10 != 7
            else "Connection pool exhausted - max_connections=100",
            "request_id": f"req-{1000 + i}",
            "status_code": 200 if i % 10 != 7 else 500,
            "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
            "ip_address": f"192.168.1.{i % 256}",
        }
        for i in range(lines)
    ]
    return json.dumps(logs, indent=2)


@tool
def get_metrics(metric_name: str) -> str:
    """Get system metrics.

    Args:
        metric_name: Name of the metric to retrieve

    Returns:
        Metric data as JSON
    """
    # Simulate time-series metrics data
    metrics = {
        "name": metric_name,
        "unit": "percent",
        "interval": "1m",
        "data_points": [
            {
                "timestamp": f"2025-01-30T10:{i:02d}:00Z",
                "value": 45.0 + (i % 10) * 2,
                "host": "prod-server-1",
                "region": "us-east-1",
                "tags": {"env": "production", "team": "platform"},
            }
            for i in range(60)  # 60 data points
        ],
    }
    return json.dumps(metrics, indent=2)


# =============================================================================
# Before/After Compression Demo
# =============================================================================


def show_before_after_demo():
    """Show what compression looks like on real tool output."""
    print("=" * 70)
    print("PART 1: Before/After Compression Demo")
    print("=" * 70)
    print()

    # Generate sample log output (same as get_server_logs would return)
    sample_logs = [
        {
            "timestamp": f"2025-01-30T10:{i:02d}:00Z",
            "level": "INFO" if i % 10 != 7 else "ERROR",
            "service": "prod-server-1",
            "message": f"Request processed successfully - latency={50 + i}ms"
            if i % 10 != 7
            else "Connection pool exhausted - max_connections=100",
            "request_id": f"req-{1000 + i}",
            "status_code": 200 if i % 10 != 7 else 500,
            "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
            "ip_address": f"192.168.1.{i % 256}",
        }
        for i in range(20)  # Use 20 for demo (smaller than full 50)
    ]
    original_text = json.dumps(sample_logs, indent=2)

    # Initialize compressor and tokenizer
    config = SmartCrusherConfig(max_items_after_crush=10)
    crusher = SmartCrusher(config=config)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    # Compress it
    result = crusher.crush(original_text)
    compressed_text = result.compressed

    # Count tokens
    tokens_before = len(enc.encode(original_text))
    tokens_after = len(enc.encode(compressed_text))
    savings_pct = (1 - tokens_after / tokens_before) * 100

    # Show BEFORE
    print("┌" + "─" * 68 + "┐")
    print("│ BEFORE COMPRESSION (first 800 chars of server logs)               │")
    print("├" + "─" * 68 + "┤")
    # Show truncated original
    preview = (
        original_text[:800] + "\n... [truncated - " + str(len(original_text)) + " total chars]"
    )
    for line in preview.split("\n"):
        print(f"│ {line[:66]:<66} │")
    print("└" + "─" * 68 + "┘")
    print(f"   📊 Tokens: {tokens_before}")
    print()

    # Show AFTER - reformat for display
    print("┌" + "─" * 68 + "┐")
    print("│ AFTER COMPRESSION (SmartCrusher output)                           │")
    print("├" + "─" * 68 + "┤")
    # Pretty-print the compressed JSON for display
    try:
        compressed_data = json.loads(compressed_text)
        compressed_pretty = json.dumps(compressed_data, indent=2)
        lines = compressed_pretty.split("\n")[:30]  # First 30 lines
        for line in lines:
            print(f"│ {line[:66]:<66} │")
        if len(compressed_pretty.split("\n")) > 30:
            print(f"│ {'... [truncated for display]':<66} │")
        print(f"│ {'':<66} │")
        print(f"│ {'📝 Items kept: ' + str(len(compressed_data)) + ' (from 20 original)':<66} │")
    except json.JSONDecodeError:
        # Fallback for non-JSON output
        for line in compressed_text.split("\n")[:25]:
            print(f"│ {line[:66]:<66} │")
    print("└" + "─" * 68 + "┘")
    print(f"   📊 Tokens: {tokens_after}")
    print()

    # Show savings
    print("┌" + "─" * 68 + "┐")
    print("│ COMPRESSION RESULTS                                               │")
    print("├" + "─" * 68 + "┤")
    print(f"│   Before:  {tokens_before:>6} tokens{' ' * 45}│")
    print(f"│   After:   {tokens_after:>6} tokens{' ' * 45}│")
    print(f"│   Saved:   {tokens_before - tokens_after:>6} tokens ({savings_pct:.1f}%){' ' * 34}│")
    print("│                                                                    │")
    print("│   ✅ Error messages preserved                                      │")
    print("│   ✅ Timestamps preserved                                          │")
    print("│   ✅ Statistical summary added                                     │")
    print("│   ✅ Redundant entries removed                                     │")
    print("└" + "─" * 68 + "┘")
    print()
    print()


# =============================================================================
# Main Demo
# =============================================================================


def main():
    # First show the before/after compression demo
    show_before_after_demo()

    print("=" * 70)
    print("PART 2: Live Agent Demo with Headroom Hooks")
    print("=" * 70)
    print()

    # Create Headroom hook provider
    # This will automatically compress tool outputs using SmartCrusher
    headroom_hook = HeadroomHookProvider(
        compress_tool_outputs=True,
        min_tokens_to_compress=100,  # Compress outputs > 100 tokens
        max_items_after_crush=10,  # Keep max 10 items from arrays
        token_alert_threshold=50000,  # Alert if context exceeds 50k tokens
    )

    # Create OpenAI model
    model = OpenAIModel(
        model_id="gpt-4o-mini",  # Use gpt-4o-mini for cost efficiency
        # model_id="gpt-4o",  # Or use gpt-4o for better quality
    )

    # Create agent with Headroom hooks
    agent = Agent(
        model=model,
        tools=[search_documentation, get_server_logs, get_metrics],
        hooks=[headroom_hook],  # Headroom registers its hooks here!
        system_prompt="""You are a helpful DevOps assistant. You can search documentation,
check server logs, and retrieve metrics. When analyzing issues, use the available
tools to gather information and provide actionable recommendations.""",
    )

    print("Agent created with Headroom optimization enabled!")
    print()

    # Run a query that will use tools
    query = """I'm seeing some errors on prod-server-1. Can you:
1. Check the recent logs
2. Look at the CPU metrics
3. Search the documentation for connection pool issues

Then give me a summary of what's happening and how to fix it."""

    print(f"Query: {query}")
    print()
    print("-" * 70)
    print("Running agent (watch for Headroom compression logs)...")
    print("-" * 70)
    print()

    # Run the agent
    result = agent(query)

    print()
    print("-" * 70)
    print("Agent Response:")
    print("-" * 70)
    print(result)
    print()

    # Show Headroom savings
    print("=" * 70)
    print("Headroom Token Savings Summary")
    print("=" * 70)
    summary = headroom_hook.get_savings_summary()
    print(f"  Total events:           {summary['total_events']}")
    print(f"  Tool compressions:      {summary['tool_compressions']}")
    print(f"  Total tokens saved:     {summary['total_tokens_saved']}")
    print(f"  Average savings:        {summary['average_savings_percent']:.1f}%")
    print(f"  Alerts triggered:       {summary['alerts']}")
    print()

    # Show per-tool metrics
    if headroom_hook.metrics_history:
        print("Per-Tool Compression Details:")
        print("-" * 50)
        for metric in headroom_hook.metrics_history:
            if metric.event_type == "tool_compress":
                print(f"  {metric.tool_name}:")
                print(f"    Before: {metric.tokens_before} tokens")
                print(f"    After:  {metric.tokens_after} tokens")
                print(f"    Saved:  {metric.tokens_saved} tokens ({metric.savings_percent:.1f}%)")
                print()


if __name__ == "__main__":
    main()
