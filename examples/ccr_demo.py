#!/usr/bin/env python3
"""Demonstration of CCR (Compress-Cache-Retrieve) architecture.

This script demonstrates:
1. How compression works with CCR caching
2. How the Response Handler automatically handles retrieval tool calls
3. How the Context Tracker enables multi-turn awareness

Run with: python examples/ccr_demo.py
"""

import asyncio
import json

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr import (
    CCR_TOOL_NAME,
    CCRResponseHandler,
    CCRToolCall,
    ContextTracker,
    ContextTrackerConfig,
    ResponseHandlerConfig,
    create_ccr_tool_definition,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_compression_store() -> str:
    """Demonstrate the compression store."""
    print_section("1. COMPRESSION STORE - Caching Original Content")

    # Reset for clean demo
    reset_compression_store()
    store = get_compression_store()

    # Simulate tool output with 100 items
    original_items = [
        {"id": i, "file": f"src/module_{i}.py", "lines": 100 + i, "status": "ok"}
        for i in range(100)
    ]
    # Add some errors for interest
    original_items[42]["status"] = "error"
    original_items[42]["error"] = "SyntaxError: unexpected indent"
    original_items[77]["status"] = "warning"
    original_items[77]["warning"] = "Unused import"

    original_json = json.dumps(original_items)

    # SmartCrusher would compress to top 15 items (keeping errors)
    compressed_items = [
        original_items[0],  # First few for context
        original_items[1],
        original_items[2],
        original_items[42],  # Error item - always kept!
        original_items[77],  # Warning item - always kept!
        original_items[97],  # Last few for recency
        original_items[98],
        original_items[99],
    ]
    compressed_json = json.dumps(compressed_items)

    # Store in CCR cache
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=100,
        compressed_item_count=8,
        tool_name="list_files",
    )

    print(f"\nOriginal: {len(original_items)} items ({len(original_json):,} chars)")
    print(f"Compressed: {len(compressed_items)} items ({len(compressed_json):,} chars)")
    print(f"Reduction: {100 - (len(compressed_json) / len(original_json) * 100):.1f}%")
    print(f"CCR Hash: {hash_key}")

    # Show that we can retrieve
    entry = store.retrieve(hash_key)
    print(f"\nRetrieved original: {entry.original_item_count} items")

    # Show search capability
    results = store.search(hash_key, "error SyntaxError")
    print(f"Search for 'error SyntaxError': found {len(results)} items")
    if results:
        print(f"  Found: {results[0]}")

    return hash_key


def demo_tool_injection(hash_key: str) -> dict:
    """Demonstrate tool injection."""
    print_section("2. TOOL INJECTION - Adding Retrieval Capability")

    # Show the tool definition that gets injected
    tool_def = create_ccr_tool_definition("anthropic")
    print(f"\nInjected tool: {tool_def['name']}")
    print(f"Description: {tool_def['description'][:100]}...")

    # Show the marker that gets added to compressed content
    marker = f"\n[100 items compressed to 8. Retrieve more: hash={hash_key}]"
    print(f"\nMarker added to output:{marker}")

    # Simulate an LLM response that calls the retrieval tool
    simulated_response = {
        "content": [
            {"type": "text", "text": "I see some files. Let me get the full list."},
            {
                "type": "tool_use",
                "id": "toolu_01ABC",
                "name": CCR_TOOL_NAME,
                "input": {"hash": hash_key},
            },
        ]
    }

    print("\nSimulated LLM response (calls headroom_retrieve):")
    print(json.dumps(simulated_response, indent=2)[:500] + "...")

    return simulated_response


async def demo_response_handler(hash_key: str, initial_response: dict) -> None:
    """Demonstrate the response handler."""
    print_section("3. RESPONSE HANDLER - Automatic Tool Call Handling")

    print("\n--- BEFORE (without Response Handler) ---")
    print("Problem: LLM calls headroom_retrieve, but no one handles it!")
    print("The tool call would go back to the client unhandled.")
    print("Client would need custom code to handle CCR tool calls.")

    print("\n--- AFTER (with Response Handler) ---")
    print("Solution: Response Handler intercepts and handles automatically!")

    handler = CCRResponseHandler(
        ResponseHandlerConfig(
            max_retrieval_rounds=3,
        )
    )

    # Check if response has CCR tool calls
    has_ccr = handler.has_ccr_tool_calls(initial_response, "anthropic")
    print(f"\nDetected CCR tool call: {has_ccr}")

    # Parse the tool call
    call = CCRToolCall(
        tool_call_id="toolu_01ABC",
        hash_key=hash_key,
    )
    print(f"Parsed: hash={call.hash_key}, query={call.query}")

    # Execute retrieval
    result = handler._execute_retrieval(call)
    print(f"\nRetrieved {result.items_retrieved} items")
    print(f"Success: {result.success}")

    # Show what would happen in full flow
    print("\nFull flow simulation:")
    print("1. LLM response contains tool_use(headroom_retrieve)")
    print("2. Handler detects CCR tool call")
    print("3. Handler retrieves from cache (instant, ~1ms)")
    print("4. Handler adds tool result to messages")
    print("5. Handler makes continuation API call")
    print("6. LLM responds with actual answer (no more CCR calls)")
    print("7. Handler returns final response to client")

    # Show handler stats
    stats = handler.get_stats()
    print(f"\nHandler stats: {stats}")


def demo_context_tracker(hash_key: str) -> None:
    """Demonstrate the context tracker."""
    print_section("4. CONTEXT TRACKER - Multi-Turn Awareness")

    print("\n--- BEFORE (without Context Tracker) ---")
    print("Problem: In turn 5, LLM forgets what was compressed in turn 1!")
    print("User: 'What about the authentication middleware?'")
    print("LLM: 'I don't see any authentication files.'")
    print("(Because auth files were in the compressed 92 items, not shown)")

    print("\n--- AFTER (with Context Tracker) ---")
    print("Solution: Tracker proactively expands relevant compressed content!")

    config = ContextTrackerConfig(
        relevance_threshold=0.1,  # Lower for demo
        max_context_age_seconds=300,
    )
    tracker = ContextTracker(config)

    # Track the compression from turn 1
    # Use keywords in sample_content that will match the query
    tracker.track_compression(
        hash_key=hash_key,
        turn_number=1,
        tool_name="list_files",
        original_count=100,
        compressed_count=8,
        query_context="list all python files",
        sample_content="authentication middleware handler auth_middleware.py auth_handler.py login security",
    )
    print(f"\nTurn 1: Tracked compression {hash_key}")
    print("        Sample: 'authentication middleware handler auth_middleware.py ...'")

    # Turn 5: User asks about auth
    query = "show authentication middleware"
    print(f"\nTurn 5: User asks '{query}'")

    recommendations = tracker.analyze_query(query, current_turn=5)
    print(f"        Tracker found {len(recommendations)} relevant contexts")

    if recommendations:
        rec = recommendations[0]
        print(f"        → hash={rec.hash_key}")
        print(f"        → relevance={rec.relevance_score:.2f}")
        print(f"        → reason: {rec.reason}")
        print(
            f"        → action: {'full expansion' if rec.expand_full else f'search for {rec.search_query}'}"
        )

        # Execute expansion
        results = tracker.execute_expansions(recommendations)
        if results:
            print(f"\nProactively expanded: {results[0]['item_count']} items")
            print("LLM now sees full file list, including auth_middleware.py!")

    # Show tracker stats
    stats = tracker.get_stats()
    print(f"\nTracker stats: {json.dumps(stats, indent=2)}")


def demo_full_flow() -> None:
    """Show the complete CCR flow."""
    print_section("5. COMPLETE CCR FLOW")

    print("""
    ┌────────────────────────────────────────────────────────────┐
    │  COMPLETE CCR ARCHITECTURE                                  │
    │                                                             │
    │  Phase 1: COMPRESSION STORE                                │
    │  └─ Cache original content with hash                       │
    │  └─ Enable instant retrieval (~1ms)                        │
    │                                                             │
    │  Phase 2: TOOL INJECTION                                   │
    │  └─ Add headroom_retrieve tool to LLM context              │
    │  └─ Add retrieval markers to compressed output             │
    │                                                             │
    │  Phase 3: RESPONSE HANDLER                                 │
    │  └─ Intercept LLM responses                                │
    │  └─ Detect CCR tool calls                                  │
    │  └─ Execute retrievals automatically                       │
    │  └─ Continue conversation until done                       │
    │                                                             │
    │  Phase 4: CONTEXT TRACKER                                  │
    │  └─ Track compressed content across turns                  │
    │  └─ Analyze new queries for relevance                      │
    │  └─ Proactively expand when needed                         │
    │                                                             │
    │  Phase 5: FEEDBACK LOOP                                    │
    │  └─ Learn from retrieval patterns                          │
    │  └─ Adjust compression for future requests                 │
    └────────────────────────────────────────────────────────────┘
    """)

    print("KEY BENEFITS:")
    print("• Reversible compression - no permanent data loss")
    print("• Automatic handling - no client code changes needed")
    print("• Multi-turn awareness - prevents context amnesia")
    print("• Feedback learning - improves over time")
    print("• Zero-risk - fallback to full data always available")


async def main() -> None:
    """Run the CCR demonstration."""
    print("\n" + "=" * 60)
    print("  HEADROOM CCR (Compress-Cache-Retrieve) DEMONSTRATION")
    print("=" * 60)

    # Demo 1: Compression Store
    hash_key = demo_compression_store()

    # Demo 2: Tool Injection
    initial_response = demo_tool_injection(hash_key)

    # Demo 3: Response Handler
    await demo_response_handler(hash_key, initial_response)

    # Demo 4: Context Tracker
    demo_context_tracker(hash_key)

    # Demo 5: Full Flow
    demo_full_flow()

    print("\n" + "=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nRun the proxy with CCR enabled:")
    print("  headroom proxy --port 8787")
    print("\nCCR is enabled by default. The proxy will:")
    print("• Cache compressed content automatically")
    print("• Inject retrieval tool when compression occurs")
    print("• Handle CCR tool calls in LLM responses")
    print("• Track context across conversation turns")
    print()


if __name__ == "__main__":
    asyncio.run(main())
