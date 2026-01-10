#!/usr/bin/env python3
"""
LangChain + Headroom Integration: Before vs After Examples

This script demonstrates the real-world impact of Headroom optimization
on LangChain applications. Run with:

    python examples/langchain_before_after.py

Requirements:
    pip install headroom[langchain] langchain-openai

Note: Set OPENAI_API_KEY environment variable for live API tests.
For dry-run mode (no API calls), the script shows simulated results.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass

# Check dependencies
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed. Install with: pip install langchain-core")

try:
    from langchain_openai import ChatOpenAI  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("langchain-openai not installed. Install with: pip install langchain-openai")

# Import Headroom
try:
    from headroom import (  # noqa: F401
        HeadroomClient,
        HeadroomConfig,
        HeadroomMode,
        OpenAIProvider,
    )

    HEADROOM_AVAILABLE = True
except ImportError:
    HEADROOM_AVAILABLE = False
    print("Headroom not installed. Install with: pip install headroom")


@dataclass
class ComparisonResult:
    """Result of before/after comparison."""

    scenario: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    latency_before_ms: float | None
    latency_after_ms: float | None
    cost_before_usd: float
    cost_after_usd: float
    cost_saved_usd: float


def estimate_cost(tokens: int, model: str = "gpt-4o") -> float:
    """Estimate cost in USD. GPT-4o: $2.50/1M input tokens."""
    rates = {
        "gpt-4o": 2.50 / 1_000_000,
        "gpt-4o-mini": 0.15 / 1_000_000,
        "claude-3-5-sonnet": 3.00 / 1_000_000,
    }
    return tokens * rates.get(model, 2.50 / 1_000_000)


def print_comparison(result: ComparisonResult) -> None:
    """Print formatted comparison results."""
    print(f"\n{'=' * 60}")
    print(f"Scenario: {result.scenario}")
    print(f"{'=' * 60}")
    print("\n[Token Comparison]")
    print(f"   Before: {result.tokens_before:,} tokens")
    print(f"   After:  {result.tokens_after:,} tokens")
    print(f"   Saved:  {result.tokens_saved:,} tokens ({result.savings_percent:.1f}%)")

    print("\n[Cost Impact] (GPT-4o pricing)")
    print(f"   Before: ${result.cost_before_usd:.4f}")
    print(f"   After:  ${result.cost_after_usd:.4f}")
    print(f"   Saved:  ${result.cost_saved_usd:.4f}")

    if result.latency_before_ms and result.latency_after_ms:
        print("\n[Latency]")
        print(f"   Before: {result.latency_before_ms:.0f}ms")
        print(f"   After:  {result.latency_after_ms:.0f}ms")


def langchain_to_openai_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to OpenAI format."""
    openai_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_dict = {"role": "assistant", "content": msg.content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            openai_messages.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
    return openai_messages


# ============================================================================
# SCENARIO 1: Agentic Workflow with Large Tool Outputs
# ============================================================================


def scenario_agentic_workflow() -> ComparisonResult:
    """
    Scenario: AI agent that searches a database and processes results.

    Common pattern: Tool returns 100+ records, but only 5-10 are relevant.
    Without optimization, ALL records are sent to the LLM.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1: Agentic Workflow with Large Tool Outputs")
    print("=" * 60)

    # Simulate a database search tool that returns many results
    search_results = [
        {
            "id": f"user-{i:04d}",
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "department": ["Engineering", "Sales", "Marketing", "Support"][i % 4],
            "status": "active" if i % 10 != 0 else "inactive",
            "created_at": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00Z",
            "last_login": f"2024-12-{(i % 28) + 1:02d}T{i % 24:02d}:00:00Z",
            "metadata": {
                "preferences": {"theme": "dark", "notifications": True},
                "tags": ["premium", "verified"] if i % 5 == 0 else [],
            },
        }
        for i in range(100)
    ]

    # The conversation in LangChain format
    lc_messages = [
        SystemMessage(
            content="""You are a helpful database assistant.
        When searching for users, analyze the results and provide a summary.
        Focus on active users in the Engineering department."""
        ),
        HumanMessage(content="Find users in the Engineering department"),
        AIMessage(
            content="I'll search the database for Engineering users.",
            tool_calls=[
                {"id": "call_1", "name": "search_users", "args": {"department": "Engineering"}}
            ],
        ),
        ToolMessage(
            content=json.dumps(search_results),  # 100 records!
            tool_call_id="call_1",
        ),
    ]

    # Convert to OpenAI format for Headroom
    messages = langchain_to_openai_messages(lc_messages)

    # Create Headroom client for simulation
    from openai import OpenAI

    db_path = os.path.join(tempfile.gettempdir(), "headroom_langchain_example.db")
    base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake-key"))
    provider = OpenAIProvider()

    client = HeadroomClient(
        original_client=base_client,
        provider=provider,
        store_url=f"sqlite:///{db_path}",
        default_mode="optimize",
    )

    # Simulate optimization
    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=messages,
    )

    tokens_before = plan.tokens_before
    tokens_after = plan.tokens_after
    tokens_saved = plan.tokens_saved
    savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

    print("\n[Before Optimization]")
    print("   - System prompt + conversation")
    print(f"   - Tool output: 100 user records ({len(json.dumps(search_results))} chars)")

    print("\n[After Optimization]")
    print("   - SmartCrusher kept: first 3, last 2, + relevance matches")
    print("   - Estimated ~15 items preserved (Engineering dept matches)")
    print(f"   - Transforms: {plan.transforms}")

    client.close()

    return ComparisonResult(
        scenario="Agentic Workflow with Large Tool Outputs",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        savings_percent=savings_percent,
        latency_before_ms=None,
        latency_after_ms=None,
        cost_before_usd=estimate_cost(tokens_before),
        cost_after_usd=estimate_cost(tokens_after),
        cost_saved_usd=estimate_cost(tokens_saved),
    )


# ============================================================================
# SCENARIO 2: Long Conversation with Context Window Pressure
# ============================================================================


def scenario_long_conversation() -> ComparisonResult:
    """
    Scenario: Multi-turn conversation approaching context window limit.

    Common pattern: Chatbot accumulates history, needs to drop old turns.
    Without optimization, either hits context limit or loses coherence.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 2: Long Conversation with Context Window Pressure")
    print("=" * 60)

    # Simulate 50-turn conversation in LangChain format
    lc_messages = [
        SystemMessage(
            content="""You are a customer support agent for TechCorp.
        You have access to customer data and can help with:
        - Account issues
        - Billing questions
        - Technical support
        - Product information

        Current date: 2024-12-15
        Agent ID: support-agent-42
        """
        ),
    ]

    # Add 50 turns of conversation
    topics = [
        "I can't log into my account",
        "What's my current subscription?",
        "Can you explain the premium features?",
        "I was charged twice this month",
        "How do I reset my password?",
    ]

    for i in range(50):
        topic = topics[i % len(topics)]
        lc_messages.append(HumanMessage(content=f"Turn {i}: {topic}"))
        lc_messages.append(
            AIMessage(
                content=f"Response to turn {i}: Thank you for reaching out about '{topic}'. "
                f"I can help you with that. Here's what I found... " * 3
            )
        )

    # Convert to OpenAI format
    messages = langchain_to_openai_messages(lc_messages)

    # Create Headroom client for simulation
    from openai import OpenAI

    db_path = os.path.join(tempfile.gettempdir(), "headroom_langchain_example.db")
    base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake-key"))
    provider = OpenAIProvider()

    client = HeadroomClient(
        original_client=base_client,
        provider=provider,
        store_url=f"sqlite:///{db_path}",
        default_mode="optimize",
    )

    # Simulate optimization
    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=messages,
    )

    tokens_before = plan.tokens_before
    tokens_after = plan.tokens_after
    tokens_saved = plan.tokens_saved
    savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

    print("\n[Before Optimization]")
    print("   - 50-turn conversation")
    print(f"   - ~{tokens_before:,} tokens total")

    print("\n[After Optimization]")
    print("   - RollingWindow kept system + last N turns")
    print("   - CacheAligner moved date to dynamic tail")
    print(f"   - Transforms: {plan.transforms}")

    client.close()

    return ComparisonResult(
        scenario="Long Conversation (50 turns)",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        savings_percent=savings_percent,
        latency_before_ms=None,
        latency_after_ms=None,
        cost_before_usd=estimate_cost(tokens_before),
        cost_after_usd=estimate_cost(tokens_after),
        cost_saved_usd=estimate_cost(tokens_saved),
    )


# ============================================================================
# SCENARIO 3: RAG with Retrieved Documents
# ============================================================================


def scenario_rag_pipeline() -> ComparisonResult:
    """
    Scenario: RAG pipeline that retrieves multiple documents.

    Common pattern: Retriever returns 10 chunks, many are redundant.
    Without optimization, all chunks consume tokens.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 3: RAG Pipeline with Retrieved Documents")
    print("=" * 60)

    # Simulate retrieved document chunks
    chunks = []
    for i in range(10):
        chunk = {
            "content": f"Document {i} content: " + "This is relevant information. " * 50,
            "source": f"doc_{i}.pdf",
            "page": i + 1,
            "relevance_score": 0.9 - (i * 0.05),
            "metadata": {
                "author": f"Author {i}",
                "date": "2024-01-15",
                "category": "Technical",
            },
        }
        chunks.append(chunk)

    context = "\n\n".join(
        [f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}" for c in chunks]
    )

    # LangChain format
    lc_messages = [
        SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
        HumanMessage(
            content=f"""Based on the following retrieved documents:

{context}

Question: What are the key technical requirements?"""
        ),
    ]

    # Convert to OpenAI format
    messages = langchain_to_openai_messages(lc_messages)

    # Create Headroom client for simulation
    from openai import OpenAI

    db_path = os.path.join(tempfile.gettempdir(), "headroom_langchain_example.db")
    base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-fake-key"))
    provider = OpenAIProvider()

    client = HeadroomClient(
        original_client=base_client,
        provider=provider,
        store_url=f"sqlite:///{db_path}",
        default_mode="optimize",
    )

    # Simulate optimization
    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=messages,
    )

    tokens_before = plan.tokens_before
    tokens_after = plan.tokens_after
    tokens_saved = plan.tokens_saved
    savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

    print("\n[Before Optimization]")
    print("   - 10 retrieved document chunks")
    print(f"   - ~{tokens_before:,} tokens total")

    print("\n[After Optimization]")
    print("   - CacheAligner normalized whitespace")
    print(f"   - Transforms: {plan.transforms}")

    client.close()

    return ComparisonResult(
        scenario="RAG Pipeline (10 chunks)",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        savings_percent=savings_percent,
        latency_before_ms=None,
        latency_after_ms=None,
        cost_before_usd=estimate_cost(tokens_before),
        cost_after_usd=estimate_cost(tokens_after),
        cost_saved_usd=estimate_cost(tokens_saved),
    )


# ============================================================================
# SCENARIO 4: Real API Comparison (if API key available)
# ============================================================================


def scenario_live_api() -> ComparisonResult | None:
    """
    Scenario: Live API comparison with actual timing.

    Only runs if OPENAI_API_KEY is set.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[!] Skipping live API test (OPENAI_API_KEY not set)")
        return None

    if not OPENAI_AVAILABLE:
        print("\n[!] Skipping live API test (langchain-openai not installed)")
        return None

    print("\n" + "=" * 60)
    print("SCENARIO 4: Live API Comparison")
    print("=" * 60)

    from openai import OpenAI

    # Create base OpenAI client
    base_client = OpenAI()

    # Create Headroom-wrapped client
    db_path = os.path.join(tempfile.gettempdir(), "headroom_langchain_live.db")
    provider = OpenAIProvider()

    headroom_client = HeadroomClient(
        original_client=base_client,
        provider=provider,
        store_url=f"sqlite:///{db_path}",
        default_mode="optimize",
    )

    # Test messages in OpenAI format
    messages = [
        {"role": "system", "content": "You are helpful. Be concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Time the base client
    start = time.time()
    base_response = base_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=50,
    )
    latency_before = (time.time() - start) * 1000

    # Time the Headroom-wrapped client
    start = time.time()
    optimized_response = headroom_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        headroom_mode="optimize",
        max_tokens=50,
    )
    latency_after = (time.time() - start) * 1000

    print(f"\n[Base Model Response] {base_response.choices[0].message.content[:50]}...")
    print(f"[Optimized Response] {optimized_response.choices[0].message.content[:50]}...")
    print(f"\n[Latency] {latency_before:.0f}ms -> {latency_after:.0f}ms")

    # Get metrics
    headroom_client.get_summary()

    headroom_client.close()

    # For this simple case, savings are minimal
    tokens_before = base_response.usage.prompt_tokens if base_response.usage else 20
    tokens_after = optimized_response.usage.prompt_tokens if optimized_response.usage else 20
    tokens_saved = max(0, tokens_before - tokens_after)

    return ComparisonResult(
        scenario="Live API (Simple Query)",
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        savings_percent=(tokens_saved / tokens_before * 100) if tokens_before > 0 else 0,
        latency_before_ms=latency_before,
        latency_after_ms=latency_after,
        cost_before_usd=estimate_cost(tokens_before, "gpt-4o-mini"),
        cost_after_usd=estimate_cost(tokens_after, "gpt-4o-mini"),
        cost_saved_usd=estimate_cost(tokens_saved, "gpt-4o-mini"),
    )


# ============================================================================
# MAIN: Run All Scenarios
# ============================================================================


def main():
    """Run all comparison scenarios."""
    print("\n" + "=" * 70)
    print("   HEADROOM + LANGCHAIN: Before vs After Comparison")
    print("=" * 70)

    if not LANGCHAIN_AVAILABLE:
        print("\n[X] Cannot run examples: LangChain not installed")
        print("   Install with: pip install langchain-core")
        return

    if not HEADROOM_AVAILABLE:
        print("\n[X] Cannot run examples: Headroom not installed")
        return

    results = []

    # Run each scenario
    try:
        results.append(scenario_agentic_workflow())
    except Exception as e:
        print(f"\n[X] Scenario 1 failed: {e}")

    try:
        results.append(scenario_long_conversation())
    except Exception as e:
        print(f"\n[X] Scenario 2 failed: {e}")

    try:
        results.append(scenario_rag_pipeline())
    except Exception as e:
        print(f"\n[X] Scenario 3 failed: {e}")

    try:
        live_result = scenario_live_api()
        if live_result:
            results.append(live_result)
    except Exception as e:
        print(f"\n[X] Live API scenario failed: {e}")

    # Print all results
    print("\n\n" + "=" * 70)
    print("   SUMMARY: All Scenarios")
    print("=" * 70)

    for result in results:
        print_comparison(result)

    # Calculate totals
    if results:
        total_saved = sum(r.tokens_saved for r in results)
        total_cost_saved = sum(r.cost_saved_usd for r in results)
        avg_savings = sum(r.savings_percent for r in results) / len(results)

        print("\n" + "=" * 70)
        print("   TOTAL IMPACT")
        print("=" * 70)
        print(f"\n[Results] Across {len(results)} scenarios:")
        print(f"   Total tokens saved: {total_saved:,}")
        print(f"   Average savings: {avg_savings:.1f}%")
        print(f"   Total cost saved: ${total_cost_saved:.4f}")
        print("\n[Projection] At scale (1M requests/month):")
        print(f"   Estimated monthly savings: ${total_cost_saved * 1_000_000 / len(results):,.2f}")


if __name__ == "__main__":
    main()
