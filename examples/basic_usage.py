#!/usr/bin/env python3
"""
Basic usage example for Headroom SDK.

This example shows how to wrap an OpenAI client with Headroom
and use both audit and optimize modes.

Run:
    export OPENAI_API_KEY='sk-...'
    python examples/basic_usage.py
"""

import logging
import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

from headroom import HeadroomClient, OpenAIProvider

# Enable logging to see what Headroom is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)

# Load API key from .env.local
load_dotenv(".env.local")

# Create base OpenAI client
base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-..."))

# Create provider for OpenAI models
provider = OpenAIProvider()

# Use temp directory for database
db_path = os.path.join(tempfile.gettempdir(), "headroom_example.db")

# Wrap with Headroom
client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path}",
    default_mode="audit",  # Start with observation
)


def example_audit_mode():
    """Example using audit mode (observe only)."""
    print("=" * 50)
    print("AUDIT MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Current Date: 2024-01-15"},
        {"role": "user", "content": "What's the weather like?"},
    ]

    # In audit mode, request passes through unchanged but metrics are logged
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print()


def example_optimize_mode():
    """Example using optimize mode (apply transforms)."""
    print("=" * 50)
    print("OPTIMIZE MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Current Date: 2024-01-15"},
        {"role": "user", "content": "Search for information."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"results": [' + ",".join([f'{{"id": {i}}}' for i in range(50)]) + "]}",
        },
        {"role": "assistant", "content": "I found 50 results."},
        {"role": "user", "content": "Summarize them."},
    ]

    # In optimize mode, transforms are applied
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        headroom_mode="optimize",
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print()


def example_simulate_mode():
    """Example using simulate mode (preview without API call)."""
    print("=" * 50)
    print("SIMULATE MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Current Date: 2024-01-15"},
        {"role": "user", "content": "Search for information."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"results": [' + ",".join([f'{{"id": {i}}}' for i in range(100)]) + "]}",
        },
        {"role": "assistant", "content": "I found 100 results."},
        {"role": "user", "content": "Summarize them."},
    ]

    # Simulate without calling API
    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=messages,
    )

    print(f"Tokens before: {plan.tokens_before}")
    print(f"Tokens after: {plan.tokens_after}")
    print(f"Tokens saved: {plan.tokens_saved}")
    print(f"Transforms applied: {plan.transforms}")
    print(f"Estimated savings: {plan.estimated_savings}")
    print()


def example_get_metrics():
    """Example of accessing stored metrics."""
    print("=" * 50)
    print("METRICS EXAMPLE")
    print("=" * 50)

    # Get summary statistics from database
    summary = client.get_summary()
    print(f"Total requests: {summary['total_requests']}")
    print(f"Total tokens saved: {summary['total_tokens_saved']}")
    print(f"Average tokens saved: {summary['avg_tokens_saved']:.0f}")
    print()


def example_validate_setup():
    """Example of validating Headroom setup."""
    print("=" * 50)
    print("VALIDATE SETUP EXAMPLE")
    print("=" * 50)

    # Validate that everything is configured correctly
    result = client.validate_setup()

    if result["valid"]:
        print("Setup is valid!")
        print(f"  Provider: {result['provider']['name']}")
        print(f"  Storage: {result['storage']['url']}")
        print(f"  Mode: {result['config']['mode']}")
    else:
        print("Setup issues detected:")
        for key, val in result.items():
            if key != "valid" and not val.get("ok"):
                print(f"  {key}: {val.get('error')}")
    print()


def example_get_stats():
    """Example of getting quick session stats."""
    print("=" * 50)
    print("SESSION STATS EXAMPLE")
    print("=" * 50)

    # Get quick stats without database query
    stats = client.get_stats()

    print("Session stats:")
    print(f"  Requests total: {stats['session']['requests_total']}")
    print(f"  Requests optimized: {stats['session']['requests_optimized']}")
    print(f"  Tokens saved: {stats['session']['tokens_saved_total']}")

    print("\nConfiguration:")
    print(f"  Mode: {stats['config']['mode']}")
    print(f"  Provider: {stats['config']['provider']}")

    print("\nTransforms enabled:")
    print(f"  SmartCrusher: {stats['transforms']['smart_crusher_enabled']}")
    print(f"  RollingWindow: {stats['transforms']['rolling_window_enabled']}")
    print(f"  CacheAligner: {stats['transforms']['cache_aligner_enabled']}")
    print()


if __name__ == "__main__":
    # First, validate the setup
    example_validate_setup()

    # Run all examples
    example_audit_mode()
    example_optimize_mode()
    example_simulate_mode()
    example_get_metrics()

    # Show session stats
    example_get_stats()

    # Clean up
    client.close()
