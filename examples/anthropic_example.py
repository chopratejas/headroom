#!/usr/bin/env python3
"""
Anthropic example for Headroom SDK.

This example shows how to use Headroom with Anthropic Claude models.
"""

import os
import tempfile

from anthropic import Anthropic
from dotenv import load_dotenv

from headroom import AnthropicProvider, HeadroomClient

# Load API key from .env.local
load_dotenv(".env.local")

# Create base Anthropic client
base_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Create provider for Anthropic models
provider = AnthropicProvider()

# Use temp directory for database
db_path = os.path.join(tempfile.gettempdir(), "headroom_anthropic.db")

# Wrap with Headroom
client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path}",
    default_mode="audit",
)


def example_audit_mode():
    """Example using audit mode (observe only)."""
    print("=" * 50)
    print("ANTHROPIC AUDIT MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "What's 2 + 2? Reply in one word."},
    ]

    # In audit mode, request passes through unchanged but metrics are logged
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        messages=messages,
        max_tokens=100,
    )

    print(f"Response: {response.content[0].text}")
    print()


def example_optimize_mode():
    """Example using optimize mode (apply transforms)."""
    print("=" * 50)
    print("ANTHROPIC OPTIMIZE MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "Search for information."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "search",
                    "input": {"query": "test"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"results": ['
                    + ",".join([f'{{"id": {i}}}' for i in range(50)])
                    + "]}",
                }
            ],
        },
        {"role": "assistant", "content": "I found 50 results."},
        {"role": "user", "content": "Summarize them briefly."},
    ]

    # In optimize mode, transforms are applied
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        messages=messages,
        headroom_mode="optimize",
        max_tokens=100,
    )

    print(f"Response: {response.content[0].text}")
    print()


def example_simulate_mode():
    """Example using simulate mode (preview without API call)."""
    print("=" * 50)
    print("ANTHROPIC SIMULATE MODE EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "Search for information."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "search",
                    "input": {"query": "test"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"results": ['
                    + ",".join([f'{{"id": {i}}}' for i in range(100)])
                    + "]}",
                }
            ],
        },
        {"role": "assistant", "content": "I found 100 results."},
        {"role": "user", "content": "Summarize them."},
    ]

    # Simulate without calling API
    plan = client.messages.simulate(
        model="claude-3-5-sonnet-latest",
        messages=messages,
    )

    print(f"Tokens before: {plan.tokens_before}")
    print(f"Tokens after: {plan.tokens_after}")
    print(f"Tokens saved: {plan.tokens_saved}")
    print(f"Transforms applied: {plan.transforms}")
    print(f"Estimated savings: {plan.estimated_savings}")
    print()


def example_streaming():
    """Example of streaming with Anthropic."""
    print("=" * 50)
    print("ANTHROPIC STREAMING EXAMPLE")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "Count from 1 to 5. Just the numbers."},
    ]

    # Stream with optimization
    with client.messages.stream(
        model="claude-3-5-haiku-latest",
        messages=messages,
        headroom_mode="optimize",
        max_tokens=100,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()
    print()


if __name__ == "__main__":
    # Run examples
    example_audit_mode()
    example_optimize_mode()
    example_simulate_mode()
    example_streaming()

    # Clean up
    client.close()
