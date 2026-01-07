#!/usr/bin/env python3
"""
Streaming example for Headroom SDK.

This example shows how to use Headroom with streaming responses.
"""

import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

from headroom import HeadroomClient, OpenAIProvider

# Load API key from .env.local
load_dotenv(".env.local")

# Create base OpenAI client
base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-..."))

# Create provider for OpenAI models
provider = OpenAIProvider()

# Use temp directory for database
db_path = os.path.join(tempfile.gettempdir(), "headroom_streaming.db")

# Wrap with Headroom
client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path}",
    default_mode="optimize",
)


def stream_example():
    """Example of streaming with Headroom."""
    print("Streaming response:")
    print("-" * 40)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Current Date: 2024-01-15. Be concise.",
        },
        {"role": "user", "content": "Count from 1 to 5 slowly."},
    ]

    # Stream with optimization
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        headroom_mode="optimize",
        max_tokens=100,
    )

    # Iterate over chunks
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()
    print("-" * 40)
    print("Stream complete!")


if __name__ == "__main__":
    stream_example()
    client.close()
