#!/usr/bin/env python3
"""
Real before/after test - NO MARKETING, JUST FACTS.

This script makes actual API calls to demonstrate Headroom compression.
"""

import json
import os

import httpx

# API Key from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable required")

# Realistic tool output: 100 search results from a code search
MOCK_TOOL_OUTPUT = json.dumps(
    [
        {
            "file": f"src/components/{['Button', 'Modal', 'Form', 'Table', 'Card'][i % 5]}.tsx",
            "line": 10 + (i * 3),
            "content": f"export function {['Button', 'Modal', 'Form', 'Table', 'Card'][i % 5]}Component{i}(props: Props) {{",
            "language": "typescript",
            "repository": "frontend-app",
            "branch": "main",
            "last_modified": "2024-12-15T10:00:00Z",
            "author": f"dev{i % 10}@company.com",
            "match_score": 0.95 - (i * 0.005),
            "context": {
                "before": ["import React from 'react';", "import { useCallback } from 'react';"],
                "after": ["  return <div>...</div>;", "}"],
            },
            "metadata": {
                "size_bytes": 1500 + (i * 10),
                "encoding": "utf-8",
                "mime_type": "text/typescript",
            },
        }
        for i in range(100)
    ]
)


# The conversation we'll send
def create_messages(tool_content: str) -> list:
    return [
        {"role": "user", "content": "Find all React components that use forms"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll search for React form components."},
                {
                    "type": "tool_use",
                    "id": "search_1",
                    "name": "code_search",
                    "input": {"query": "React form component", "limit": 100},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "search_1", "content": tool_content}
            ],
        },
    ]


def count_tokens_anthropic(text: str) -> int:
    """Rough token estimate (actual would use anthropic tokenizer)"""
    # Claude's tokenizer is roughly 4 chars per token for JSON
    return len(text) // 4


def make_api_call(base_url: str, messages: list, label: str) -> dict:
    """Make actual API call and return usage stats."""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 500,
        "messages": messages,
        "tools": [
            {
                "name": "code_search",
                "description": "Search for code in the repository",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                },
            }
        ],
    }

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Endpoint: {base_url}")

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{base_url}/v1/messages", headers=headers, json=payload)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text[:500])
                return {"error": response.text}

            data = response.json()
            usage = data.get("usage", {})

            result = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "response_preview": str(data.get("content", [{}])[0].get("text", ""))[:200],
            }

            print(f"Input tokens:  {result['input_tokens']:,}")
            print(f"Output tokens: {result['output_tokens']:,}")
            print(f"Response: {result['response_preview']}...")

            return result

    except Exception as e:
        print(f"Exception: {e}")
        return {"error": str(e)}


def main():
    print("\n" + "=" * 70)
    print("HEADROOM REAL BEFORE/AFTER TEST")
    print("NO MARKETING - JUST ACTUAL API RESULTS")
    print("=" * 70)

    # Show what we're testing
    print(f"\nTest data: {len(json.loads(MOCK_TOOL_OUTPUT))} code search results")
    print(f"Raw JSON size: {len(MOCK_TOOL_OUTPUT):,} characters")
    print(f"Estimated tokens: ~{count_tokens_anthropic(MOCK_TOOL_OUTPUT):,}")

    messages = create_messages(MOCK_TOOL_OUTPUT)

    # Test 1: Direct to Anthropic API (baseline)
    baseline = make_api_call(
        "https://api.anthropic.com", messages, "BASELINE: Direct to Anthropic API"
    )

    # Test 2: Through Headroom proxy
    optimized = make_api_call(
        "http://localhost:8787", messages, "OPTIMIZED: Through Headroom Proxy"
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if "error" not in baseline and "error" not in optimized:
        baseline_input = baseline["input_tokens"]
        optimized_input = optimized["input_tokens"]
        saved = baseline_input - optimized_input
        percent = (saved / baseline_input * 100) if baseline_input > 0 else 0

        # Cost calculation (Claude Sonnet: $3/1M input, $15/1M output)
        cost_baseline = (baseline_input * 3 + baseline["output_tokens"] * 15) / 1_000_000
        cost_optimized = (optimized_input * 3 + optimized["output_tokens"] * 15) / 1_000_000
        cost_saved = cost_baseline - cost_optimized

        print(f"""
Input Tokens:
  Baseline:  {baseline_input:,}
  Optimized: {optimized_input:,}
  Saved:     {saved:,} ({percent:.1f}%)

Cost per request (Claude Sonnet pricing):
  Baseline:  ${cost_baseline:.6f}
  Optimized: ${cost_optimized:.6f}
  Saved:     ${cost_saved:.6f}

At 10,000 requests/day:
  Daily savings:   ${cost_saved * 10000:.2f}
  Monthly savings: ${cost_saved * 10000 * 30:.2f}
""")

        # Return data for README
        return {
            "baseline_tokens": baseline_input,
            "optimized_tokens": optimized_input,
            "tokens_saved": saved,
            "percent_saved": percent,
            "tool_output_size": len(MOCK_TOOL_OUTPUT),
            "num_items": len(json.loads(MOCK_TOOL_OUTPUT)),
        }
    else:
        print("Test failed - check errors above")
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + "=" * 70)
        print("JSON FOR README:")
        print("=" * 70)
        print(json.dumps(result, indent=2))
