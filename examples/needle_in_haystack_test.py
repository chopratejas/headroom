#!/usr/bin/env python3
"""
Needle-in-Haystack Test: Proving Headroom Preserves Critical Information

This test embeds a SPECIFIC answer deep in a large JSON response,
then verifies both baseline and Headroom-optimized calls can find it.

The scenario: Production incident logs with ONE critical error buried in the middle.
The question: "What caused the outage and what's the fix?"
"""

import json
import os

import httpx

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable required")

# =============================================================================
# THE NEEDLE: A critical error buried at position 67 out of 100
# =============================================================================
CRITICAL_ERROR = {
    "timestamp": "2024-12-15T03:47:23Z",
    "level": "FATAL",
    "service": "payment-gateway",
    "message": "Connection pool exhausted - max_connections=100 exceeded",
    "error_code": "PG-5523",
    "stack_trace": "ConnectionPoolError at PaymentService.processTransaction()",
    "resolution": "Increase max_connections to 500 in config/database.yml",
    "affected_transactions": 1847,
    "incident_id": "INC-2024-1215-001",
}


# =============================================================================
# THE HAYSTACK: 99 normal log entries
# =============================================================================
def create_log_entries(n: int = 100, needle_position: int = 67) -> list:
    """Create n log entries with the critical error at needle_position."""
    logs = []
    for i in range(n):
        if i == needle_position:
            # Insert the needle
            logs.append(CRITICAL_ERROR)
        else:
            # Normal log entry
            logs.append(
                {
                    "timestamp": f"2024-12-15T{(i % 24):02d}:{(i % 60):02d}:00Z",
                    "level": "INFO",
                    "service": ["api-gateway", "user-service", "inventory", "auth"][i % 4],
                    "message": f"Request processed successfully - latency={50 + (i % 100)}ms",
                    "request_id": f"req-{i:06d}",
                    "status_code": 200,
                    "endpoint": ["/api/users", "/api/products", "/api/orders", "/health"][i % 4],
                    "metadata": {
                        "region": ["us-east-1", "us-west-2", "eu-west-1"][i % 3],
                        "version": "2.4.1",
                    },
                }
            )
    return logs


# =============================================================================
# THE QUESTION
# =============================================================================
QUESTION = """Based on these production logs, answer these specific questions:

1. What service caused the outage?
2. What was the exact error code?
3. What is the specific fix mentioned in the logs?
4. How many transactions were affected?

Be precise - cite the exact values from the logs."""


def create_messages(tool_content: str) -> list:
    """Create the conversation with tool output."""
    return [
        {
            "role": "user",
            "content": "Search the production logs for the root cause of last night's outage",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll search the production logs for errors."},
                {
                    "type": "tool_use",
                    "id": "logs_1",
                    "name": "search_logs",
                    "input": {"query": "error OR fatal", "limit": 100},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "logs_1",
                    "content": tool_content,
                }
            ],
        },
        {
            "role": "user",
            "content": QUESTION,
        },
    ]


def make_api_call(base_url: str, messages: list, label: str) -> dict:
    """Make API call and return response + usage."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": messages,
        "tools": [
            {
                "name": "search_logs",
                "description": "Search production logs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ],
    }

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/v1/messages",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text[:500])
                return {"error": response.text}

            data = response.json()
            usage = data.get("usage", {})

            # Extract text response
            response_text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "response": response_text,
            }

    except Exception as e:
        print(f"Exception: {e}")
        return {"error": str(e)}


def verify_answer(response: str) -> dict:
    """Check if the response contains the correct answers."""
    checks = {
        "service": "payment-gateway" in response.lower(),
        "error_code": "PG-5523" in response or "pg-5523" in response.lower(),
        "fix": "max_connections" in response.lower() or "500" in response,
        "transactions": "1847" in response or "1,847" in response,
    }
    return checks


def main():
    print("\n" + "=" * 70)
    print("NEEDLE-IN-HAYSTACK TEST")
    print("Proving Headroom preserves critical information")
    print("=" * 70)

    # Create the test data
    logs = create_log_entries(100, needle_position=67)
    log_json = json.dumps(logs)

    print("\nTest setup:")
    print(f"  - Total log entries: {len(logs)}")
    print("  - Critical error at position: 67")
    print(f"  - JSON size: {len(log_json):,} characters")

    print("\nThe needle (what we're looking for):")
    print(f"  - Service: {CRITICAL_ERROR['service']}")
    print(f"  - Error code: {CRITICAL_ERROR['error_code']}")
    print(f"  - Fix: {CRITICAL_ERROR['resolution']}")
    print(f"  - Affected: {CRITICAL_ERROR['affected_transactions']} transactions")

    messages = create_messages(log_json)

    # Test 1: Baseline (direct to Anthropic)
    print("\n" + "-" * 70)
    baseline = make_api_call(
        "https://api.anthropic.com",
        messages,
        "BASELINE: Direct to Anthropic API",
    )

    if "error" not in baseline:
        print(f"\nInput tokens: {baseline['input_tokens']:,}")
        print(f"\nResponse:\n{baseline['response'][:1000]}...")
        baseline_checks = verify_answer(baseline["response"])
        print(f"\nAnswer verification: {baseline_checks}")

    # Test 2: Headroom optimized
    print("\n" + "-" * 70)
    optimized = make_api_call(
        "http://localhost:8787",
        messages,
        "HEADROOM: Through optimization proxy",
    )

    if "error" not in optimized:
        print(f"\nInput tokens: {optimized['input_tokens']:,}")
        print(f"\nResponse:\n{optimized['response'][:1000]}...")
        optimized_checks = verify_answer(optimized["response"])
        print(f"\nAnswer verification: {optimized_checks}")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if "error" not in baseline and "error" not in optimized:
        baseline_input = baseline["input_tokens"]
        optimized_input = optimized["input_tokens"]
        saved = baseline_input - optimized_input
        percent = (saved / baseline_input * 100) if baseline_input > 0 else 0

        baseline_score = sum(baseline_checks.values())
        optimized_score = sum(optimized_checks.values())

        print(f"""
TOKEN COMPARISON:
  Baseline:  {baseline_input:,} tokens
  Headroom:  {optimized_input:,} tokens
  Saved:     {saved:,} tokens ({percent:.1f}% reduction)

ANSWER ACCURACY (4 questions):
  Baseline:  {baseline_score}/4 correct
  Headroom:  {optimized_score}/4 correct

VERIFICATION DETAILS:
  Baseline: {baseline_checks}
  Headroom: {optimized_checks}

CONCLUSION:
  {"PASS - Headroom found the needle!" if optimized_score >= 3 else "NEEDS REVIEW"}
  {"Answer quality maintained with " + f"{percent:.0f}% fewer tokens" if optimized_score >= baseline_score else ""}
""")

        return {
            "baseline_tokens": baseline_input,
            "optimized_tokens": optimized_input,
            "tokens_saved": saved,
            "percent_saved": percent,
            "baseline_accuracy": baseline_score,
            "optimized_accuracy": optimized_score,
            "baseline_checks": baseline_checks,
            "optimized_checks": optimized_checks,
            "baseline_response": baseline["response"],
            "optimized_response": optimized["response"],
        }

    return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + "=" * 70)
        print("JSON SUMMARY:")
        print("=" * 70)
        summary = {k: v for k, v in result.items() if not k.endswith("_response")}
        print(json.dumps(summary, indent=2))
