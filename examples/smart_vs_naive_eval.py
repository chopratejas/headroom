#!/usr/bin/env python3
"""
Comparison evaluation: SmartCrusher vs ToolCrusher (Naive Crushing)

This script compares:
1. ToolCrusher - Fixed rules (keep first N items)
2. SmartCrusher - Statistical analysis (preserve change points, factor constants)

We'll use the same SRE incident response scenario and see which produces
better token reduction while maintaining response quality.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

from headroom import HeadroomClient, OpenAIProvider, SmartCrusherConfig, ToolCrusherConfig
from headroom.config import HeadroomConfig
from headroom.transforms import TransformPipeline

load_dotenv(".env.local")

# Initialize base client
base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
provider = OpenAIProvider()


# =============================================================================
# CREATE THREE CLIENT CONFIGURATIONS
# =============================================================================

# 1. NAIVE CRUSHER (fixed rules - keep first 10 items)
naive_config = HeadroomConfig()
naive_config.tool_crusher = ToolCrusherConfig(
    enabled=True,
    min_tokens_to_crush=200,
    max_array_items=10,
    max_string_length=1000,
    max_depth=5,
)
naive_config.smart_crusher.enabled = False

naive_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{tempfile.gettempdir()}/headroom_naive.db",
    default_mode="audit",
)
naive_client._config = naive_config
naive_client._pipeline = TransformPipeline(naive_config, provider=provider)

# 2. SMART CRUSHER (statistical analysis)
smart_config = HeadroomConfig()
smart_config.tool_crusher.enabled = False  # Disable naive
smart_config.smart_crusher = SmartCrusherConfig(
    enabled=True,
    min_items_to_analyze=5,
    min_tokens_to_crush=200,
    variance_threshold=2.0,
    max_items_after_crush=15,
    preserve_change_points=True,
    factor_out_constants=True,
    include_summaries=True,
)

smart_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{tempfile.gettempdir()}/headroom_smart.db",
    default_mode="audit",
)
smart_client._config = smart_config
smart_client._pipeline = TransformPipeline(smart_config, provider=provider)

# 3. BASELINE (no optimization)
baseline_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{tempfile.gettempdir()}/headroom_baseline.db",
    default_mode="audit",
)


# =============================================================================
# GENERATE TEST DATA WITH CLEAR PATTERNS
# =============================================================================


def generate_metrics_with_spike() -> str:
    """
    Generate metrics data with a CLEAR spike pattern.
    SmartCrusher should detect and preserve the change point.
    NaiveCrusher will just keep first 10 items (missing the spike).
    """
    base_time = datetime.now() - timedelta(hours=1)

    data_points = []
    for i in range(60):
        ts = base_time + timedelta(minutes=i)

        # PATTERN: Stable at ~45 for first 45 minutes, then spike to 95
        if i < 45:
            cpu = 45 + (i % 3)  # Small variance: 45-47
            error_rate = 0.1
        else:
            cpu = 85 + (i - 45) * 2  # Spike: 85 -> 115
            error_rate = 5 + (i - 45)  # Error spike too

        data_points.append(
            {
                "timestamp": ts.isoformat(),
                "host": "prod-api-1",  # CONSTANT - should be factored out
                "region": "us-east-1",  # CONSTANT - should be factored out
                "datacenter": "dc-01",  # CONSTANT - should be factored out
                "cpu_percent": min(cpu, 99),
                "memory_percent": 62,  # CONSTANT
                "error_rate": round(error_rate, 2),
                "request_count": 1500 + (i * 10),
            }
        )

    return json.dumps({"status": "success", "metrics": data_points, "query_time_ms": 127})


def generate_clusterable_logs() -> str:
    """
    Generate logs with REPEATING patterns.
    SmartCrusher should cluster and dedupe.
    NaiveCrusher will just keep first 10.
    """
    base_time = datetime.now() - timedelta(minutes=30)

    # 4 distinct message types, repeated many times
    message_templates = [
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("WARN", "Connection pool exhausted, waiting for available connection"),
        ("WARN", "Connection pool exhausted, waiting for available connection"),
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("ERROR", "Max retries exceeded for database operation"),
        ("INFO", "Retry attempt 1/3 for database connection"),
        ("INFO", "Retry attempt 2/3 for database connection"),
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("WARN", "Circuit breaker OPEN for database-primary"),
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("ERROR", "OOM killed: api-server process exceeded memory limit"),  # UNIQUE
        ("ERROR", "Connection timeout to database-primary after 30000ms"),
        ("WARN", "Connection pool exhausted, waiting for available connection"),
    ]

    logs = []
    for i in range(50):
        ts = base_time + timedelta(seconds=i * 36)
        level, msg = message_templates[i % len(message_templates)]

        logs.append(
            {
                "@timestamp": ts.isoformat(),
                "level": level,
                "message": msg,
                "service": "api-server",  # CONSTANT
                "environment": "production",  # CONSTANT
                "version": "2.4.1",  # CONSTANT
                "host": f"prod-api-{i % 3 + 1}",
                "trace_id": f"trace-{1000 + i:04d}",
            }
        )

    return json.dumps({"took": 234, "hits": {"total": len(logs), "hits": logs}})


def generate_search_results() -> str:
    """
    Generate search results with scores.
    SmartCrusher should use TOP_N strategy.
    """
    results = []
    for i in range(30):
        results.append(
            {
                "id": f"doc-{i + 1}",
                "title": f"Result document {i + 1}",
                "snippet": f"This is the snippet for document {i + 1} with relevant content...",
                "score": 0.95 - (i * 0.02),  # Decreasing relevance
                "source": "knowledge_base",  # CONSTANT
                "category": "technical",  # CONSTANT
            }
        )

    return json.dumps({"results": results, "total": 30})


# =============================================================================
# BUILD TEST CONVERSATION
# =============================================================================


def build_test_conversation() -> list[dict]:
    """Build a conversation that exercises all SmartCrusher strategies."""

    messages = [
        {
            "role": "system",
            "content": """You are an SRE assistant. Analyze the data and provide insights.
Current Date: 2024-12-15T14:30:00Z""",
        },
        {"role": "user", "content": "Check the metrics for the last hour."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "query_metrics", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": generate_metrics_with_spike()},
        {"role": "assistant", "content": "I see CPU metrics. Let me check the logs."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search_logs", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": generate_clusterable_logs()},
        {"role": "assistant", "content": "Found error patterns. Let me search docs."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": generate_search_results()},
        {"role": "user", "content": "What's the root cause and what should we do?"},
    ]

    return messages


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvalResult:
    name: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    reduction_pct: float
    transforms: list[str]
    response: str
    latency_ms: float


def evaluate(client, messages: list[dict], name: str, mode: str) -> EvalResult:
    """Run evaluation with a client."""
    # Simulate first to get transform info
    sim = client.chat.completions.simulate(model="gpt-4o-mini", messages=messages)

    # Actual call
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        headroom_mode=mode,
    )
    latency = (time.time() - start) * 1000

    tokens_in = response.usage.prompt_tokens if response.usage else 0

    return EvalResult(
        name=name,
        tokens_before=sim.tokens_before,
        tokens_after=tokens_in,
        tokens_saved=sim.tokens_before - tokens_in,
        reduction_pct=(sim.tokens_before - tokens_in) / sim.tokens_before * 100
        if sim.tokens_before
        else 0,
        transforms=sim.transforms,
        response=response.choices[0].message.content or "",
        latency_ms=latency,
    )


def evaluate_quality(baseline_response: str, test_response: str, test_name: str) -> dict:
    """Use GPT-4o to judge response quality."""
    judge_prompt = f"""Compare these two AI responses to an SRE incident investigation.

BASELINE (unoptimized):
{baseline_response}

{test_name.upper()}:
{test_response}

Score the {test_name} response on:
1. ROOT_CAUSE: Does it correctly identify the database connection issue? (1-5)
2. DATA_AWARENESS: Does it reference specific metrics (CPU spike, error rate)? (1-5)
3. ACTIONABILITY: Does it provide concrete next steps? (1-5)

Respond in JSON:
{{
    "root_cause": {{"score": N, "reason": "..."}},
    "data_awareness": {{"score": N, "reason": "..."}},
    "actionability": {{"score": N, "reason": "..."}},
    "overall": N,
    "verdict": "PASS" or "FAIL"
}}

PASS = overall >= 4.0"""

    response = base_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"error": "Parse failed"}


def main():
    print("=" * 70)
    print("SMART CRUSHER vs NAIVE CRUSHER COMPARISON")
    print("=" * 70)
    print()

    messages = build_test_conversation()

    # Count tool outputs
    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"Test conversation: {len(messages)} messages, {tool_count} tool outputs")
    print()

    # Tool output breakdown
    print("Tool output patterns for SmartCrusher:")
    print("  1. Metrics: 60 data points with SPIKE at point 45")
    print("     - 3 CONSTANT fields (host, region, datacenter)")
    print("     - SmartCrusher should: detect spike, factor constants")
    print("  2. Logs: 50 entries with 4 REPEATING message types")
    print("     - SmartCrusher should: cluster and dedupe")
    print("  3. Search: 30 results with relevance scores")
    print("     - SmartCrusher should: keep top N by score")
    print()

    # Run evaluations
    print("-" * 70)
    print("RUNNING EVALUATIONS")
    print("-" * 70)

    print("\n1. BASELINE (no optimization)...")
    baseline = evaluate(baseline_client, messages, "Baseline", "audit")
    print(f"   Tokens: {baseline.tokens_after:,}")

    print("\n2. NAIVE CRUSHER (fixed rules: keep first 10)...")
    naive = evaluate(naive_client, messages, "Naive", "optimize")
    print(
        f"   Tokens: {naive.tokens_after:,} (saved {naive.tokens_saved:,}, {naive.reduction_pct:.1f}%)"
    )
    print(f"   Transforms: {naive.transforms}")

    print("\n3. SMART CRUSHER (statistical analysis)...")
    smart = evaluate(smart_client, messages, "Smart", "optimize")
    print(
        f"   Tokens: {smart.tokens_after:,} (saved {smart.tokens_saved:,}, {smart.reduction_pct:.1f}%)"
    )
    print(f"   Transforms: {smart.transforms}")

    # Results comparison
    print()
    print("=" * 70)
    print("TOKEN COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Tokens':>10} {'Saved':>10} {'Reduction':>10}")
    print("-" * 52)
    print(f"{'Baseline':<20} {baseline.tokens_after:>10,} {'-':>10} {'-':>10}")
    print(
        f"{'Naive Crusher':<20} {naive.tokens_after:>10,} {naive.tokens_saved:>10,} {naive.reduction_pct:>9.1f}%"
    )
    print(
        f"{'Smart Crusher':<20} {smart.tokens_after:>10,} {smart.tokens_saved:>10,} {smart.reduction_pct:>9.1f}%"
    )

    # Show the difference
    diff = naive.tokens_after - smart.tokens_after
    if diff > 0:
        print(
            f"\n→ Smart Crusher saves {diff:,} MORE tokens than Naive ({diff / naive.tokens_after * 100:.1f}% better)"
        )
    elif diff < 0:
        print(
            f"\n→ Naive Crusher saves {-diff:,} MORE tokens than Smart ({-diff / smart.tokens_after * 100:.1f}% better)"
        )
    else:
        print("\n→ Both methods produce same token count")

    # Quality evaluation
    print()
    print("-" * 70)
    print("QUALITY EVALUATION (GPT-4o Judge)")
    print("-" * 70)

    print("\nEvaluating Naive Crusher response...")
    naive_quality = evaluate_quality(baseline.response, naive.response, "naive crusher")

    print("Evaluating Smart Crusher response...")
    smart_quality = evaluate_quality(baseline.response, smart.response, "smart crusher")

    if "error" not in naive_quality and "error" not in smart_quality:
        print(f"\n{'Criterion':<20} {'Naive':>10} {'Smart':>10}")
        print("-" * 42)
        for criterion in ["root_cause", "data_awareness", "actionability"]:
            n_score = naive_quality.get(criterion, {}).get("score", "?")
            s_score = smart_quality.get(criterion, {}).get("score", "?")
            print(f"{criterion.replace('_', ' ').title():<20} {n_score:>10}/5 {s_score:>10}/5")
        print("-" * 42)
        print(
            f"{'OVERALL':<20} {naive_quality.get('overall', '?'):>10}/5 {smart_quality.get('overall', '?'):>10}/5"
        )
        print(
            f"{'VERDICT':<20} {naive_quality.get('verdict', '?'):>10} {smart_quality.get('verdict', '?'):>10}"
        )

        print("\n[Quality Analysis]")
        print(f"  Naive: {naive_quality.get('data_awareness', {}).get('reason', 'N/A')}")
        print(f"  Smart: {smart_quality.get('data_awareness', {}).get('reason', 'N/A')}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_verdict = naive_quality.get("verdict", "?") if "error" not in naive_quality else "ERROR"
    s_verdict = smart_quality.get("verdict", "?") if "error" not in smart_quality else "ERROR"
    n_overall = naive_quality.get("overall", 0) if "error" not in naive_quality else 0
    s_overall = smart_quality.get("overall", 0) if "error" not in smart_quality else 0

    print(f"""
SmartCrusher vs NaiveCrusher on SRE incident data:

Token Efficiency:
  - Naive:  {naive.reduction_pct:.1f}% reduction
  - Smart:  {smart.reduction_pct:.1f}% reduction
  - Winner: {"SMART" if smart.reduction_pct > naive.reduction_pct else "NAIVE" if naive.reduction_pct > smart.reduction_pct else "TIE"} (+{abs(smart.reduction_pct - naive.reduction_pct):.1f}% {"more" if smart.reduction_pct > naive.reduction_pct else "less"} reduction)

Response Quality:
  - Naive:  {n_overall}/5 ({n_verdict})
  - Smart:  {s_overall}/5 ({s_verdict})
  - Winner: {"SMART" if s_overall > n_overall else "NAIVE" if n_overall > s_overall else "TIE"}

Key Insight:
  SmartCrusher uses statistical analysis to preserve important data:
  - Change points (CPU spike at minute 45) are PRESERVED
  - Constants (host, region, datacenter) are FACTORED OUT
  - Logs are CLUSTERED by message similarity
  - Search results keep TOP items by score

  NaiveCrusher blindly keeps first N items, potentially missing the spike!
""")


if __name__ == "__main__":
    main()
