#!/usr/bin/env python3
"""
Real-world OpenAI Agentic Evaluation for Headroom SDK.

Scenario: DevOps/SRE Agent investigating a production incident
- This is NOT hand-crafted for Headroom optimizations
- Tool outputs are realistic sizes based on actual tool responses
- The conversation flow mirrors real incident response

Tools used:
1. query_metrics - Prometheus/Datadog style metrics
2. search_logs - ELK/Splunk style log search
3. get_service_status - Health check endpoints
4. query_deployments - CI/CD deployment history
5. get_runbook - Documentation lookup
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

from headroom import HeadroomClient, OpenAIProvider, ToolCrusherConfig
from headroom.config import HeadroomConfig
from headroom.transforms import TransformPipeline

load_dotenv(".env.local")

# Initialize
base_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
provider = OpenAIProvider()

db_path = os.path.join(tempfile.gettempdir(), "headroom_openai_eval.db")
client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path}",
    default_mode="audit",
)

# Aggressive config
aggressive_config = HeadroomConfig()
aggressive_config.tool_crusher = ToolCrusherConfig(
    min_tokens_to_crush=100,
    max_array_items=5,
    max_string_length=300,
    max_depth=4,
)

db_path_agg = os.path.join(tempfile.gettempdir(), "headroom_openai_eval_agg.db")
aggressive_client = HeadroomClient(
    original_client=base_client,
    provider=provider,
    store_url=f"sqlite:///{db_path_agg}",
    default_mode="audit",
)
aggressive_client._config = aggressive_config
aggressive_client._pipeline = TransformPipeline(aggressive_config, provider=provider)


# =============================================================================
# REALISTIC TOOL OUTPUTS - Based on actual production systems
# =============================================================================


def generate_metrics_response() -> str:
    """
    Realistic Prometheus/Datadog metrics query response.
    This is what you actually get from a metrics API.
    """
    base_time = datetime.now() - timedelta(hours=1)

    # CPU metrics - 60 data points (1 per minute for 1 hour)
    cpu_data = []
    for i in range(60):
        ts = base_time + timedelta(minutes=i)
        # Simulate spike around minute 45
        value = 45 + (i * 0.5) if i < 45 else 85 + (i - 45) * 2
        cpu_data.append(
            {
                "timestamp": ts.isoformat(),
                "value": min(value, 98),
                "labels": {"instance": "prod-api-1", "job": "api-server"},
            }
        )

    # Memory metrics
    memory_data = []
    for i in range(60):
        ts = base_time + timedelta(minutes=i)
        value = 62 + (i * 0.3)
        memory_data.append(
            {
                "timestamp": ts.isoformat(),
                "value": min(value, 89),
                "labels": {"instance": "prod-api-1", "job": "api-server"},
            }
        )

    # Request latency (p99)
    latency_data = []
    for i in range(60):
        ts = base_time + timedelta(minutes=i)
        value = 120 if i < 45 else 450 + (i - 45) * 50
        latency_data.append(
            {
                "timestamp": ts.isoformat(),
                "value": min(value, 2500),
                "labels": {"instance": "prod-api-1", "endpoint": "/api/v1/users"},
            }
        )

    # Error rate
    error_data = []
    for i in range(60):
        ts = base_time + timedelta(minutes=i)
        value = 0.1 if i < 45 else 2.5 + (i - 45) * 0.5
        error_data.append(
            {
                "timestamp": ts.isoformat(),
                "value": min(value, 15),
                "labels": {"instance": "prod-api-1", "status_code": "5xx"},
            }
        )

    return json.dumps(
        {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {"__name__": "cpu_usage_percent"}, "values": cpu_data},
                    {"metric": {"__name__": "memory_usage_percent"}, "values": memory_data},
                    {
                        "metric": {"__name__": "http_request_duration_p99_ms"},
                        "values": latency_data,
                    },
                    {"metric": {"__name__": "http_errors_rate_percent"}, "values": error_data},
                ],
            },
            "query_time_ms": 127,
        }
    )


def generate_logs_response() -> str:
    """
    Realistic ELK/Splunk log search response.
    This is what production log searches actually return.
    """
    base_time = datetime.now() - timedelta(minutes=30)

    logs = []
    log_templates = [
        ("ERROR", "Connection pool exhausted, waiting for available connection", "api-server"),
        (
            "WARN",
            "Slow query detected: SELECT * FROM users WHERE status = 'active' took 2.3s",
            "api-server",
        ),
        ("ERROR", "Database connection timeout after 30000ms", "api-server"),
        ("INFO", "Retry attempt 1/3 for database connection", "api-server"),
        ("ERROR", "Max retries exceeded for database operation", "api-server"),
        ("WARN", "Circuit breaker OPEN for database-primary", "api-server"),
        ("ERROR", "Failed to process request: upstream connect error", "api-server"),
        ("INFO", "Health check failed for database-primary", "health-checker"),
        ("ERROR", "PostgreSQL: too many connections for role 'api_user'", "database"),
        ("WARN", "Connection refused to database-primary:5432", "api-server"),
        ("ERROR", "Request timeout: /api/v1/users after 30s", "nginx"),
        ("INFO", "Scaling up api-server replicas from 3 to 5", "autoscaler"),
        ("ERROR", "OOM killed process api-server (pid 12345)", "kernel"),
        ("WARN", "Memory pressure detected, initiating garbage collection", "jvm"),
        ("ERROR", "Unhandled exception in request handler", "api-server"),
    ]

    for i in range(40):  # 40 log entries
        ts = base_time + timedelta(seconds=i * 45)
        level, msg, source = log_templates[i % len(log_templates)]

        logs.append(
            {
                "@timestamp": ts.isoformat(),
                "level": level,
                "message": msg,
                "service": source,
                "trace_id": f"trace-{1000 + i:04d}-abcd-{i:04d}",
                "span_id": f"span-{i:04d}",
                "host": f"prod-{source}-{i % 3 + 1}",
                "environment": "production",
                "version": "2.4.1",
                "kubernetes": {
                    "namespace": "production",
                    "pod": f"{source}-{i % 5 + 1}-abc123",
                    "container": source,
                    "node": f"node-{i % 3 + 1}.prod.internal",
                },
                "request": {
                    "method": "GET" if i % 2 == 0 else "POST",
                    "path": "/api/v1/users" if i % 3 == 0 else "/api/v1/orders",
                    "status_code": 500 if level == "ERROR" else 200,
                    "duration_ms": 150 + (i * 100) if level != "ERROR" else 30000,
                },
            }
        )

    return json.dumps(
        {
            "took": 234,
            "timed_out": False,
            "hits": {
                "total": {"value": len(logs), "relation": "eq"},
                "max_score": 1.0,
                "hits": logs,
            },
        }
    )


def generate_service_status() -> str:
    """
    Realistic health check / service status response.
    """
    return json.dumps(
        {
            "services": [
                {
                    "name": "api-server",
                    "status": "degraded",
                    "instances": [
                        {
                            "id": "api-1",
                            "status": "unhealthy",
                            "cpu": 94,
                            "memory": 87,
                            "connections": 500,
                        },
                        {
                            "id": "api-2",
                            "status": "healthy",
                            "cpu": 45,
                            "memory": 62,
                            "connections": 150,
                        },
                        {
                            "id": "api-3",
                            "status": "unhealthy",
                            "cpu": 91,
                            "memory": 85,
                            "connections": 480,
                        },
                    ],
                    "last_check": datetime.now().isoformat(),
                    "error_rate": 12.5,
                    "p99_latency_ms": 2100,
                },
                {
                    "name": "database-primary",
                    "status": "critical",
                    "instances": [
                        {
                            "id": "db-primary",
                            "status": "unhealthy",
                            "connections": 500,
                            "max_connections": 500,
                            "replication_lag_ms": 0,
                            "disk_usage_percent": 78,
                        }
                    ],
                    "last_check": datetime.now().isoformat(),
                    "active_queries": 487,
                    "blocked_queries": 52,
                },
                {
                    "name": "database-replica",
                    "status": "healthy",
                    "instances": [
                        {
                            "id": "db-replica-1",
                            "status": "healthy",
                            "connections": 120,
                            "max_connections": 500,
                            "replication_lag_ms": 150,
                            "disk_usage_percent": 76,
                        },
                        {
                            "id": "db-replica-2",
                            "status": "healthy",
                            "connections": 115,
                            "max_connections": 500,
                            "replication_lag_ms": 180,
                            "disk_usage_percent": 77,
                        },
                    ],
                    "last_check": datetime.now().isoformat(),
                },
                {
                    "name": "redis-cache",
                    "status": "healthy",
                    "instances": [
                        {
                            "id": "redis-1",
                            "status": "healthy",
                            "memory_used_mb": 2048,
                            "memory_max_mb": 4096,
                            "connected_clients": 45,
                            "hit_rate": 0.94,
                        }
                    ],
                    "last_check": datetime.now().isoformat(),
                },
                {
                    "name": "nginx-ingress",
                    "status": "healthy",
                    "instances": [
                        {
                            "id": "nginx-1",
                            "status": "healthy",
                            "active_connections": 1250,
                            "requests_per_sec": 450,
                        },
                        {
                            "id": "nginx-2",
                            "status": "healthy",
                            "active_connections": 1180,
                            "requests_per_sec": 420,
                        },
                    ],
                    "last_check": datetime.now().isoformat(),
                },
            ],
            "overall_status": "critical",
            "timestamp": datetime.now().isoformat(),
        }
    )


def generate_deployments_response() -> str:
    """
    Realistic CI/CD deployment history.
    """
    base_time = datetime.now()

    deployments = []
    for i in range(15):
        ts = base_time - timedelta(hours=i * 4)
        deployments.append(
            {
                "id": f"deploy-{1000 - i}",
                "service": "api-server" if i % 3 != 2 else "database-migration",
                "version": f"2.4.{15 - i}",
                "status": "success" if i != 1 else "success",  # Recent deploy
                "timestamp": ts.isoformat(),
                "commit": f"abc{i:04d}def",
                "author": f"dev{i % 5 + 1}@company.com",
                "message": [
                    "feat: Add new user endpoint",
                    "fix: Connection pool sizing",
                    "chore: Update dependencies",
                    "feat: Implement caching layer",
                    "fix: Memory leak in request handler",
                ][i % 5],
                "changes": {
                    "files_changed": 5 + i,
                    "insertions": 100 + i * 20,
                    "deletions": 30 + i * 5,
                },
                "rollback_available": True,
                "canary_status": "completed" if i > 0 else "in_progress",
            }
        )

    return json.dumps(
        {"deployments": deployments, "total_count": len(deployments), "page": 1, "per_page": 20}
    )


def generate_runbook_response() -> str:
    """
    Realistic runbook/documentation lookup.
    """
    return json.dumps(
        {
            "runbook": {
                "title": "Database Connection Pool Exhaustion",
                "id": "RUN-DB-001",
                "severity": "P1",
                "last_updated": "2024-11-15",
                "owner": "platform-team",
                "symptoms": [
                    "High error rate on API endpoints",
                    "Connection timeout errors in logs",
                    "Database showing max connections reached",
                    "Increased latency across all services",
                ],
                "diagnosis_steps": [
                    "1. Check current connection count: SELECT count(*) FROM pg_stat_activity",
                    "2. Identify connection holders: SELECT * FROM pg_stat_activity WHERE state != 'idle'",
                    "3. Check for long-running queries: SELECT * FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '1 minute'",
                    "4. Verify connection pool settings in application config",
                    "5. Check for connection leaks in recent deployments",
                ],
                "remediation_steps": [
                    "1. IMMEDIATE: Kill idle connections older than 10 minutes",
                    "2. IMMEDIATE: Scale up API server replicas to distribute load",
                    "3. SHORT-TERM: Increase max_connections on database (requires restart)",
                    "4. SHORT-TERM: Review and optimize connection pool settings",
                    "5. LONG-TERM: Implement connection pooler (PgBouncer)",
                ],
                "commands": {
                    "kill_idle_connections": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '10 minutes'",
                    "check_pool_settings": "kubectl get configmap api-server-config -o yaml | grep -A5 'database'",
                    "scale_replicas": "kubectl scale deployment api-server --replicas=5",
                },
                "related_incidents": ["INC-2024-089", "INC-2024-056", "INC-2024-023"],
                "escalation_path": ["on-call-sre", "database-team", "platform-lead"],
            }
        }
    )


# =============================================================================
# BUILD REALISTIC INCIDENT RESPONSE CONVERSATION
# =============================================================================


def build_incident_conversation() -> list[dict]:
    """
    Build a realistic incident response agentic conversation.
    This mirrors how SRE teams actually investigate incidents.
    """

    messages = [
        # System prompt (realistic for an SRE agent)
        {
            "role": "system",
            "content": """You are an SRE incident response assistant. You help investigate and resolve production incidents.

Current Date: 2024-12-15T14:30:00Z
On-call: @platform-team
Incident Channel: #inc-2024-142

You have access to the following tools:
- query_metrics: Query Prometheus metrics
- search_logs: Search ELK logs
- get_service_status: Get current service health
- query_deployments: Get recent deployment history
- get_runbook: Lookup runbook documentation

Always be concise and focus on actionable insights.""",
        },
        # User reports incident
        {
            "role": "user",
            "content": "We're seeing elevated error rates on the API. Users reporting timeouts. Can you investigate?",
        },
        # Agent starts investigation - queries metrics
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_metrics_1",
                    "type": "function",
                    "function": {
                        "name": "query_metrics",
                        "arguments": json.dumps(
                            {
                                "query": "api_error_rate, cpu_usage, memory_usage, latency_p99",
                                "time_range": "1h",
                            }
                        ),
                    },
                }
            ],
        },
        # Metrics response
        {"role": "tool", "tool_call_id": "call_metrics_1", "content": generate_metrics_response()},
        # Agent analyzes and queries logs
        {
            "role": "assistant",
            "content": "I see a clear pattern: around 45 minutes ago, CPU spiked to 95%, latency jumped from 120ms to 2500ms, and error rate went from 0.1% to 15%. Let me check the logs for that timeframe.",
            "tool_calls": [
                {
                    "id": "call_logs_1",
                    "type": "function",
                    "function": {
                        "name": "search_logs",
                        "arguments": json.dumps(
                            {
                                "query": "level:ERROR OR level:WARN",
                                "time_range": "30m",
                                "services": ["api-server", "database"],
                            }
                        ),
                    },
                }
            ],
        },
        # Logs response
        {"role": "tool", "tool_call_id": "call_logs_1", "content": generate_logs_response()},
        # Agent identifies root cause, checks service status
        {
            "role": "assistant",
            "content": "The logs show database connection issues: 'Connection pool exhausted', 'too many connections', and 'connection timeout'. This is a database connection saturation issue. Let me check current service status.",
            "tool_calls": [
                {
                    "id": "call_status_1",
                    "type": "function",
                    "function": {"name": "get_service_status", "arguments": json.dumps({})},
                }
            ],
        },
        # Service status response
        {"role": "tool", "tool_call_id": "call_status_1", "content": generate_service_status()},
        # Agent checks for recent deployments
        {
            "role": "assistant",
            "content": "Confirmed: database-primary is CRITICAL with 500/500 connections (100% saturated), 52 blocked queries. API servers are degraded. Let me check if a recent deployment might have caused this.",
            "tool_calls": [
                {
                    "id": "call_deploys_1",
                    "type": "function",
                    "function": {
                        "name": "query_deployments",
                        "arguments": json.dumps({"service": "api-server", "limit": 15}),
                    },
                }
            ],
        },
        # Deployments response
        {
            "role": "tool",
            "tool_call_id": "call_deploys_1",
            "content": generate_deployments_response(),
        },
        # Agent gets runbook
        {
            "role": "assistant",
            "content": "I see deploy-999 (version 2.4.14) was deployed 4 hours ago with 'fix: Connection pool sizing' - this might have changed pool settings. There's also a canary deployment in progress. Let me get the runbook for this issue.",
            "tool_calls": [
                {
                    "id": "call_runbook_1",
                    "type": "function",
                    "function": {
                        "name": "get_runbook",
                        "arguments": json.dumps({"search": "database connection pool exhaustion"}),
                    },
                }
            ],
        },
        # Runbook response
        {"role": "tool", "tool_call_id": "call_runbook_1", "content": generate_runbook_response()},
        # Agent provides final analysis
        {
            "role": "assistant",
            "content": """**Incident Summary: Database Connection Pool Exhaustion**

**Root Cause:** The database primary is at 100% connection capacity (500/500). This started ~45 minutes ago, likely related to deploy-999 which modified connection pool settings.

**Impact:**
- API error rate: 15% (up from 0.1%)
- P99 latency: 2500ms (up from 120ms)
- 2 of 3 API instances unhealthy

**Immediate Actions Recommended:**
1. Kill idle database connections (per runbook RUN-DB-001)
2. Scale API servers to 5 replicas to distribute load
3. Consider rolling back deploy-999 if connection pool change is confirmed

Shall I proceed with the remediation steps?""",
        },
        # User asks for action
        {
            "role": "user",
            "content": "Yes, give me the exact commands to run and summarize the incident for the post-mortem.",
        },
    ]

    return messages


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvalResult:
    mode: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    response: str
    cost_estimate: float


def evaluate_response_quality(
    baseline_response: str, optimized_response: str, optimization_level: str
) -> dict:
    """
    Use GPT-4o as judge to evaluate if the optimized response maintains quality.
    """
    judge_prompt = f"""You are evaluating two AI assistant responses to an SRE incident investigation request.
The user asked: "Give me the exact commands to run and summarize the incident for the post-mortem."

BASELINE RESPONSE (no optimization):
{baseline_response}

OPTIMIZED RESPONSE ({optimization_level} optimization):
{optimized_response}

Evaluate the optimized response compared to baseline on these criteria:
1. CORRECTNESS: Are the SQL/kubectl commands identical and correct? (1-5)
2. COMPLETENESS: Does it include all necessary remediation steps? (1-5)
3. INCIDENT SUMMARY: Is the post-mortem summary accurate and complete? (1-5)
4. ACTIONABILITY: Is the response equally actionable for an SRE? (1-5)

For each criterion, score 1-5 where:
- 5 = Equivalent or better than baseline
- 4 = Minor differences, still fully usable
- 3 = Some information missing but acceptable
- 2 = Noticeable quality degradation
- 1 = Critical information missing

Respond in this exact JSON format:
{{
    "correctness": {{"score": N, "reason": "..."}},
    "completeness": {{"score": N, "reason": "..."}},
    "incident_summary": {{"score": N, "reason": "..."}},
    "actionability": {{"score": N, "reason": "..."}},
    "overall_score": N,
    "verdict": "PASS" or "FAIL",
    "summary": "One sentence summary"
}}

PASS means overall_score >= 4.0, FAIL means < 4.0."""

    response = base_client.chat.completions.create(
        model="gpt-4o",  # Using stronger model for judging
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    import json as json_module

    try:
        return json_module.loads(response.choices[0].message.content)
    except Exception:
        return {
            "error": "Failed to parse judge response",
            "raw": response.choices[0].message.content,
        }


def run_eval(messages: list[dict], mode: str, use_aggressive: bool = False) -> EvalResult:
    """Run evaluation."""
    c = aggressive_client if use_aggressive else client

    start_time = time.time()
    response = c.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
        headroom_mode=mode,
    )
    latency_ms = (time.time() - start_time) * 1000

    response_text = response.choices[0].message.content or ""
    tokens_in = response.usage.prompt_tokens if response.usage else 0
    tokens_out = response.usage.completion_tokens if response.usage else 0

    # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
    cost = (tokens_in / 1_000_000) * 0.15 + (tokens_out / 1_000_000) * 0.60

    return EvalResult(
        mode=mode,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=latency_ms,
        response=response_text,
        cost_estimate=cost,
    )


def main():
    print("=" * 70)
    print("REAL-WORLD OPENAI AGENTIC EVALUATION")
    print("Scenario: SRE Incident Response Investigation")
    print("=" * 70)
    print()

    messages = build_incident_conversation()

    # Count tool outputs
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    print(f"Conversation: {len(messages)} messages")
    print(f"Tool calls: {len(tool_messages)} (metrics, logs, status, deployments, runbook)")
    print()

    # Simulations
    print("-" * 70)
    print("SIMULATIONS")
    print("-" * 70)

    sim_default = client.chat.completions.simulate(model="gpt-4o-mini", messages=messages)
    sim_aggressive = aggressive_client.chat.completions.simulate(
        model="gpt-4o-mini", messages=messages
    )

    print(f"\n{'Mode':<15} {'Before':>10} {'After':>10} {'Saved':>10} {'%':>8}")
    print("-" * 55)
    print(
        f"{'Default':<15} {sim_default.tokens_before:>10,} {sim_default.tokens_after:>10,} {sim_default.tokens_saved:>10,} {sim_default.tokens_saved / sim_default.tokens_before * 100:>7.1f}%"
    )
    print(
        f"{'Aggressive':<15} {sim_aggressive.tokens_before:>10,} {sim_aggressive.tokens_after:>10,} {sim_aggressive.tokens_saved:>10,} {sim_aggressive.tokens_saved / sim_aggressive.tokens_before * 100:>7.1f}%"
    )
    print(f"\nTransforms: {sim_default.transforms}")
    print()

    # Actual API calls
    print("-" * 70)
    print("ACTUAL API CALLS")
    print("-" * 70)

    print("\n1. BASELINE (No optimization)...")
    baseline = run_eval(messages, "audit")
    print(f"   Tokens: {baseline.tokens_input:,} in / {baseline.tokens_output:,} out")
    print(f"   Cost: ${baseline.cost_estimate:.6f} | Latency: {baseline.latency_ms:.0f}ms")

    print("\n2. DEFAULT OPTIMIZATION...")
    default_opt = run_eval(messages, "optimize")
    print(f"   Tokens: {default_opt.tokens_input:,} in / {default_opt.tokens_output:,} out")
    print(f"   Cost: ${default_opt.cost_estimate:.6f} | Latency: {default_opt.latency_ms:.0f}ms")

    print("\n3. AGGRESSIVE OPTIMIZATION...")
    aggressive_opt = run_eval(messages, "optimize", use_aggressive=True)
    print(f"   Tokens: {aggressive_opt.tokens_input:,} in / {aggressive_opt.tokens_output:,} out")
    print(
        f"   Cost: ${aggressive_opt.cost_estimate:.6f} | Latency: {aggressive_opt.latency_ms:.0f}ms"
    )

    # Results table
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    def_savings = baseline.tokens_input - default_opt.tokens_input
    def_pct = (def_savings / baseline.tokens_input * 100) if baseline.tokens_input else 0
    agg_savings = baseline.tokens_input - aggressive_opt.tokens_input
    agg_pct = (agg_savings / baseline.tokens_input * 100) if baseline.tokens_input else 0

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Default Opt':>12} {'Aggressive':>12}")
    print("-" * 60)
    print(
        f"{'Input Tokens':<20} {baseline.tokens_input:>12,} {default_opt.tokens_input:>12,} {aggressive_opt.tokens_input:>12,}"
    )
    print(
        f"{'Output Tokens':<20} {baseline.tokens_output:>12,} {default_opt.tokens_output:>12,} {aggressive_opt.tokens_output:>12,}"
    )
    print(
        f"{'Cost':<20} ${baseline.cost_estimate:>11.6f} ${default_opt.cost_estimate:>11.6f} ${aggressive_opt.cost_estimate:>11.6f}"
    )
    print(
        f"{'Latency (ms)':<20} {baseline.latency_ms:>12.0f} {default_opt.latency_ms:>12.0f} {aggressive_opt.latency_ms:>12.0f}"
    )
    print()
    print(
        f"{'Token Savings':<20} {'-':>12} {def_savings:>10,} ({def_pct:.0f}%) {agg_savings:>10,} ({agg_pct:.0f}%)"
    )

    # Show responses
    print()
    print("-" * 70)
    print("RESPONSE COMPARISON")
    print("-" * 70)

    print("\n[BASELINE RESPONSE]")
    print(baseline.response[:800])
    print("..." if len(baseline.response) > 800 else "")

    print("\n[AGGRESSIVE OPTIMIZATION RESPONSE]")
    print(aggressive_opt.response[:800])
    print("..." if len(aggressive_opt.response) > 800 else "")

    # Quality Evaluation with LLM Judge
    print()
    print("-" * 70)
    print("QUALITY EVALUATION (GPT-4o as Judge)")
    print("-" * 70)

    print("\nEvaluating DEFAULT optimization vs Baseline...")
    default_eval = evaluate_response_quality(baseline.response, default_opt.response, "default")

    print("\nEvaluating AGGRESSIVE optimization vs Baseline...")
    aggressive_eval = evaluate_response_quality(
        baseline.response, aggressive_opt.response, "aggressive"
    )

    print(f"\n{'Criterion':<20} {'Default':>12} {'Aggressive':>12}")
    print("-" * 46)

    if "error" not in default_eval and "error" not in aggressive_eval:
        for criterion in ["correctness", "completeness", "incident_summary", "actionability"]:
            d_score = default_eval.get(criterion, {}).get("score", "N/A")
            a_score = aggressive_eval.get(criterion, {}).get("score", "N/A")
            print(f"{criterion.replace('_', ' ').title():<20} {d_score:>12}/5 {a_score:>12}/5")

        print("-" * 46)
        d_overall = default_eval.get("overall_score", "N/A")
        a_overall = aggressive_eval.get("overall_score", "N/A")
        print(f"{'OVERALL SCORE':<20} {d_overall:>12}/5 {a_overall:>12}/5")

        d_verdict = default_eval.get("verdict", "N/A")
        a_verdict = aggressive_eval.get("verdict", "N/A")
        print(f"{'VERDICT':<20} {d_verdict:>12} {a_verdict:>12}")

        print("\n[Default Optimization Judge Summary]")
        print(f"  {default_eval.get('summary', 'N/A')}")

        print("\n[Aggressive Optimization Judge Summary]")
        print(f"  {aggressive_eval.get('summary', 'N/A')}")

        # Detailed reasoning
        print("\n[Detailed Evaluation - Aggressive]")
        for criterion in ["correctness", "completeness", "incident_summary", "actionability"]:
            reason = aggressive_eval.get(criterion, {}).get("reason", "N/A")
            print(f"  {criterion.title()}: {reason}")
    else:
        print("Error in evaluation:")
        print(f"  Default: {default_eval}")
        print(f"  Aggressive: {aggressive_eval}")

    # Summary
    cost_save_monthly = (baseline.cost_estimate - aggressive_opt.cost_estimate) * 1000 * 30

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    # Get verdicts for summary
    d_verdict = default_eval.get("verdict", "N/A") if "error" not in default_eval else "ERROR"
    a_verdict = aggressive_eval.get("verdict", "N/A") if "error" not in aggressive_eval else "ERROR"
    d_score = default_eval.get("overall_score", 0) if "error" not in default_eval else 0
    a_score = aggressive_eval.get("overall_score", 0) if "error" not in aggressive_eval else 0

    print(f"""
Real-world SRE incident investigation with 5 tool calls:

Tool Outputs:
  - Metrics query:     ~240 data points (60 per metric x 4 metrics)
  - Log search:        40 log entries with full metadata
  - Service status:    5 services with instance details
  - Deployments:       15 deployment records
  - Runbook:           Structured documentation

Token Savings:
  - Default optimization:    {def_savings:,} tokens saved ({def_pct:.1f}%)
  - Aggressive optimization: {agg_savings:,} tokens saved ({agg_pct:.1f}%)

Quality Evaluation (GPT-4o Judge):
  - Default:    {d_verdict} (Score: {d_score}/5)
  - Aggressive: {a_verdict} (Score: {a_score}/5)

Cost Impact @ 1K requests/day:
  - Monthly savings: ${cost_save_monthly:.2f}

CONCLUSION:
  {"✓ Headroom achieves " + f"{agg_pct:.0f}% token reduction with PASSING quality scores." if a_verdict == "PASS" else "⚠ Aggressive optimization may degrade response quality - use conservative settings."}
  {"  The compressed context maintains semantic equivalence for model reasoning." if a_verdict == "PASS" else ""}
""")


if __name__ == "__main__":
    main()
