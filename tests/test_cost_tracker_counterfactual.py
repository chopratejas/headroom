"""Tests for CostTracker compression counterfactual (avg $/input token).

Unit tests use LiteLLM pricing only (no network).
Optional live test calls Anthropic once when ANTHROPIC_API_KEY is set (loads .env).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env for live test (same pattern as test_compression_summary_integration.py)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


pytest.importorskip("litellm")


def test_savings_is_saved_times_average_effective_price_per_token():
    """savings_usd = total_tokens_saved * (cost_with / billed_input_tokens)."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    model = "claude-sonnet-4-20250514"
    # Heavy cache reads + some uncached — average $/token should be well below list price
    ct.record_tokens(
        model,
        tokens_saved=100_000,
        tokens_sent=50_000,
        cache_read_tokens=900_000,
        cache_write_tokens=0,
        uncached_tokens=50_000,
    )
    stats = ct.stats()
    cost_with = stats["cost_with_headroom_usd"]
    billed = 900_000 + 50_000
    avg = cost_with / billed
    expected_savings = 100_000 * avg

    assert stats["total_tokens_saved"] == 100_000
    assert abs(stats["savings_usd"] - expected_savings) < 0.01
    assert abs(stats["cost_without_headroom_usd"] - (cost_with + expected_savings)) < 0.01


def test_fallback_list_price_when_no_billed_tokens():
    """If API never reported billable breakdown, fall back to list price on saved."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    model = "claude-sonnet-4-20250514"
    ct.record_tokens(
        model,
        tokens_saved=10_000,
        tokens_sent=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        uncached_tokens=0,
    )
    stats = ct.stats()
    # No billed tokens → fallback path uses saved * uncached list price
    assert stats["cost_with_headroom_usd"] == 0.0
    assert stats["savings_usd"] > 0
    assert stats["cost_without_headroom_usd"] == stats["savings_usd"]


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic test",
)
def test_live_anthropic_usage_roundtrip_with_cost_tracker():
    """One real Messages call; feed usage into CostTracker and verify accounting."""
    import httpx

    from headroom.proxy.server import CostTracker

    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = "claude-sonnet-4-20250514"
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    assert "usage" in data, "Anthropic response should include usage"
    usage = data.get("usage") or {}
    cr = int(usage.get("cache_read_input_tokens") or 0)
    cw_tok = int(usage.get("cache_creation_input_tokens") or 0)
    unc = int(usage.get("input_tokens") or 0)

    saved = 25_000
    sent = 12_000
    ct = CostTracker()
    ct.record_tokens(
        model,
        tokens_saved=saved,
        tokens_sent=sent,
        cache_read_tokens=cr,
        cache_write_tokens=cw_tok,
        uncached_tokens=unc,
    )
    stats = ct.stats()
    cost_with = stats["cost_with_headroom_usd"]
    cost_without = stats["cost_without_headroom_usd"]
    savings = stats["savings_usd"]
    assert abs((cost_without - cost_with) - savings) < 0.001

    # Average-price path: billed input tokens got a non-zero LiteLLM cost
    billed = cr + cw_tok + unc if (cr + cw_tok + unc) > 0 else sent
    if cost_with > 1e-6 and billed > 0:
        expected = saved * (cost_with / billed)
        assert abs(savings - expected) < 0.05
    else:
        # Fallback path (no billable token cost computed) — still self-consistent above
        assert savings >= 0
