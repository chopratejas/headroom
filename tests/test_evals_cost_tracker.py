from __future__ import annotations

from headroom.evals.cost_tracker import CostTracker, UsageRecord


def test_cost_tracker_records_budget_and_summary(capsys) -> None:
    tracker = CostTracker(budget_usd=1.0)
    record = tracker.record("gsm8k", "gpt-4o-mini", input_tokens=500_000, output_tokens=100_000)

    assert isinstance(record, UsageRecord)
    assert round(record.cost_usd, 4) == 0.135
    assert round(tracker.spent_usd, 4) == 0.135
    assert round(tracker.remaining_usd, 4) == 0.865
    assert tracker.check_budget() is True

    tracker.record("humaneval", "unknown-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert tracker._get_pricing("unknown-model")["input"] == 1.0
    assert tracker.check_budget() is False

    summary = tracker.summary()
    assert summary["n_calls"] == 2
    assert summary["total_input_tokens"] == 1_500_000
    assert summary["total_output_tokens"] == 1_100_000
    assert summary["by_benchmark"]["gsm8k"] == 0.135
    assert summary["by_benchmark"]["humaneval"] == 6.0

    tracker.print_summary()
    printed = capsys.readouterr().out
    assert "Cost Summary" in printed
    assert "humaneval" in printed


def test_cost_tracker_estimation_and_affordability() -> None:
    tracker = CostTracker(budget_usd=0.01)
    estimated = tracker.estimate_cost(
        "claude-sonnet-4-20250514",
        n_samples=10,
        avg_input_tokens=1_000,
        avg_output_tokens=200,
        multiplier=1,
    )
    assert round(estimated, 4) == 0.06
    assert tracker.can_afford("claude-sonnet-4-20250514", 10, 1_000, 200, 1) is False
    assert tracker.can_afford("gpt-4o-mini", 1, 100, 20, 1) is True
