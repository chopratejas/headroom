"""Tests for Claude session mode simulation benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from benchmarks.claude_session_mode_benchmark import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    ModeSummary,
    ReplayTurn,
    SessionReplay,
    _write_checkpoint_by_session_id,
    decode_project_key,
    determine_winners,
    load_session_replay,
    simulate_replays,
    summarize_observed_usage,
)


def test_decode_project_key_windows_path() -> None:
    assert decode_project_key("C--git-BetBlocker") == r"C:\git\BetBlocker"


def test_load_session_replay_groups_assistant_request_events(tmp_path: Path) -> None:
    project_dir = tmp_path / "C--git-BetBlocker"
    project_dir.mkdir()
    session_file = project_dir / "sess-1.jsonl"
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2026-03-13T01:00:00Z",
        },
        {
            "type": "assistant",
            "requestId": "req-1",
            "timestamp": "2026-03-13T01:00:01Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "thinking", "thinking": "..."}],
                "usage": {"output_tokens": 2},
            },
        },
        {
            "type": "assistant",
            "requestId": "req-1",
            "timestamp": "2026-03-13T01:00:02Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"output_tokens": 5},
            },
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Next"},
            "timestamp": "2026-03-13T01:01:00Z",
        },
        {
            "type": "assistant",
            "requestId": "req-2",
            "timestamp": "2026-03-13T01:01:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"output_tokens": 3},
            },
        },
    ]
    session_file.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    replay = load_session_replay(session_file)

    assert replay is not None
    assert len(replay.turns) == 2
    assert replay.turns[0].request_id == "req-1"
    assert replay.turns[0].output_tokens == 5
    assert replay.turns[0].input_messages == [{"role": "user", "content": "Hello"}]
    assert replay.turns[1].input_messages == [{"role": "user", "content": "Next"}]
    assert replay.turns[1].assistant_message["content"] == [{"type": "text", "text": "Done"}]


def test_simulation_and_winner_logic() -> None:
    tool_blob = '{"rows":[1,2,3,4]}' * 80
    turn1 = ReplayTurn(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        request_id="r1",
        model="claude-sonnet-4-6",
        timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
        input_messages=[
            {"role": "user", "content": "Summarize this JSON"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": tool_blob,
                    }
                ],
            },
        ],
        assistant_message={"role": "assistant", "content": "ok"},
        output_tokens=20,
    )
    turn2 = ReplayTurn(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        request_id="r2",
        model="claude-sonnet-4-6",
        timestamp=datetime.fromisoformat("2026-03-13T01:03:00+00:00"),
        input_messages=[
            {"role": "user", "content": "Now tell me the anomalies again"},
        ],
        assistant_message={"role": "assistant", "content": "ok2"},
        output_tokens=25,
    )
    replay = SessionReplay(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        turns=[turn1, turn2],
    )

    dataset, summaries = simulate_replays([replay], cache_ttl_minutes=5)

    assert dataset.requests == 2
    assert summaries["baseline"].raw_input_tokens > 0
    assert (
        summaries[PROXY_MODE_TOKEN].forwarded_input_tokens
        <= summaries["baseline"].forwarded_input_tokens
    )
    assert summaries[PROXY_MODE_CACHE].cache_read_tokens >= 0

    winners = determine_winners(summaries)
    assert winners["total_cost"] in {"baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE}
    assert winners["window_with_cache"] in {"baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE}


def test_observed_usage_summary_tracks_cache_patterns() -> None:
    turns = [
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r1",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
            input_messages=[{"role": "user", "content": "a"}],
            assistant_message={"role": "assistant", "content": "x"},
            output_tokens=5,
            observed_input_tokens=10,
            observed_cache_read_tokens=0,
            observed_cache_write_tokens=100,
        ),
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r2",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:01:00+00:00"),
            input_messages=[{"role": "user", "content": "b"}],
            assistant_message={"role": "assistant", "content": "y"},
            output_tokens=6,
            observed_input_tokens=9,
            observed_cache_read_tokens=80,
            observed_cache_write_tokens=90,
        ),
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r3",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:02:00+00:00"),
            input_messages=[{"role": "user", "content": "c"}],
            assistant_message={"role": "assistant", "content": "z"},
            output_tokens=7,
            observed_input_tokens=9,
            observed_cache_read_tokens=80,
            observed_cache_write_tokens=120,
        ),
    ]
    replay = SessionReplay(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        turns=turns,
    )

    observed = summarize_observed_usage([replay])

    assert observed.requests == 3
    assert observed.cache_read_tokens == 160
    assert observed.cache_write_tokens == 310
    assert observed.healthy_growth_turns == 1
    assert observed.broken_prefix_turns == 2


def test_checkpoint_write_omits_per_turn_payload(tmp_path: Path) -> None:
    summary = ModeSummary(
        mode=PROXY_MODE_TOKEN,
        sessions=1,
        requests=1,
        turns=[],
    )

    _write_checkpoint_by_session_id(tmp_path, PROXY_MODE_TOKEN, "session-1", summary)

    payload = json.loads((tmp_path / f"{PROXY_MODE_TOKEN}--session-1.json").read_text())
    assert payload["turns"] == []
