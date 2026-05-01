from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from headroom.subscription import session_tracking
from headroom.subscription.models import WindowTokens


def test_claude_config_dir_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert session_tracking._claude_config_dir() == tmp_path


def test_get_model_weight_matches_expected_families() -> None:
    assert session_tracking.get_model_weight("claude-3-opus-20240229") == 2.0
    assert session_tracking.get_model_weight("claude-3.7-sonnet") == 1.0
    assert session_tracking.get_model_weight("haiku-v1") == 0.5
    assert (
        session_tracking.get_model_weight("octopus-model") == session_tracking.DEFAULT_MODEL_WEIGHT
    )


def test_find_transcript_files_recurses_under_projects(monkeypatch, tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    nested = projects / "team-a" / "run-1"
    nested.mkdir(parents=True)
    (projects / "root.jsonl").write_text("{}", encoding="utf-8")
    (nested / "nested.jsonl").write_text("{}", encoding="utf-8")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    found = sorted(path.relative_to(projects) for path in session_tracking.find_transcript_files())

    assert found == [Path("root.jsonl"), Path("team-a") / "run-1" / "nested.jsonl"]


def test_walk_jsonl_handles_directory_and_entry_errors(tmp_path: Path) -> None:
    results: list[Path] = []

    class BrokenDirectory:
        def iterdir(self):
            raise PermissionError("blocked")

    session_tracking._walk_jsonl(BrokenDirectory(), results)
    assert results == []

    class BrokenEntry:
        suffix = ".jsonl"

        def is_dir(self) -> bool:
            raise OSError("bad entry")

    class MixedDirectory:
        def __init__(self, file_entry: Path) -> None:
            self._file_entry = file_entry

        def iterdir(self):
            return [BrokenEntry(), self._file_entry]

    transcript = tmp_path / "good.jsonl"
    transcript.write_text("{}", encoding="utf-8")
    session_tracking._walk_jsonl(MixedDirectory(transcript), results)

    assert results == [transcript]


def test_read_transcript_lines_filters_blank_lines_and_handles_missing_files(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "sample.jsonl"
    transcript.write_bytes(b'{"ok": 1}\n\n \n{"ok": 2}\n')

    assert session_tracking._read_transcript_lines(transcript) == ['{"ok": 1}', '{"ok": 2}']
    assert session_tracking._read_transcript_lines(tmp_path / "missing.jsonl") == []


def test_add_usage_to_tokens_tracks_nested_and_explicit_cache_writes() -> None:
    totals = WindowTokens()
    session_tracking._add_usage_to_tokens(
        totals,
        {
            "input_tokens": "2",
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 5,
                "ephemeral_1h_input_tokens": "6",
            },
        },
    )
    session_tracking._add_usage_to_tokens(
        totals,
        {
            "cache_creation_input_tokens": 9,
        },
    )

    assert totals.input == 2
    assert totals.output == 3
    assert totals.cache_reads == 4
    assert totals.cache_writes_5m == 5
    assert totals.cache_writes_1h == 6
    assert totals.cache_writes_total == 20


def test_compute_window_tokens_aggregates_models_and_unattributed(
    monkeypatch, tmp_path: Path
) -> None:
    transcript = tmp_path / "window.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-24T12:30:00Z",
                        "message": {
                            "model": "claude-3-opus-20240229",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "cache_read_input_tokens": 4,
                                "cache_creation": {
                                    "ephemeral_5m_input_tokens": 2,
                                    "ephemeral_1h_input_tokens": 3,
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-24T12:45:00Z",
                        "message": {
                            "model": "claude-3-5-sonnet",
                            "usage": {
                                "input_tokens": 7,
                                "output_tokens": 1,
                                "cache_creation_input_tokens": 6,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-24T12:50:00Z",
                        "message": {
                            "usage": {
                                "input_tokens": 3,
                                "output_tokens": 2,
                            }
                        },
                    }
                ),
                '{"invalid-json"',
                json.dumps({"timestamp": "2026-04-24T12:55:00Z", "message": {}}),
                json.dumps(
                    {
                        "timestamp": "2026-04-24T10:59:59Z",
                        "message": {"usage": {"input_tokens": 100}},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(session_tracking, "find_transcript_files", lambda: [transcript])
    start_ts = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc).timestamp()
    end_ts = datetime(2026, 4, 24, 13, 0, tzinfo=timezone.utc).timestamp()

    totals = session_tracking.compute_window_tokens(start_ts=start_ts, end_ts=end_ts)

    assert totals.input == 20
    assert totals.output == 8
    assert totals.cache_reads == 4
    assert totals.cache_writes_5m == 2
    assert totals.cache_writes_1h == 3
    assert totals.cache_writes_total == 11
    assert totals.by_model == {
        "claude-3-opus-20240229": {
            "input": 10,
            "output": 5,
            "cache_reads": 4,
            "cache_writes_5m": 2,
            "cache_writes_1h": 3,
            "cache_writes_total": 5,
        },
        "claude-3-5-sonnet": {
            "input": 7,
            "output": 1,
            "cache_reads": 0,
            "cache_writes_5m": 0,
            "cache_writes_1h": 0,
            "cache_writes_total": 6,
        },
    }
    assert totals.weighted_token_equivalent == 67.0
