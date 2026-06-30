"""Tests for _write_claude_wrap_base_url / _restore_claude_wrap_base_url (issue #951)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def test_write_creates_env_key_in_fresh_file(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_preserves_other_env_keys(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"KEEP": "1", "ANOTHER": "2"}}), encoding="utf-8")
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["KEEP"] == "1"
    assert payload["env"]["ANOTHER"] == "2"
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_returns_none_when_key_absent(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None


def test_write_returns_previous_value_when_key_present(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://old.proxy:9000"}}),
        encoding="utf-8",
    )
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev == "http://old.proxy:9000"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_foundry_mode_sets_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url(
        "http://127.0.0.1:8787", foundry_mode=True, settings_path=path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:8787"
    assert "ANTHROPIC_BASE_URL" not in payload["env"]


def test_restore_removes_key_when_previous_none(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    # file is deleted when payload becomes empty — key is gone
    assert not path.exists()


def test_restore_removes_env_dict_when_empty(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    # entire payload was {"env": {...only our key...}} — file deleted rather than left as {}
    assert not path.exists()


def test_restore_preserves_sibling_env_keys(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787", "KEEP": "1"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in payload["env"]
    assert payload["env"]["KEEP"] == "1"


def test_restore_sets_key_back_to_previous_value(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url("http://old.proxy:9000", settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://old.proxy:9000"


def test_restore_foundry_mode_removes_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_FOUNDRY_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, foundry_mode=True, settings_path=path)
    # file deleted when payload empties
    assert not path.exists()


def test_restore_noop_when_file_absent(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_key_not_present(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"OTHER": "1"}}), encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # key absent — no-op
    assert json.loads(path.read_text())["env"]["OTHER"] == "1"


def test_restore_noop_when_env_not_dict(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": "not-a-dict"}), encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_payload_not_dict(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON but not a dict
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_file_corrupt(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {{{{", encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_write_recovers_from_corrupt_file(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {{{{", encoding="utf-8")
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None  # treated as fresh
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_recovers_from_non_dict_payload(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON but not a dict
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_restore_roundtrip(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "opus", "env": {"OTHER": "x"}}), encoding="utf-8")
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert payload["model"] == "opus"

    wrap_cli._restore_claude_wrap_base_url(prev, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in payload.get("env", {})
    assert payload["env"]["OTHER"] == "x"
    assert payload["model"] == "opus"


def test_claude_project_settings_enabled_respects_flag_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_CLAUDE_PROJECT_SETTINGS", raising=False)
    assert wrap_cli._claude_project_settings_enabled(False) is False
    assert wrap_cli._claude_project_settings_enabled(True) is True

    monkeypatch.setenv("HEADROOM_CLAUDE_PROJECT_SETTINGS", "1")
    assert wrap_cli._claude_project_settings_enabled(False) is True


def test_wrap_claude_skips_project_settings_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        settings_path = Path(".claude") / "settings.local.json"
        settings_path.parent.mkdir()
        original = {
            "env": {
                "ANTHROPIC_BASE_URL": "http://direct.example",
                "KEEP": "1",
            }
        }
        settings_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

        with (
            patch("headroom.cli.wrap.shutil.which", return_value="claude"),
            patch("headroom.cli.wrap._ensure_proxy", return_value=None),
            patch("headroom.cli.wrap._setup_rtk", return_value=None),
            patch("headroom.cli.wrap._setup_headroom_mcp", return_value=None),
            patch("headroom.cli.wrap._setup_coding_compressor", return_value=None),
            patch("headroom.cli.wrap._write_claude_wrap_base_url") as write_mock,
            patch("headroom.cli.wrap._restore_claude_wrap_base_url") as restore_mock,
            patch("headroom.cli.wrap.subprocess.run", return_value=completed) as run_mock,
        ):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "claude",
                    "--no-context-tool",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

        assert result.exit_code == 0, result.output
        assert json.loads(settings_path.read_text(encoding="utf-8")) == original
        write_mock.assert_not_called()
        restore_mock.assert_not_called()
        launched_env = run_mock.call_args.kwargs["env"]
        assert launched_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_wrap_claude_project_settings_flag_writes_and_restores(tmp_path: Path) -> None:
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with (
            patch("headroom.cli.wrap.shutil.which", return_value="claude"),
            patch("headroom.cli.wrap._ensure_proxy", return_value=None),
            patch("headroom.cli.wrap._setup_rtk", return_value=None),
            patch("headroom.cli.wrap._setup_headroom_mcp", return_value=None),
            patch("headroom.cli.wrap._setup_coding_compressor", return_value=None),
            patch("headroom.cli.wrap._write_claude_wrap_base_url", return_value="old") as write_mock,
            patch("headroom.cli.wrap._restore_claude_wrap_base_url") as restore_mock,
            patch("headroom.cli.wrap.subprocess.run", return_value=completed),
        ):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "claude",
                    "--project-settings",
                    "--no-context-tool",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

        assert result.exit_code == 0, result.output
        write_mock.assert_called_once_with("http://127.0.0.1:8787", foundry_mode=False)
        restore_mock.assert_called_once_with("old", foundry_mode=False)
