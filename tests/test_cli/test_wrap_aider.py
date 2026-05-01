"""Tests for `headroom wrap aider` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_aider_sets_provider_envs(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="aider"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "aider", "--no-rtk", "--", "--model", "gpt-4o"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8787/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert captured["tool_label"] == "AIDER"
    assert captured["agent_type"] == "aider"
    assert captured["args"] == ("--model", "gpt-4o")


def test_wrap_aider_reports_missing_binary(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "aider", "--no-rtk"])

    assert result.exit_code == 1
    assert "Error: 'aider' not found in PATH." in result.output
