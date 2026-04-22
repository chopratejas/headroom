"""Tests for `headroom wrap copilot` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main
from headroom.copilot_auth import DEFAULT_API_URL


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def no_running_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda _port: False)


def test_wrap_copilot_auto_anthropic_injects_instructions(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
                result = runner.invoke(
                    main,
                    ["wrap", "copilot", "--", "--model", "claude-sonnet-4-20250514"],
                )

    assert result.exit_code == 0, result.output
    instructions = tmp_path / ".github" / "copilot-instructions.md"
    assert instructions.exists()
    content = instructions.read_text()
    assert wrap_cli._RTK_MARKER in content
    assert "RTK (Rust Token Killer)" in content

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787"
    assert "COPILOT_PROVIDER_WIRE_API" not in env
    assert captured["agent_type"] == "copilot"
    assert captured["tool_label"] == "COPILOT"
    assert captured["args"] == ("--model", "claude-sonnet-4-20250514")


def test_wrap_copilot_openai_backend_sets_completions_env(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "copilot",
                    "--no-rtk",
                    "--backend",
                    "anyllm",
                    "--anyllm-provider",
                    "groq",
                    "--region",
                    "us-central1",
                    "--",
                    "--model",
                    "gpt-4o",
                ],
            )

    assert result.exit_code == 0, result.output

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert captured["backend"] == "anyllm"
    assert captured["anyllm_provider"] == "groq"
    assert captured["region"] == "us-central1"
    assert captured["args"] == ("--model", "gpt-4o")


def test_wrap_copilot_auto_detects_running_proxy_backend(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap._check_proxy", return_value=True):
            with patch("headroom.cli.wrap._detect_running_proxy_backend", return_value="anyllm"):
                with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
                    result = runner.invoke(
                        main,
                        ["wrap", "copilot", "--no-rtk", "--", "--model", "gpt-4o"],
                    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"


def test_wrap_copilot_prefers_existing_oauth_session(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap.resolve_client_bearer_token", return_value="gho-existing"):
            with patch("headroom.cli.wrap.has_oauth_auth", return_value=True):
                with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
                    result = runner.invoke(
                        main,
                        ["wrap", "copilot", "--no-rtk", "--", "--model", "claude-sonnet-4.6"],
                    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-existing"
    assert "COPILOT_PROVIDER_API_KEY" not in env
    assert captured["openai_api_url"] == DEFAULT_API_URL


def test_wrap_copilot_translated_backend_still_requires_byok(
    runner: CliRunner,
) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap.has_oauth_auth", return_value=True):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "copilot",
                    "--backend",
                    "anyllm",
                    "--",
                    "--model",
                    "gpt-4o",
                ],
            )

    assert result.exit_code == 1
    assert "Copilot BYOK mode requires a provider API key" in result.output


def test_wrap_copilot_rejects_wire_api_for_anthropic_provider(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--wire-api",
                "responses",
                "--",
                "--model",
                "claude-sonnet-4-20250514",
            ],
        )

    assert result.exit_code != 0
    assert "--wire-api is only valid" in result.output


def test_wrap_copilot_rejects_responses_for_translated_backends(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--backend",
                "anyllm",
                "--wire-api",
                "responses",
                "--",
                "--model",
                "gpt-4o",
            ],
        )

    assert result.exit_code != 0
    assert "not supported with translated backends" in result.output


def test_wrap_copilot_clears_stale_wire_api_in_anthropic_mode(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                ["wrap", "copilot", "--no-rtk", "--", "--model", "claude-sonnet-4-20250514"],
                env={
                    "COPILOT_PROVIDER_WIRE_API": "responses",
                    "ANTHROPIC_API_KEY": "sk-test-dummy",
                },
            )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert "COPILOT_PROVIDER_WIRE_API" not in env


def test_wrap_copilot_fails_when_binary_missing(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "copilot", "--", "--model", "gpt-4o"])

    assert result.exit_code == 1
    assert "'copilot' not found in PATH" in result.output
    assert "Install GitHub Copilot CLI" in result.output
