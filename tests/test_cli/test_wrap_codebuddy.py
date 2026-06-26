"""Tests for CodeBuddy wrap CLI commands."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _fake_proxy():
    p = MagicMock()
    p.proxy_base_url = "http://127.0.0.1:8787/v2"
    p.pid = 12345
    return p


def _mock_run_success():
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def test_wrap_codebuddy_prepare_only(runner: CliRunner) -> None:
    with (
        patch("headroom.cli.wrap._setup_rtk_for_codebuddy"),
        patch("headroom.cli.wrap._prepare_wrap_rtk"),
        patch("headroom.cli.wrap._setup_lean_ctx_agent"),
        patch("headroom.cli.wrap._ensure_proxy"),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--prepare-only", "--no-mcp", "--no-proxy"],
        )
    assert result.exit_code == 0, result.output


def test_wrap_codebuddy_no_context_tool(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._setup_rtk_for_codebuddy") as setup_rtk,
        patch("headroom.cli.wrap._setup_lean_ctx_agent") as setup_lean,
        patch("headroom.cli.wrap._ensure_proxy", return_value=fake_proxy),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-context-tool", "--no-mcp"],
        )
    assert result.exit_code == 0, result.output
    setup_rtk.assert_not_called()
    setup_lean.assert_not_called()


def test_wrap_codebuddy_with_rtk(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._setup_rtk_for_codebuddy") as setup_rtk,
        patch("headroom.cli.wrap._ensure_proxy", return_value=fake_proxy),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
        patch("headroom.cli.wrap._selected_context_tool", return_value="rtk"),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-mcp"],
        )
    assert result.exit_code == 0, result.output
    setup_rtk.assert_called_once()


def test_wrap_codebuddy_no_mcp(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._setup_rtk_for_codebuddy"),
        patch("headroom.cli.wrap._ensure_proxy", return_value=fake_proxy),
        patch("headroom.cli.wrap._setup_headroom_mcp") as register_mcp,
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
        patch("headroom.cli.wrap._selected_context_tool", return_value="rtk"),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-mcp"],
        )
    assert result.exit_code == 0, result.output
    register_mcp.assert_not_called()


def test_wrap_codebuddy_no_serena(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._setup_rtk_for_codebuddy"),
        patch("headroom.cli.wrap._ensure_proxy", return_value=fake_proxy),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
        patch("headroom.cli.wrap._setup_serena_mcp") as register_serena,
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
        patch("headroom.cli.wrap._selected_context_tool", return_value="rtk"),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-mcp", "--no-serena"],
        )
    assert result.exit_code == 0, result.output
    register_serena.assert_not_called()
