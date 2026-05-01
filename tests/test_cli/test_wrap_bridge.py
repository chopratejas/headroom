"""Tests for Docker-bridge wrap preparation flows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


def _set_test_home(monkeypatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


def test_wrap_claude_prepare_only_skips_host_binary_lookup() -> None:
    runner = CliRunner()

    with patch("headroom.cli.wrap._prepare_wrap_rtk") as prepare_rtk:
        with patch("headroom.cli.wrap.shutil.which") as which_mock:
            result = runner.invoke(main, ["wrap", "claude", "--prepare-only"])

    assert result.exit_code == 0, result.output
    prepare_rtk.assert_called_once()
    which_mock.assert_not_called()


def test_wrap_codex_prepare_only_updates_config(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()
    assert 'model_provider = "headroom"' in config_file.read_text()
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()


def test_wrap_aider_prepare_only_injects_conventions(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "aider", "--prepare-only"])

        assert result.exit_code == 0, result.output
        conventions = Path("CONVENTIONS.md")
        assert conventions.exists()
        assert "headroom:rtk-instructions" in conventions.read_text()


def test_wrap_cursor_prepare_only_injects_cursorrules(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "cursor", "--prepare-only"])

        assert result.exit_code == 0, result.output
        cursorrules = Path(".cursorrules")
        assert cursorrules.exists()
        assert "headroom:rtk-instructions" in cursorrules.read_text()


def test_wrap_openclaw_prepare_only_emits_config_without_python_default() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "wrap",
            "openclaw",
            "--prepare-only",
            "--gateway-provider-id",
            "codex",
            "--gateway-provider-id",
            "anthropic",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["gatewayProviderIds"] == ["codex", "anthropic"]
    assert "pythonPath" not in payload["config"]


def test_unwrap_openclaw_prepare_only_preserves_unmanaged_config() -> None:
    runner = CliRunner()
    existing_entry = json.dumps(
        {
            "enabled": True,
            "config": {
                "pythonPath": "C:\\Python312\\python.exe",
                "proxyPort": 8787,
                "customFlag": True,
            },
        }
    )

    result = runner.invoke(
        main,
        [
            "unwrap",
            "openclaw",
            "--prepare-only",
            "--existing-entry-json",
            existing_entry,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"enabled": False, "config": {"customFlag": True}}


def test_ensure_rtk_binary_reuses_existing_install() -> None:
    with (
        patch("headroom.rtk.get_rtk_path", return_value=Path("rtk")),
        patch("headroom.cli.wrap.click.echo") as echo,
    ):
        result = wrap_cli._ensure_rtk_binary(verbose=True)

    assert result == Path("rtk")
    echo.assert_any_call("  rtk found at rtk")


def test_ensure_rtk_binary_downloads_when_missing_and_reports_failure() -> None:
    with (
        patch("headroom.rtk.get_rtk_path", return_value=None),
        patch("headroom.rtk.installer.ensure_rtk", side_effect=[Path("downloaded-rtk"), None]),
        patch("headroom.cli.wrap.click.echo") as echo,
    ):
        installed = wrap_cli._ensure_rtk_binary()
        failed = wrap_cli._ensure_rtk_binary()

    assert installed == Path("downloaded-rtk")
    assert failed is None
    echo.assert_any_call("  Downloading rtk (Rust Token Killer)...")
    echo.assert_any_call("  rtk installed at downloaded-rtk")
    echo.assert_any_call("  rtk download failed — continuing without it")


def test_prepare_wrap_rtk_prints_label_before_install() -> None:
    with (
        patch.object(wrap_cli, "_ensure_rtk_binary", return_value=Path("rtk")) as ensure_rtk,
        patch("headroom.cli.wrap.click.echo") as echo,
    ):
        result = wrap_cli._prepare_wrap_rtk(label="Claude")

    assert result == Path("rtk")
    ensure_rtk.assert_called_once_with(verbose=False)
    echo.assert_any_call("  Preparing rtk for Claude...")


def test_wrap_cursor_reports_unexpected_proxy_exit(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    class _ExitedProxy:
        def poll(self) -> int:
            return 1

    cleanup_calls: list[str] = []
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: _ExitedProxy())
    monkeypatch.setattr(wrap_cli, "_render_cursor_setup_lines", lambda port: ["line 1", "line 2"])
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port: lambda signum=None, frame=None: cleanup_calls.append("cleanup"),
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)

    result = runner.invoke(main, ["wrap", "cursor", "--no-rtk"])

    assert result.exit_code == 1
    assert "Proxy process exited unexpectedly." in result.output
    assert cleanup_calls == ["cleanup"]


def test_wrap_claude_with_memory_sync_launches_and_reports_sync_stats(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "claude-user")
    calls: list[dict[str, object]] = []
    cleanup_calls: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[0] == wrap_cli.sys.executable:
            return SimpleNamespace(
                returncode=0,
                stdout='ignored\n{"imported":1,"exported":2,"ms":33}\n',
                stderr="",
            )
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(
        wrap_cli.shutil, "which", lambda name: "claude" if name == "claude" else None
    )
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port: lambda signum=None, frame=None: cleanup_calls.append("cleanup"),
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(wrap_cli, "_print_telemetry_notice", lambda: None)

    result = runner.invoke(
        main,
        ["wrap", "claude", "--memory", "--no-rtk", "-v", "--", "--resume", "session-1"],
    )

    assert result.exit_code == 7, result.output
    assert "Syncing memory (user=claude-user)..." in result.output
    assert "Memory synced: 1 imported, 2 exported (33ms)" in result.output
    assert "Skipping rtk (--no-rtk)" in result.output
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8787" in result.output
    assert "Extra args: --resume session-1" in result.output
    assert calls[0]["cmd"] == [
        wrap_cli.sys.executable,
        "-m",
        "headroom.memory.sync",
        "--db",
        str(tmp_path / ".headroom" / "memory.db"),
        "--user",
        "claude-user",
        "--agent",
        "claude",
        "--force",
    ]
    assert calls[1]["cmd"] == ["claude", "--resume", "session-1"]
    launch_env = calls[1]["env"]
    assert isinstance(launch_env, dict)
    assert launch_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert cleanup_calls == ["cleanup"]


def test_wrap_claude_reports_missing_binary() -> None:
    runner = CliRunner()

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "claude", "--no-rtk"])

    assert result.exit_code == 1
    assert "Error: 'claude' not found in PATH." in result.output


def test_wrap_claude_reports_up_to_date_sync_and_runs_rtk_and_code_graph(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    calls: list[dict[str, object]] = []
    cleanup_calls: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[0] == wrap_cli.sys.executable:
            return SimpleNamespace(
                returncode=0,
                stdout='ignored\n{"imported":0,"exported":0,"ms":12}\n',
                stderr="",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        wrap_cli.shutil, "which", lambda name: "claude" if name == "claude" else None
    )
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port: lambda signum=None, frame=None: cleanup_calls.append("cleanup"),
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(wrap_cli, "_print_telemetry_notice", lambda: None)

    with (
        patch("headroom.cli.wrap._setup_rtk") as setup_rtk,
        patch("headroom.cli.wrap._setup_code_graph") as setup_code_graph,
    ):
        result = runner.invoke(main, ["wrap", "claude", "--memory", "--code-graph"])

    assert result.exit_code == 0, result.output
    assert "Memory: up to date (12ms)" in result.output
    setup_rtk.assert_called_once_with(verbose=False)
    setup_code_graph.assert_called_once_with(verbose=False)
    assert cleanup_calls == ["cleanup"]


def test_wrap_claude_surfaces_sync_stderr_and_launch_errors(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if cmd[0] == wrap_cli.sys.executable:
            return SimpleNamespace(returncode=1, stdout="", stderr="x" * 400)
        raise RuntimeError("launch failed")

    monkeypatch.setattr(
        wrap_cli.shutil, "which", lambda name: "claude" if name == "claude" else None
    )
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port: lambda signum=None, frame=None: None,
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(wrap_cli, "_print_telemetry_notice", lambda: None)
    monkeypatch.setattr(wrap_cli, "_setup_rtk", lambda verbose=False: None)

    result = runner.invoke(main, ["wrap", "claude", "--memory"])

    assert result.exit_code == 1
    assert "Warning: memory sync error:" in result.output
    assert "Error: launch failed" in result.output


def test_wrap_claude_warns_when_memory_sync_fails(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    calls = {"count": 0}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("sync exploded")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        wrap_cli.shutil, "which", lambda name: "claude" if name == "claude" else None
    )
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port: lambda signum=None, frame=None: None,
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(wrap_cli, "_print_telemetry_notice", lambda: None)

    result = runner.invoke(main, ["wrap", "claude", "--memory", "--no-rtk"])

    assert result.exit_code == 0, result.output
    assert "Warning: memory sync failed: sync exploded" in result.output
