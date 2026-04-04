"""Tests for `headroom wrap openclaw` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a minimal OpenClaw plugin directory fixture."""
    plugin = tmp_path / "plugins" / "openclaw"
    plugin.mkdir(parents=True)
    (plugin / "package.json").write_text('{"name":"headroom-openclaw"}\n')
    (plugin / "openclaw.plugin.json").write_text('{"id":"headroom"}\n')
    return plugin


def _make_successful_run(calls: list[dict]) -> object:
    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        return MagicMock(returncode=0, stdout="", stderr="")

    return run


def test_wrap_openclaw_default_installs_from_npm_and_restarts(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(main, ["wrap", "openclaw"])

    assert result.exit_code == 0, result.output

    cmds = [c["cmd"] for c in calls]
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        "headroom-ai/openclaw",
    ] in cmds
    assert ["openclaw", "config", "validate"] in cmds
    assert ["openclaw", "gateway", "restart"] in cmds
    assert ["openclaw", "plugins", "inspect", "headroom"] in cmds

    # Verify plugin install in npm mode does not set cwd
    install_call = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "plugins", "install", "--dangerously-force-unsafe-install"]
    )
    assert install_call["cwd"] is None

    # No local build in npm mode
    assert ["npm", "install"] not in cmds
    assert ["npm", "run", "build"] not in cmds

    # Verify config payload includes enabled + expected defaults
    set_entry = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    payload = json.loads(set_entry["cmd"][4])
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["autoStart"] is True
    assert payload["config"]["startupTimeoutMs"] == 20000


def test_wrap_openclaw_skip_build_and_no_restart(runner: CliRunner, plugin_dir: Path) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--skip-build",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] not in cmds
    assert ["npm", "run", "build"] not in cmds
    assert ["openclaw", "gateway", "restart"] not in cmds


def test_wrap_openclaw_local_source_mode_builds_and_links(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                ["wrap", "openclaw", "--plugin-path", str(plugin_dir)],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] in cmds
    assert ["npm", "run", "build"] in cmds
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        "--link",
        ".",
    ] in cmds


def test_wrap_openclaw_fails_when_openclaw_missing(runner: CliRunner, plugin_dir: Path) -> None:
    def which(name: str) -> str | None:
        return None if name == "openclaw" else "npm"

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code != 0
    assert "'openclaw' not found in PATH" in result.output


def test_wrap_openclaw_fails_when_plugin_path_invalid(runner: CliRunner, tmp_path: Path) -> None:
    invalid = tmp_path / "missing-plugin"

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(invalid)])

    assert result.exit_code != 0
    assert "Plugin path not found" in result.output


def test_wrap_openclaw_uses_extension_fallback_on_linked_install_bug(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="Also not a valid hook pack",
                stderr='Plugin installation blocked despite "--dangerously-force-unsafe-install"',
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            with patch(
                "headroom.cli.wrap._copy_openclaw_plugin_into_extensions",
                return_value=Path("C:/Users/test/.openclaw/extensions/headroom"),
            ) as copy_fallback:
                result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code == 0, result.output
    copy_fallback.assert_called_once()


def test_wrap_openclaw_continues_when_plugin_already_exists(
    runner: CliRunner,
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="plugin already exists: C:\\Users\\test\\.openclaw\\extensions\\headroom",
                stderr="",
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw", "--no-restart"])

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["openclaw", "config", "validate"] in cmds
    assert ["openclaw", "plugins", "inspect", "headroom"] in cmds


def test_wrap_openclaw_verbose_prints_install_restart_and_inspect_output(
    runner: CliRunner,
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(returncode=0, stdout="install-ok", stderr="")
        if cmd[:3] == ["openclaw", "gateway", "restart"]:
            return MagicMock(returncode=0, stdout="restart-ok", stderr="")
        if cmd[:3] == ["openclaw", "plugins", "inspect"]:
            return MagicMock(returncode=0, stdout="inspect-ok", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw", "--verbose"])

    assert result.exit_code == 0, result.output
    assert "install-ok" in result.output
    assert "restart-ok" in result.output
    assert "inspect-ok" in result.output


def test_wrap_openclaw_fails_for_npm_mode_hook_pack_bug_without_local_fallback(
    runner: CliRunner,
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="Also not a valid hook pack",
                stderr='Blocked despite "--dangerously-force-unsafe-install"',
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw"])

    assert result.exit_code != 0
    assert "openclaw plugins install failed" in result.output


def test_wrap_openclaw_copy_mode_uses_path_install(runner: CliRunner, plugin_dir: Path) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--copy",
                    "--skip-build",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        str(plugin_dir),
    ] in cmds


def test_wrap_openclaw_fails_when_npm_missing_for_local_build(
    runner: CliRunner, plugin_dir: Path
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": None,
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code != 0
    assert "'npm' not found in PATH" in result.output


def test_wrap_openclaw_fails_when_local_path_missing_manifest_files(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugins" / "openclaw"
    plugin.mkdir(parents=True)

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin)])
    assert result.exit_code != 0
    assert "missing package.json" in result.output

    (plugin / "package.json").write_text("{}\n")
    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin)])
    assert result.exit_code != 0
    assert "missing openclaw.plugin.json" in result.output


def test_run_checked_raises_click_exception_on_command_errors() -> None:
    with patch("headroom.cli.wrap.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(Exception, match="command not found"):
            wrap_cli._run_checked(["missing"], action="demo")

    cpe_stderr = wrap_cli.subprocess.CalledProcessError(
        returncode=2,
        cmd=["x"],
        stderr="bad-stderr",
    )
    with patch("headroom.cli.wrap.subprocess.run", side_effect=cpe_stderr):
        with pytest.raises(Exception, match="bad-stderr"):
            wrap_cli._run_checked(["x"], action="demo")

    cpe_stdout = wrap_cli.subprocess.CalledProcessError(
        returncode=3,
        cmd=["x"],
        output="bad-stdout",
        stderr="",
    )
    with patch("headroom.cli.wrap.subprocess.run", side_effect=cpe_stdout):
        with pytest.raises(Exception, match="bad-stdout"):
            wrap_cli._run_checked(["x"], action="demo")


def test_resolve_openclaw_extensions_dir_empty_output_raises() -> None:
    with patch(
        "headroom.cli.wrap._run_checked",
        return_value=MagicMock(stdout="   \n", stderr="", returncode=0),
    ):
        with pytest.raises(Exception, match="Unable to resolve OpenClaw config path"):
            wrap_cli._resolve_openclaw_extensions_dir("openclaw")


def test_copy_openclaw_plugin_into_extensions_handles_missing_and_existing_dist(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()

    with pytest.raises(Exception, match="Plugin dist folder missing"):
        wrap_cli._copy_openclaw_plugin_into_extensions(plugin_dir=plugin, openclaw_bin="openclaw")

    dist = plugin / "dist"
    dist.mkdir()
    (dist / "index.js").write_text("x\n")
    (plugin / "package.json").write_text("{}\n")
    (plugin / "openclaw.plugin.json").write_text("{}\n")

    ext_root = tmp_path / ".openclaw" / "extensions"
    target_headroom = ext_root / "headroom"
    target_dist = target_headroom / "dist"
    target_dist.mkdir(parents=True)
    (target_dist / "old.js").write_text("old\n")

    with patch("headroom.cli.wrap._resolve_openclaw_extensions_dir", return_value=ext_root):
        out = wrap_cli._copy_openclaw_plugin_into_extensions(
            plugin_dir=plugin, openclaw_bin="openclaw"
        )

    assert out == target_headroom
    assert (target_dist / "index.js").exists()
    assert not (target_dist / "old.js").exists()
