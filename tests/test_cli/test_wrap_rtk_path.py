"""Tests for making Headroom-managed RTK available to wrapped agents.

RTK's generated Claude hook and ``rtk rewrite`` output intentionally invoke a
bare ``rtk`` command. Headroom should therefore make the managed binary's
directory available on the wrapped process ``PATH`` instead of mutating RTK's
canonical hook script, which would fail RTK's integrity check (#1669).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from headroom.cli import wrap as wrap_mod
from headroom.cli.wrap import _append_path_entry, _ensure_rtk_on_path, _setup_rtk


def test_ensure_rtk_on_path_appends_managed_rtk_dir(tmp_path: Path) -> None:
    rtk_path = tmp_path / ".headroom" / "bin" / "rtk"
    other_bin = tmp_path / "other-bin"
    other_bin.mkdir()
    env = {"PATH": str(other_bin)}

    changed = _ensure_rtk_on_path(env, rtk_path)

    assert changed is True
    assert env["PATH"].split(os.pathsep) == [str(other_bin), str(rtk_path.parent)]


def test_ensure_rtk_on_path_sets_empty_path(tmp_path: Path) -> None:
    rtk_path = tmp_path / ".headroom" / "bin" / "rtk"
    env: dict[str, str] = {}

    changed = _ensure_rtk_on_path(env, rtk_path)

    assert changed is True
    assert env["PATH"] == str(rtk_path.parent)


def test_append_path_entry_is_idempotent(tmp_path: Path) -> None:
    rtk_dir = tmp_path / ".headroom" / "bin"
    env = {"PATH": os.pathsep.join([str(rtk_dir), "/usr/bin"])}

    changed = _append_path_entry(env, rtk_dir)

    assert changed is False
    assert env["PATH"].split(os.pathsep) == [str(rtk_dir), "/usr/bin"]


def test_ensure_rtk_on_path_preserves_existing_rtk_on_path(tmp_path: Path) -> None:
    managed_rtk = tmp_path / ".headroom" / "bin" / "rtk"
    system_bin = tmp_path / "system-bin"
    system_bin.mkdir()
    system_rtk = system_bin / "rtk"
    system_rtk.write_text("#!/bin/sh\n")
    system_rtk.chmod(0o755)
    env = {"PATH": str(system_bin)}

    changed = _ensure_rtk_on_path(env, managed_rtk)

    assert changed is False
    assert env["PATH"] == str(system_bin)


def test_ensure_rtk_on_path_uses_discovered_rtk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rtk_path = tmp_path / ".headroom" / "bin" / "rtk"
    other_bin = tmp_path / "other-bin"
    other_bin.mkdir()
    env = {"PATH": str(other_bin)}

    monkeypatch.setattr("headroom.rtk.get_rtk_path", lambda: rtk_path)

    assert _ensure_rtk_on_path(env) is True
    assert env["PATH"].split(os.pathsep) == [str(other_bin), str(rtk_path.parent)]


def test_setup_rtk_leaves_canonical_hook_unmodified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rtk_path = tmp_path / ".headroom" / "bin" / "rtk"
    hook_script = tmp_path / ".claude" / "hooks" / "rtk-rewrite.sh"
    hook_script.parent.mkdir(parents=True)
    original = 'exec rtk rewrite "$@"\n'
    hook_script.write_text(original)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.rtk.get_rtk_path", lambda: rtk_path)
    monkeypatch.setattr("headroom.rtk.installer.register_claude_hooks", lambda _rtk: True)

    assert _setup_rtk() == rtk_path
    assert hook_script.read_text() == original


def _invoke_launch_tool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str],
    rtk_path: Path | None,
) -> dict[str, str]:
    captured: dict[str, str] = {}

    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda _holder, _port: lambda *_: None)
    monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda _port: None)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wrap_mod, "_push_runtime_env", lambda _port, _no_proxy: None)
    monkeypatch.setattr(wrap_mod, "_print_telemetry_notice", lambda: None)

    def fake_run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        captured.update(env)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(wrap_mod.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        wrap_mod._launch_tool(
            binary="agent",
            args=(),
            env=env,
            port=8787,
            no_proxy=False,
            tool_label="AGENT",
            env_vars_display=[],
            rtk_path=rtk_path,
        )

    assert exc_info.value.code == 0
    return captured


def test_launch_tool_leaves_path_unchanged_without_rtk_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other_bin = tmp_path / "other-bin"
    env = {"PATH": str(other_bin)}

    def fail_ensure_rtk_on_path(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("_launch_tool should not patch PATH without RTK setup")

    monkeypatch.setattr(wrap_mod, "_ensure_rtk_on_path", fail_ensure_rtk_on_path)

    captured = _invoke_launch_tool(monkeypatch, env=env, rtk_path=None)

    assert captured["PATH"] == str(other_bin)


def test_launch_tool_adds_selected_rtk_dir_when_context_uses_rtk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other_bin = tmp_path / "other-bin"
    rtk_path = tmp_path / ".headroom" / "bin" / "rtk"
    env = {"PATH": str(other_bin)}

    captured = _invoke_launch_tool(monkeypatch, env=env, rtk_path=rtk_path)

    assert captured["PATH"].split(os.pathsep) == [str(other_bin), str(rtk_path.parent)]
