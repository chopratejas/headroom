"""Detachment of the shared proxy subprocess in ``headroom wrap``.

``_start_proxy`` launches the proxy that every wrapped agent on a port shares.
It must outlive an *ungraceful* close of the agent that happened to start it
(closing the terminal window, taskkill, a crash) -- otherwise the OS tree-kills
the proxy and breaks the other live clients, bypassing the marker-based
reference counting in ``_make_cleanup``.

On Windows that means detaching from the launcher's console and Job object via
creation flags; on POSIX ``start_new_session`` already detaches via setsid().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from headroom.cli import wrap as wrap_cli

# Windows-only creation flag; not always exported by the host's ``subprocess``.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


class _FakeProc:
    """Stand-in for a live proxy process (``poll() is None``)."""

    returncode = 0

    def poll(self) -> None:
        return None


def _capture_popen_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """Invoke ``_start_proxy`` with all I/O stubbed; return the Popen kwargs."""
    captured: dict[str, Any] = {}

    def _fake_popen(cmd: Any, **kwargs: Any) -> _FakeProc:
        captured.clear()
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(wrap_cli.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_cli, "_resolve_wrap_proxy_timeout_seconds", lambda: 1)

    wrap_cli._start_proxy(8787)
    return captured


def test_start_proxy_detaches_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force the Windows branch regardless of the host OS, and supply the
    # Windows-only ``subprocess`` constants the host lacks on POSIX.
    monkeypatch.setattr(wrap_cli.sys, "platform", "win32")
    monkeypatch.setattr(wrap_cli.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(
        wrap_cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False
    )

    flags = _capture_popen_kwargs(monkeypatch, tmp_path)["creationflags"]

    assert flags & 0x8  # DETACHED_PROCESS: no shared console
    assert flags & 0x200  # CREATE_NEW_PROCESS_GROUP: ignore the parent's Ctrl-C
    assert flags & _CREATE_BREAKAWAY_FROM_JOB  # survive Job kill-on-close


def test_start_proxy_keeps_creationflags_zero_off_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(wrap_cli.sys, "platform", "linux")

    kwargs = _capture_popen_kwargs(monkeypatch, tmp_path)

    # POSIX detaches via setsid(); no Windows creation flags are applied.
    assert kwargs["creationflags"] == 0
    assert isinstance(kwargs["start_new_session"], bool)
