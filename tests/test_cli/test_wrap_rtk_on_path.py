"""Tests for ``_ensure_rtk_on_path``.

``rtk init --global --auto-patch`` writes ``~/.claude/hooks/rtk-rewrite.sh``,
and ``rtk rewrite`` emits a bare ``rtk`` token at runtime that the hook feeds
back to the shell — so bare ``rtk`` must resolve on PATH. Since
``~/.headroom/bin`` is not on PATH by default, that lookup fails and token
compression never runs (issue #487).

The earlier fix rewrote the generated hook to hard-code rtk's absolute path,
but that mutates the hook after ``rtk init`` bakes in its expected SHA-256, so
rtk's integrity guard rejects it (issue #1631). ``_ensure_rtk_on_path`` instead
leaves the canonical hook untouched and links the managed binary into a PATH
directory so bare ``rtk`` resolves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.cli import wrap
from headroom.cli.wrap import _ensure_rtk_on_path


@pytest.fixture
def rtk_binary(tmp_path: Path) -> Path:
    managed = tmp_path / ".headroom" / "bin" / "rtk"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n")
    managed.chmod(0o755)
    return managed


def test_noop_when_rtk_already_on_path(rtk_binary: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: "/usr/bin/rtk")

    assert _ensure_rtk_on_path(rtk_binary, path_dirs=["/usr/bin"]) is None


def test_noop_on_windows(rtk_binary: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "win32")

    assert _ensure_rtk_on_path(rtk_binary, path_dirs=["C:\\bin"]) is None


def test_links_into_path_dir_when_missing(
    rtk_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: None)
    bindir = tmp_path / "path-bin"
    bindir.mkdir()

    link = _ensure_rtk_on_path(rtk_binary, path_dirs=[str(bindir)])

    assert link == bindir / "rtk"
    assert link.is_symlink()
    assert link.resolve() == rtk_binary.resolve()


def test_prefers_local_bin(
    rtk_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: None)
    home = tmp_path / "home"
    monkeypatch.setattr(wrap.Path, "home", classmethod(lambda _cls: home))
    other = tmp_path / "other-bin"
    other.mkdir()
    local_bin = home / ".local" / "bin"

    # ~/.local/bin does not exist yet but is on PATH — it is created on demand
    # and preferred over the other writable dir.
    link = _ensure_rtk_on_path(rtk_binary, path_dirs=[str(other), str(local_bin)])

    assert link == local_bin / "rtk"
    assert link.is_symlink()


def test_idempotent_second_run(
    rtk_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: None)
    bindir = tmp_path / "path-bin"
    bindir.mkdir()

    first = _ensure_rtk_on_path(rtk_binary, path_dirs=[str(bindir)])
    second = _ensure_rtk_on_path(rtk_binary, path_dirs=[str(bindir)])

    assert first == second == bindir / "rtk"
    assert second.resolve() == rtk_binary.resolve()


def test_does_not_clobber_existing_file(
    rtk_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: None)
    occupied = tmp_path / "occupied-bin"
    occupied.mkdir()
    foreign = occupied / "rtk"
    foreign.write_text("#!/bin/sh\n# a different rtk\n")
    fallback = tmp_path / "fallback-bin"
    fallback.mkdir()

    link = _ensure_rtk_on_path(rtk_binary, path_dirs=[str(occupied), str(fallback)])

    # The real file is left untouched; the link lands in the next writable dir.
    assert foreign.read_text() == "#!/bin/sh\n# a different rtk\n"
    assert not foreign.is_symlink()
    assert link == fallback / "rtk"
    assert link.is_symlink()


def test_noop_when_no_writable_path_dir(
    rtk_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrap.sys, "platform", "linux")
    monkeypatch.setattr(wrap.shutil, "which", lambda _cmd: None)
    home = tmp_path / "home"
    monkeypatch.setattr(wrap.Path, "home", classmethod(lambda _cls: home))

    # Only a non-existent, non-preferred dir on PATH — nothing to link into.
    assert _ensure_rtk_on_path(rtk_binary, path_dirs=[str(tmp_path / "ghost")]) is None
