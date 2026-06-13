"""Proxy-runtime subprocess calls must decode UTF-8 explicitly.

On Windows, ``subprocess.run(text=True)`` without an ``encoding`` decodes
child output with the console code page (cp1252). rtk's emoji-laden output
then kills the reader thread with ``UnicodeDecodeError: 'charmap' codec
can't decode byte ...`` (seen in user proxy logs). These tests pin the
``encoding="utf-8"`` kwarg on every proxy-runtime subprocess call.
"""

import subprocess

import headroom.lean_ctx
import headroom.proxy.helpers as helpers
import headroom.rtk
from headroom.proxy.interceptors import astgrep


def _capture_run(captured, returncode=0, stdout='{"summary": {}}'):
    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return fake_run


def test_rtk_stats_subprocess_uses_utf8(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(helpers.subprocess, "run", _capture_run(captured))
    monkeypatch.setattr(headroom.rtk, "get_rtk_path", lambda: "/fake/rtk")

    helpers._read_rtk_lifetime_stats()

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_lean_ctx_stats_subprocess_uses_utf8(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(helpers.subprocess, "run", _capture_run(captured))
    monkeypatch.setattr(headroom.lean_ctx, "get_lean_ctx_path", lambda: "/fake/lean-ctx")

    helpers._read_lean_ctx_lifetime_stats()

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_ast_grep_subprocess_uses_utf8(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(astgrep.subprocess, "run", _capture_run(captured, returncode=1, stdout=""))

    astgrep._run_ast_grep("/fake/sg", "python", "def foo():\n    pass\n")

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
