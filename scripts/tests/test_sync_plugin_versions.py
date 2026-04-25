"""Tests for sync-plugin-versions.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).parent.parent / "sync-plugin-versions.py"
    spec = importlib.util.spec_from_file_location("sync_plugin_versions", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_repo_semver_uses_release_helpers(monkeypatch) -> None:
    module = _load_module()
    calls: dict[str, object] = {}

    monkeypatch.setattr(module, "list_release_tags", lambda root: ["v0.9.0"])
    monkeypatch.setattr(module, "find_latest_release_tag", lambda tags: "v0.9.0")
    monkeypatch.setattr(module, "list_release_commits", lambda root, tag: ["feat: add init"])
    monkeypatch.setattr(module, "determine_bump_level", lambda commits: "minor")
    monkeypatch.setattr(module, "get_canonical_version", lambda root: "0.5.25")

    def fake_compute_release_version(*, canonical_version: str, level: str, tags: list[str]):
        calls["canonical_version"] = canonical_version
        calls["level"] = level
        calls["tags"] = tags
        return type("Info", (), {"npm_version": "0.10.0"})()

    monkeypatch.setattr(module, "compute_release_version", fake_compute_release_version)

    assert module.compute_repo_semver(Path("repo")) == "0.10.0"
    assert calls == {
        "canonical_version": "0.5.25",
        "level": "minor",
        "tags": ["v0.9.0"],
    }


def test_main_runs_plugin_only_version_sync(monkeypatch) -> None:
    module = _load_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(module, "compute_repo_semver", lambda root: "0.10.0")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, cwd, check: commands.append(command),
    )

    module.main()

    assert commands == [
        [
            module.sys.executable,
            str(module.ROOT / "scripts" / "version-sync.py"),
            "--root",
            str(module.ROOT),
            "--version",
            "0.10.0",
            "--plugin-manifests-only",
        ]
    ]
