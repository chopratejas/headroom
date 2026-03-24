"""Tests for _decode_project_path and _greedy_path_decode (issue #47).

Directory names that contain dots (e.g. ``GitHub.nosync``) or multiple
hyphens (e.g. ``my-cool-project``) were silently dropped because
_greedy_path_decode only tried joining two consecutive tokens with a hyphen,
making it impossible to reconstruct names formed from three or more tokens.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from headroom.learn.scanner import _decode_project_path, _greedy_path_decode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dirs(base: Path, *rel_paths: str) -> None:
    """Create one or more relative directory paths under *base*."""
    for rel in rel_paths:
        (base / rel).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# _greedy_path_decode
# ---------------------------------------------------------------------------


class TestGreedyPathDecode:
    """Unit tests for _greedy_path_decode."""

    def test_simple_directory(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path, "headroom")
        result = _greedy_path_decode(tmp_path, ["headroom"])
        assert result == tmp_path / "headroom"

    def test_single_hyphen_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains one literal hyphen."""
        _make_dirs(tmp_path, "my-project")
        result = _greedy_path_decode(tmp_path, ["my", "project"])
        assert result == tmp_path / "my-project"

    def test_multiple_hyphens_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains multiple literal hyphens (the regression case)."""
        _make_dirs(tmp_path, "my-cool-project")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project"])
        assert result == tmp_path / "my-cool-project"

    def test_dot_only_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains a dot but no hyphen (e.g. GitHub.nosync)."""
        _make_dirs(tmp_path, "GitHub.nosync")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync"])
        assert result == tmp_path / "GitHub.nosync"

    def test_dot_and_single_hyphen_in_dirname(self, tmp_path: Path) -> None:
        """Directory name has both a dot and a single hyphen (e.g. my-project.nosync)."""
        _make_dirs(tmp_path, "my-project.nosync")
        result = _greedy_path_decode(tmp_path, ["my", "project.nosync"])
        assert result == tmp_path / "my-project.nosync"

    def test_dot_and_multiple_hyphens_in_dirname(self, tmp_path: Path) -> None:
        """Directory name has a dot and multiple hyphens (e.g. my-cool-project.nosync).

        This was the primary regression: the old code only joined pairs, so it
        could never reconstruct a three-token hyphenated name.
        """
        _make_dirs(tmp_path, "my-cool-project.nosync")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project.nosync"])
        assert result == tmp_path / "my-cool-project.nosync"

    def test_dot_dir_containing_hyphenated_subdir(self, tmp_path: Path) -> None:
        """Path like GitHub.nosync/my-project — dot parent + hyphen child."""
        _make_dirs(tmp_path, "GitHub.nosync/my-project")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync", "my", "project"])
        assert result == tmp_path / "GitHub.nosync" / "my-project"

    def test_dot_dir_with_multi_hyphen_subdir(self, tmp_path: Path) -> None:
        """Path like GitHub.nosync/my-cool-app — dot parent + multi-hyphen child."""
        _make_dirs(tmp_path, "GitHub.nosync/my-cool-app")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync", "my", "cool", "app"])
        assert result == tmp_path / "GitHub.nosync" / "my-cool-app"

    def test_multi_hyphen_dot_dir_containing_subproject(self, tmp_path: Path) -> None:
        """Path like my-cool-project.nosync/headroom — hardest combination."""
        _make_dirs(tmp_path, "my-cool-project.nosync/headroom")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project.nosync", "headroom"])
        assert result == tmp_path / "my-cool-project.nosync" / "headroom"

    def test_dot_flattened_into_separate_tokens(self, tmp_path: Path) -> None:
        """Flattened encoding like GitHub-nosync should map back to GitHub.nosync."""
        _make_dirs(tmp_path, "GitHub.nosync/thebest")
        result = _greedy_path_decode(tmp_path, ["GitHub", "nosync", "thebest"])
        assert result == tmp_path / "GitHub.nosync" / "thebest"

    def test_hybrid_hyphen_and_dot_flattening(self, tmp_path: Path) -> None:
        """Flattened encoding should reconstruct mixed separators in one component."""
        _make_dirs(tmp_path, "my-cool-project.nosync/headroom")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project", "nosync", "headroom"])
        assert result == tmp_path / "my-cool-project.nosync" / "headroom"

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        result = _greedy_path_decode(tmp_path, ["does", "not", "exist"])
        assert result is None

    def test_empty_parts_returns_base_when_exists(self, tmp_path: Path) -> None:
        result = _greedy_path_decode(tmp_path, [])
        assert result == tmp_path

    def test_empty_parts_returns_none_when_not_exists(self) -> None:
        result = _greedy_path_decode(Path("/nonexistent/path"), [])
        assert result is None


# ---------------------------------------------------------------------------
# _decode_project_path
# ---------------------------------------------------------------------------


class TestDecodeProjectPath:
    """Integration-level tests for _decode_project_path.

    Note: _decode_project_path's greedy branch only activates for paths whose
    first component is ``Users`` (the common macOS home prefix).  Tests that
    exercise the greedy decoder therefore synthesise an encoded name rooted at
    ``/Users/<username>/…`` inside a real temporary directory created under
    that prefix.  When the temp directory does not exist under ``/Users`` the
    tests fall back to ``/tmp`` and rely only on the fast simple-replace path.
    """

    def test_returns_none_for_non_absolute_encoded_name(self) -> None:
        assert _decode_project_path("Users-foo-bar") is None

    def test_simple_replace_finds_dot_path(self, users_tmp: Path) -> None:
        """Simple replace-all works when no dir names contain hyphens.

        The encoded name maps directly to the real path because every ``-`` is
        a path separator; dots in directory names are preserved unchanged.
        """
        project = users_tmp / "GitHub.nosync" / "headroom"
        project.mkdir(parents=True)
        # Build the encoded name exactly as Claude Code does (/  →  -)
        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)
        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    # ------------------------------------------------------------------
    # Greedy-decoder tests — require a /Users-rooted path to activate.
    # We try to create a temp dir under the real /Users tree; if that is
    # not writable we skip rather than fail (CI typically runs as a real
    # macOS user whose home IS under /Users).
    # ------------------------------------------------------------------

    @pytest.fixture()
    def users_tmp(self, tmp_path: Path) -> Path:
        """Return a temporary directory whose path starts with /Users/…

        On macOS the system temp dir is under /private/var, so we create a
        disposable directory directly inside the real user's home instead.
        Falls back to tmp_path so tests still run on non-macOS platforms
        (where the greedy branch isn't reached but no crash occurs either).
        """

        home = Path.home()
        if str(home).startswith("/Users/"):
            base = home / ".pytest_headroom_tmp"
            try:
                base.mkdir(exist_ok=True)
            except PermissionError:
                pytest.skip("Cannot create /Users-rooted temp dir in this environment")
            # Use a sub-directory unique to this test invocation
            unique = base / uuid4().hex
            try:
                unique.mkdir()
            except PermissionError:
                pytest.skip("Cannot create /Users-rooted temp dir in this environment")
            yield unique
            import shutil

            shutil.rmtree(unique, ignore_errors=True)
        else:
            yield tmp_path

    def test_dot_and_hyphen_in_dirname_via_greedy(self, users_tmp: Path) -> None:
        """GitHub.nosync/my-project — dot parent + hyphenated child (issue #47).

        Simple replace-all gives ``…/GitHub.nosync/my/project`` which does not
        exist, so the greedy decoder must reconstruct ``my-project``.
        """
        project = users_tmp / "GitHub.nosync" / "my-project"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            # Greedy branch not reached outside /Users; just confirm no crash
            assert result is None or result == project

    def test_multi_hyphen_dot_dirname_via_greedy(self, users_tmp: Path) -> None:
        """my-cool-project.nosync/app — primary regression from issue #47.

        Three tokens joined by hyphens form the parent dir name; the old code
        only tried pairs and therefore could never reconstruct this component.
        """
        project = users_tmp / "my-cool-project.nosync" / "app"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    def test_flattened_dot_dirname_via_greedy(self, users_tmp: Path) -> None:
        """GitHub.nosync/thebest should decode from GitHub-nosync-thebest."""
        project = users_tmp / "GitHub.nosync" / "thebest"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-").replace(".", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project
