"""Tests for release version normalization and bumping."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from headroom.release_version import (
    CommitInfo,
    ReleaseVersionInfo,
    SemVer,
    classify_commit_bump,
    commit_height_since,
    compute_release_version,
    determine_bump_level,
    find_latest_release_tag,
    get_canonical_version,
    list_release_commits,
    list_release_tags,
    main,
    normalize_release_tag,
    parse_release_tag,
    write_github_outputs,
)

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_release_tag_preserves_three_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.20")) == "0.5.20"


def test_normalize_release_tag_collapses_four_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.25.2")) == "0.5.25"


def test_compute_patch_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.5.26"
    assert info.npm_version == "0.5.26"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "patch"


def test_compute_minor_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="minor",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "minor"


def test_compute_patch_release_from_canonical_without_tags() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=[],
    )

    assert info.version == "0.5.26"
    assert info.npm_version == "0.5.26"
    assert info.previous_tag == ""


def test_manual_version_override_uses_single_semver() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.25.2"],
        manual_version="0.6.0",
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == ""
    assert info.bump == "manual"


def test_manual_version_override_rejects_legacy_four_part_version() -> None:
    with pytest.raises(ValueError, match="Invalid semantic version"):
        compute_release_version(
            canonical_version="0.5.25",
            level="patch",
            tags=["v0.5.25.2"],
            manual_version="0.5.25.3",
        )


def test_find_latest_release_tag_prefers_highest_normalized_version() -> None:
    assert find_latest_release_tag(["v0.5.25.2", "v0.5.27", "not-a-tag"]) == "v0.5.27"


def test_find_latest_release_tag_prefers_higher_legacy_height_with_same_base() -> None:
    assert find_latest_release_tag(["v0.5.25.2", "v0.5.25.3", "v0.5.25"]) == "v0.5.25.3"


def test_parse_release_tag_preserves_legacy_height_for_sorting() -> None:
    tag = parse_release_tag("v0.5.25.3")
    assert str(tag.version) == "0.5.25"
    assert tag.legacy_height == 3


def test_semver_helpers_cover_parse_bump_and_str() -> None:
    version = SemVer.parse("1.2.3")

    assert str(version) == "1.2.3"
    assert version.bump("major") == SemVer(2, 0, 0)
    assert version.bump("minor") == SemVer(1, 3, 0)
    assert version.bump("patch") == SemVer(1, 2, 4)

    with pytest.raises(ValueError, match="Invalid semantic version"):
        SemVer.parse("v1.2.3")

    with pytest.raises(ValueError, match="Unsupported bump level"):
        version.bump("build")


def test_release_version_info_as_outputs() -> None:
    info = ReleaseVersionInfo(
        version="1.2.4",
        npm_version="1.2.4",
        canonical="1.2.3",
        height="7",
        bump="patch",
        previous_tag="v1.2.3",
    )

    assert info.as_outputs() == {
        "version": "1.2.4",
        "npm_version": "1.2.4",
        "canonical": "1.2.3",
        "height": "7",
        "bump": "patch",
        "previous_tag": "v1.2.3",
    }


def test_parse_release_tag_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Invalid release tag"):
        parse_release_tag("release-1.2.3")


def test_classify_commit_bump_treats_breaking_change_as_major() -> None:
    assert (
        classify_commit_bump(
            CommitInfo(subject="fix(api)!: change response shape", body=""),
        )
        == "major"
    )


def test_classify_commit_bump_uses_merge_summary_and_breaking_body() -> None:
    assert (
        classify_commit_bump(
            CommitInfo(
                subject="Merge pull request #1 from feature/thing",
                body="\n\nfeat(api): add window export\n\nmore detail",
            )
        )
        == "minor"
    )
    assert (
        classify_commit_bump(
            CommitInfo(subject="Merge branch 'topic'", body="BREAKING CHANGE: drop support"),
        )
        == "major"
    )


def test_determine_bump_level_uses_greatest_commit_level() -> None:
    commits = [
        CommitInfo(subject="fix: patch one", body=""),
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(subject="chore: maintenance", body=""),
    ]

    assert determine_bump_level(commits) == "minor"


def test_determine_bump_level_prefers_major_over_minor_and_patch() -> None:
    commits = [
        CommitInfo(subject="fix: patch one", body=""),
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(
            subject="docs: update migration guide",
            body="BREAKING CHANGE: the API changed",
        ),
    ]

    assert determine_bump_level(commits) == "major"


def test_list_release_commits_parses_empty_body_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    run.return_value = Mock(
        stdout="feat: add capability\x1f\x1efix: patch bug\x1fbody text\x1e",
    )
    monkeypatch.setattr("headroom.release_version.subprocess.run", run)

    commits = list_release_commits(ROOT, "")

    assert commits == [
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(subject="fix: patch bug", body="body text"),
    ]


def test_list_release_tags_filters_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "headroom.release_version.subprocess.run",
        lambda *args, **kwargs: Mock(stdout="v0.1.0\n\nv0.2.0\n"),
    )

    assert list_release_tags(ROOT) == ["v0.1.0", "v0.2.0"]


def test_commit_height_since_handles_empty_and_missing_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    assert commit_height_since(ROOT, "") == "0"

    monkeypatch.setattr(
        "headroom.release_version.subprocess.run",
        lambda *args, **kwargs: Mock(stdout="\n"),
    )

    assert commit_height_since(ROOT, "v0.1.0") == "0"


def test_write_github_outputs_appends_all_values(tmp_path: Path) -> None:
    output_path = tmp_path / "github-output.txt"
    info = ReleaseVersionInfo(
        version="1.0.1",
        npm_version="1.0.1",
        canonical="1.0.0",
        height="2",
        bump="patch",
        previous_tag="v1.0.0",
    )

    write_github_outputs(info, str(output_path))

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "version=1.0.1",
        "npm_version=1.0.1",
        "canonical=1.0.0",
        "height=2",
        "bump=patch",
        "previous_tag=v1.0.0",
    ]


def test_main_prints_outputs_when_github_output_is_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setenv("LEVEL", "")
    monkeypatch.setattr("headroom.release_version.list_release_tags", lambda root: ["v1.2.3"])
    monkeypatch.setattr("headroom.release_version.find_latest_release_tag", lambda tags: "v1.2.3")
    monkeypatch.setattr(
        "headroom.release_version.list_release_commits",
        lambda root, previous_tag: [CommitInfo(subject="feat: add thing", body="")],
    )
    monkeypatch.setattr("headroom.release_version.determine_bump_level", lambda commits: "minor")
    monkeypatch.setattr("headroom.release_version.get_canonical_version", lambda root: "1.2.3")
    monkeypatch.setattr(
        "headroom.release_version.compute_release_version",
        lambda canonical_version, level, tags, manual_version="": ReleaseVersionInfo(
            version="1.3.0",
            npm_version="1.3.0",
            canonical="1.2.3",
            height="0",
            bump="minor",
            previous_tag="v1.2.3",
        ),
    )
    monkeypatch.setattr(
        "headroom.release_version.commit_height_since", lambda root, previous_tag: "4"
    )

    main()

    assert capsys.readouterr().out.splitlines() == [
        "version=1.3.0",
        "npm_version=1.3.0",
        "canonical=1.2.3",
        "height=4",
        "bump=minor",
        "previous_tag=v1.2.3",
    ]


def test_release_version_script_runs_directly_without_importing_headroom_package(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "github-output.txt"
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(output_path)
    env["LEVEL"] = "patch"
    env["MANUAL_VER"] = "0.6.0"

    result = subprocess.run(
        [sys.executable, str(ROOT / "headroom" / "release_version.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    canonical_version = get_canonical_version(ROOT)
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "version=0.6.0",
        "npm_version=0.6.0",
        f"canonical={canonical_version}",
        "height=0",
        "bump=manual",
        "previous_tag=",
    ]
