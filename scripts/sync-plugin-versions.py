"""Sync plugin manifest versions to the repo's computed release semver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from headroom.release_version import (  # noqa: E402
    compute_release_version,
    determine_bump_level,
    find_latest_release_tag,
    get_canonical_version,
    list_release_commits,
    list_release_tags,
)


def compute_repo_semver(root: Path) -> str:
    """Return the npm-style semver for the repo's next release."""
    tags = list_release_tags(root)
    previous_tag = find_latest_release_tag(tags) or ""
    level = determine_bump_level(list_release_commits(root, previous_tag))
    info = compute_release_version(
        canonical_version=get_canonical_version(root),
        level=level,
        tags=tags,
    )
    return info.npm_version


def main() -> None:
    root = ROOT
    version = compute_repo_semver(root)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "version-sync.py"),
            "--root",
            str(root),
            "--version",
            version,
            "--plugin-manifests-only",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
