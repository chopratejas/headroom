"""Tests for version-sync.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def temp_project(tmp_path: Path) -> dict[str, Path]:
    """Create a temporary project with all versioned files."""
    # Create directory structure
    root = tmp_path / "project"
    headroom = root / "headroom"
    headroom.mkdir(parents=True)
    plugins = root / "plugins"
    openclaw = plugins / "openclaw"
    openclaw.mkdir(parents=True)
    sdk = root / "sdk"
    typescript = sdk / "typescript"
    typescript.mkdir(parents=True)

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.5.25"\n')

    # headroom/_version.py
    version_py = headroom / "_version.py"
    version_py.write_text('"""Package version metadata."""\n\n__version__ = "0.5.25"\n')

    # plugins/openclaw/package.json
    openclaw_pkg = openclaw / "package.json"
    openclaw_pkg.write_text(json.dumps({"name": "test", "version": "0.5.25"}))

    # sdk/typescript/package.json
    typescript_pkg = typescript / "package.json"
    typescript_pkg.write_text(json.dumps({"name": "test", "version": "0.5.25"}))

    return {
        "root": root,
        "pyproject": pyproject,
        "version_py": version_py,
        "openclaw_pkg": openclaw_pkg,
        "typescript_pkg": typescript_pkg,
    }


def test_version_sync_explicit_version(temp_project: dict[str, Path]) -> None:
    """Test --version flag updates all files."""
    root = temp_project["root"]
    script = Path(__file__).parent.parent / "version-sync.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--version", "0.7.0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify pyproject.toml
    pyproject_content = temp_project["pyproject"].read_text()
    assert 'version = "0.7.0"' in pyproject_content

    # Verify headroom/_version.py
    version_py_content = temp_project["version_py"].read_text()
    assert '__version__ = "0.7.0"' in version_py_content

    # Verify plugins/openclaw/package.json
    openclaw_pkg = json.loads(temp_project["openclaw_pkg"].read_text())
    assert openclaw_pkg["version"] == "0.7.0"

    # Verify sdk/typescript/package.json
    typescript_pkg = json.loads(temp_project["typescript_pkg"].read_text())
    assert typescript_pkg["version"] == "0.7.0"

    # Verify .releaseetadata was created
    release_metadata = root / ".releaseetadata"
    assert release_metadata.exists()
    metadata = json.loads(release_metadata.read_text())
    assert metadata["version"] == "0.7.0"
    assert metadata["packages"]["pypi"] == "0.7.0"
    assert metadata["packages"]["npm-sdk"] == "0.7.0"
    assert metadata["packages"]["npm-openclaw"] == "0.7.0"


def test_bump_patch(temp_project: dict[str, Path]) -> None:
    """Test --bump patch bumps 0.5.25 to 0.5.26."""
    root = temp_project["root"]
    script = Path(__file__).parent.parent / "version-sync.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--bump", "patch"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify all files updated to 0.5.26
    pyproject_content = temp_project["pyproject"].read_text()
    assert 'version = "0.5.26"' in pyproject_content

    version_py_content = temp_project["version_py"].read_text()
    assert '__version__ = "0.5.26"' in version_py_content

    openclaw_pkg = json.loads(temp_project["openclaw_pkg"].read_text())
    assert openclaw_pkg["version"] == "0.5.26"

    typescript_pkg = json.loads(temp_project["typescript_pkg"].read_text())
    assert typescript_pkg["version"] == "0.5.26"


def test_bump_minor(temp_project: dict[str, Path]) -> None:
    """Test --bump minor bumps 0.5.25 to 0.6.0."""
    root = temp_project["root"]
    script = Path(__file__).parent.parent / "version-sync.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--bump", "minor"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify all files updated to 0.6.0
    pyproject_content = temp_project["pyproject"].read_text()
    assert 'version = "0.6.0"' in pyproject_content

    version_py_content = temp_project["version_py"].read_text()
    assert '__version__ = "0.6.0"' in version_py_content

    openclaw_pkg = json.loads(temp_project["openclaw_pkg"].read_text())
    assert openclaw_pkg["version"] == "0.6.0"

    typescript_pkg = json.loads(temp_project["typescript_pkg"].read_text())
    assert typescript_pkg["version"] == "0.6.0"


def test_bump_major(temp_project: dict[str, Path]) -> None:
    """Test --bump major bumps 0.5.25 to 1.0.0."""
    root = temp_project["root"]
    script = Path(__file__).parent.parent / "version-sync.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--bump", "major"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify all files updated to 1.0.0
    pyproject_content = temp_project["pyproject"].read_text()
    assert 'version = "1.0.0"' in pyproject_content

    version_py_content = temp_project["version_py"].read_text()
    assert '__version__ = "1.0.0"' in version_py_content

    openclaw_pkg = json.loads(temp_project["openclaw_pkg"].read_text())
    assert openclaw_pkg["version"] == "1.0.0"

    typescript_pkg = json.loads(temp_project["typescript_pkg"].read_text())
    assert typescript_pkg["version"] == "1.0.0"


def test_release_metadata_written(temp_project: dict[str, Path]) -> None:
    """Test .releaseetadata is written correctly."""
    root = temp_project["root"]
    script = Path(__file__).parent.parent / "version-sync.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--version", "0.6.0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    release_metadata = root / ".releaseetadata"
    assert release_metadata.exists()

    metadata = json.loads(release_metadata.read_text())
    assert metadata == {
        "version": "0.6.0",
        "packages": {
            "pypi": "0.6.0",
            "npm-sdk": "0.6.0",
            "npm-openclaw": "0.6.0",
        },
    }
