"""Workflow regression tests for release publishing behavior."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_docker_workflow_normalizes_repository_name_for_signing() -> None:
    content = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")

    assert "id: image-name" in content
    assert "tr '[:upper:]' '[:lower:]'" in content
    assert "steps.image-name.outputs.image_name" in content


def test_release_workflow_publishes_both_node_packages_to_github_packages() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Publish ${{ env.NPM_SDK_PACKAGE }} to GitHub Package Registry" in content
    assert "Publish ${{ env.NPM_OPENCLAW_PACKAGE }} to GitHub Package Registry" in content
    assert "pkg.name = `@${process.env.GITHUB_PACKAGES_SCOPE}/${pkg.name}`;" in content


def test_create_release_runs_after_successful_build_even_if_other_publishes_fail() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert (
        "needs: [detect-version, build, publish-pypi, publish-npm, publish-github-packages, publish-docker]"
        in content
    )
    assert "always()" in content
    assert "needs.build.result == 'success'" in content
