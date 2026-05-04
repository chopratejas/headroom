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
    assert (
        'unscoped_sdk_tarball="$(npm pack --pack-destination "$assets_dir" | tail -n 1)"' in content
    )
    assert "SDK_TARBALL: ${{ steps.gpr-sdk-publish.outputs.unscoped_sdk_tarball }}" in content


def test_release_workflow_publishes_python_distributions_to_github_release() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Publish ${{ env.PYPI_PACKAGE }} Python distributions to GitHub Release" in content
    assert (
        'gh release upload "$TAG" release-assets/*.whl release-assets/*.tar.gz --clobber' in content
    )
    assert "Publish Node package tarballs to GitHub Release" in content
    assert 'gh release upload "$TAG" release-assets/*.tgz --clobber' in content


def test_create_release_runs_after_successful_build_even_if_other_publishes_fail() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Single-wheel maturin refactor (PR #360) added `build-wheels` (the
    # cross-platform matrix that produces the linux/macos/aarch64 wheels)
    # and `collect-dist` (aggregator that merges wheel artifacts + npm
    # release-assets) between `build` and the publish jobs. create-release
    # must wait for all of them.
    assert (
        "needs: [detect-version, build, build-wheels, collect-dist, publish-pypi, publish-npm, publish-github-packages, publish-docker]"
        in content
    )
    assert "always()" in content
    assert "needs.build.result == 'success'" in content
    assert "needs.build-wheels.result == 'success'" in content
    assert "needs.collect-dist.result == 'success'" in content


def test_macos_native_wrapper_dependency_install_retries_pypi_downloads() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m pip install --retries 10 --timeout 60 pytest" in content


def test_ci_commitlint_skips_default_github_merge_commits() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "github.event_name != 'push'" in content
    assert "!startsWith(github.event.head_commit.message, 'Merge pull request ')" in content


def test_build_wheels_installs_openssl_devel_on_linux_via_before_script() -> None:
    """The wheel matrix uses PyO3/maturin-action's manylinux container,
    which does not inherit our e2e Dockerfiles' yum installs. Without an
    explicit before-script-linux, openssl-sys (transitive via fastembed
    → hf-hub → ureq → native-tls) fails to find pkg-config'd OpenSSL
    and the Linux x86_64/aarch64 wheels never build.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "before-script-linux:" in content
    assert "yum install -y openssl-devel" in content
    # apt-get fallback for non-RHEL manylinux variants
    assert "libssl-dev pkg-config" in content


def test_build_wheels_resolves_openssl_dir_explicitly_on_macos() -> None:
    """macOS Intel runners (`macos-15-intel`) keep Homebrew under
    `/usr/local/Cellar`, which `openssl-sys` does not auto-discover.
    aarch64 runners use `/opt/homebrew/opt/openssl@3` which IS on the
    default path. Setting OPENSSL_DIR explicitly fixes Intel without
    regressing aarch64.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Install OpenSSL (macOS)" in content
    assert "if: runner.os == 'macOS'" in content
    assert "brew install openssl@3" in content
    assert 'echo "OPENSSL_DIR=$openssl_dir"' in content


def test_npm_publish_jobs_do_not_download_dist_artifact() -> None:
    """`publish-npm` and `publish-github-packages` `npm pack`+`npm publish`
    directly from the checked-out source tree; they never read the
    Python `dist` artifact. The earlier speculative download was failing
    "Artifact not found" because neither job is gated on `collect-dist`.
    Ensure no future refactor re-adds the dead step.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Locate publish-npm + publish-github-packages bodies and assert
    # neither contains a download-artifact step that pulls `name: dist`.
    npm_start = content.index("\n  publish-npm:")
    npm_end = content.index("\n  publish-github-packages:")
    publish_npm_body = content[npm_start:npm_end]

    gpr_start = content.index("\n  publish-github-packages:")
    gpr_end = content.index("\n  publish-docker:")
    publish_gpr_body = content[gpr_start:gpr_end]

    for body, label in (
        (publish_npm_body, "publish-npm"),
        (publish_gpr_body, "publish-github-packages"),
    ):
        assert "download-artifact" not in body, (
            f"{label} must not download the `dist` artifact — it `npm pack`s "
            f"its own tarball and the speculative download fails when "
            f"collect-dist hasn't run."
        )
