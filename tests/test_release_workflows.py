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


def test_headroom_proxy_vendors_openssl() -> None:
    """`hf-hub` (transitive via `fastembed`) hard-codes `native-tls` as a
    default feature, forcing `openssl-sys` for the entire workspace
    despite our `reqwest`/`tokio-tungstenite`/`tokio-rustls` preferences.
    The wheel matrix breaks on every Linux + Intel-macOS target where
    the manylinux container's system OpenSSL is too old (manylinux2014
    ships 1.0.2k) or the cross-compile sysroot has none (aarch64).
    Enabling the `vendored` Cargo feature on `openssl` instructs
    `openssl-sys` to compile OpenSSL from source — works on every
    target uniformly.
    """
    cargo_toml = (ROOT / "crates" / "headroom-proxy" / "Cargo.toml").read_text(encoding="utf-8")

    # Guard against a future refactor that re-removes the vendored dep.
    assert 'openssl = { version = "0.10", features = ["vendored"] }' in cargo_toml


def test_build_wheels_installs_perl_ipc_cmd_for_vendored_openssl() -> None:
    """OpenSSL's vendored `Configure` script needs `IPC::Cmd`. Without
    it the build fails with `Can't locate IPC/Cmd.pm`. The before-script
    must (a) probe the module first to skip a no-op install when the
    container already has it, (b) cover RHEL family (dnf/yum) AND
    Debian/Ubuntu (apt-get) AND musllinux (apk) since the maturin-action
    uses different containers per target, and (c) fail loud with an
    explicit `perl -MIPC::Cmd -e 1` assertion if the install path
    didn't actually resolve the module — no silent fallback into a
    later confusing openssl-src build error.

    The previous shape used `libipc-cmd-perl` on the apt branch, which
    is a deprecated alias and is not in the default sources of the
    aarch64-cross container; that broke the aarch64 wheel build.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "before-script-linux:" in content

    # Pre-probe so we no-op when IPC::Cmd is already importable.
    assert "perl -MIPC::Cmd -e 1" in content

    # All three RHEL-family + Debian-family + Alpine package managers covered.
    assert "dnf install -y perl-IPC-Cmd" in content
    assert "yum install -y perl-IPC-Cmd" in content
    # The plain `perl` meta-package pulls perl-modules-* on Debian/Ubuntu,
    # which contains IPC::Cmd. The legacy `libipc-cmd-perl` alias must NOT
    # be referenced as an installable target — it is no longer in default
    # Debian/Ubuntu sources. We allow the string in YAML/shell comments
    # (operator hints) by checking only non-comment lines.
    assert "apt-get install -y --no-install-recommends perl" in content
    assert "apk add --no-cache perl-utils" in content

    non_comment_lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.lstrip()
        # Drop YAML and shell `#` comments. Mid-line `#` comments would
        # require a real YAML/shell tokenizer; the file's actual install
        # commands are never on the same physical line as a comment.
        if stripped.startswith("#"):
            continue
        non_comment_lines.append(raw)
    code_only = "\n".join(non_comment_lines)
    assert "libipc-cmd-perl" not in code_only, (
        "libipc-cmd-perl must not appear as an apt-get target — it is a "
        "deprecated alias not in default Debian/Ubuntu sources."
    )

    # Final fail-loud assertion (no silent fallback).
    assert "perl -MIPC::Cmd -e 'print \"IPC::Cmd loaded OK" in content


def test_build_wheels_does_not_set_openssl_dir() -> None:
    """`OPENSSL_DIR` (and the related `OPENSSL_LIB_DIR` /
    `OPENSSL_INCLUDE_DIR`) DEFEATS the `vendored` feature: openssl-sys
    will use the system OpenSSL at that path instead of compiling from
    source. The earlier macOS hot-fix exported these env vars; with
    vendored enabled they must NOT be set, otherwise we silently
    regress to the system-OpenSSL path that broke originally.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Locate the build-wheels job and assert it does not export OPENSSL_DIR.
    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    assert "OPENSSL_DIR" not in body, (
        "build-wheels must not set OPENSSL_DIR — it disables openssl-sys's "
        "vendored feature and reintroduces the system-OpenSSL dependency."
    )


def test_build_wheels_matrix_excludes_intel_macos() -> None:
    """`ort-sys 2.0.0-rc.12` (transitive via the ML compression backend)
    has no prebuilt ONNX Runtime binaries for `x86_64-apple-darwin`.
    Building ORT from source would add CMake + ~5 minutes per build.
    Apple Silicon macOS is fully covered; Intel-mac users install from
    the platform-independent sdist this matrix also produces.

    We assert against the actual matrix entry shape (`target: <triple>`
    on a non-comment line) so explanatory comments mentioning the
    excluded triple don't false-positive.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    matrix_targets: list[str] = []
    for raw in body.splitlines():
        stripped = raw.lstrip()
        # Skip YAML comments — only look at real matrix-entry lines.
        if stripped.startswith("#"):
            continue
        if stripped.startswith("target:"):
            # `target: x86_64-apple-darwin` → `x86_64-apple-darwin`
            matrix_targets.append(stripped.split(":", 1)[1].strip())

    assert "aarch64-apple-darwin" in matrix_targets, "Apple Silicon must stay in the matrix"
    assert "x86_64-unknown-linux-gnu" in matrix_targets
    assert "aarch64-unknown-linux-gnu" in matrix_targets

    # Intel macOS must NOT be a matrix entry — re-add only after switching
    # off ort-sys (e.g., to ort-tract) or adding a CMake-from-source step.
    assert "x86_64-apple-darwin" not in matrix_targets, (
        f"x86_64-apple-darwin must not be a wheel-matrix target; got {matrix_targets}"
    )

    # The runner OS itself shouldn't appear as a configured `os:` either.
    matrix_os: list[str] = []
    for raw in body.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("os:"):
            matrix_os.append(stripped.split(":", 1)[1].strip())
    assert "macos-15-intel" not in matrix_os


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
