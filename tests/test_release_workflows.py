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


def test_no_openssl_sys_in_wheel_build_tree() -> None:
    """STRUCTURAL INVARIANT: openssl-sys must NOT appear in the wheel
    build's resolved dependency graph.

    This is the load-bearing assertion for the entire build pipeline.
    If openssl-sys is in the wheel-build resolution graph, every
    Linux/macOS surface that builds from source needs system OpenSSL
    + perl modules + pkg-config — and we've spent five hot-fixes
    chasing whichever combination of perl modules / OpenSSL versions
    / pkg-config paths was missing in each manylinux/Dockerfile/
    devcontainer surface. The cleanest fix is to NOT depend on
    OpenSSL at all.

    fastembed exposes `hf-hub-rustls-tls` and
    `ort-download-binaries-rustls-tls` features that replace its
    default `native-tls` path. With `default-features = false` plus
    those rustls features enabled in headroom-core, our entire build
    tree uses rustls and no crate pulls openssl-sys.

    This test runs `cargo tree` (so it actually exercises the
    resolved feature graph, not just declared Cargo.toml features).
    A future refactor that adds a transitive native-tls user will
    fail here, surfaced at PR time rather than 5 minutes into a CI
    wheel-build error.
    """
    import subprocess

    for crate in ("headroom-py", "headroom-proxy", "headroom-core"):
        result = subprocess.run(
            [
                "cargo",
                "tree",
                "--target",
                "x86_64-unknown-linux-gnu",
                "-p",
                crate,
                "-i",
                "openssl-sys",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        # `cargo tree -i <pkg>` exits 101 with "did not match any
        # packages" when the package is NOT in the tree — the GREEN
        # case. Exit 0 with a tree of consumers means it IS pulled.
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
        assert not_in_tree, (
            f"openssl-sys is back in {crate}'s build tree:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            "Find the new native-tls user (likely a default-features=true "
            "on a transitive crate) and disable it. Switching every "
            "transitive HTTP+TLS consumer to rustls is the load-bearing "
            "invariant that keeps wheel builds working without system "
            "OpenSSL or perl modules."
        )


def test_no_native_tls_in_wheel_build_tree() -> None:
    """The dual of the openssl-sys gate: native-tls is the proximate
    cause of openssl-sys being pulled. Catch it earlier with a more
    specific error message so future debugging starts at the right
    place.
    """
    import subprocess

    for crate in ("headroom-py", "headroom-proxy", "headroom-core"):
        result = subprocess.run(
            [
                "cargo",
                "tree",
                "--target",
                "x86_64-unknown-linux-gnu",
                "-p",
                crate,
                "-i",
                "native-tls",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
        assert not_in_tree, (
            f"native-tls is back in {crate}'s build tree — likely some "
            f"crate's `default-features = true` re-enabled native-tls "
            f"transitively:\n{result.stdout}"
        )


def test_fastembed_uses_rustls_features() -> None:
    """The mechanism that keeps openssl-sys out of the build is
    fastembed's explicit rustls feature selection in headroom-core.
    fastembed's default features include `hf-hub-native-tls` and
    `ort-download-binaries-native-tls` — both pull openssl-sys.
    Disabling defaults and enabling the rustls equivalents removes
    the OpenSSL surface entirely.
    """
    cargo = (ROOT / "crates" / "headroom-core" / "Cargo.toml").read_text(encoding="utf-8")

    assert "default-features = false" in cargo
    assert '"hf-hub-rustls-tls"' in cargo
    assert '"ort-download-binaries-rustls-tls"' in cargo
    # `image-models` is in default; we re-enable it explicitly so we
    # don't lose the image-embedding capability when defaults are off.
    assert '"image-models"' in cargo


def test_dockerfiles_no_longer_install_openssl_devel() -> None:
    """Once openssl-sys is out of the build tree, every Dockerfile
    that used to install `openssl-devel` / `libssl-dev` for the Rust
    build can drop those packages. This test enforces the cleanup so
    a future refactor doesn't carry the old packages forward "just
    in case".

    The check looks only at non-comment lines so explanatory comments
    that mention the historical packages don't false-positive.
    """
    targets = [
        ROOT / "e2e" / "wrap" / "Dockerfile",
        ROOT / "e2e" / "init" / "Dockerfile",
        ROOT / "Dockerfile",
        ROOT / ".devcontainer" / "Dockerfile",
    ]

    forbidden = ["openssl-devel", "libssl-dev"]

    for target in targets:
        content = target.read_text(encoding="utf-8")
        non_comment = "\n".join(
            line for line in content.splitlines() if not line.lstrip().startswith("#")
        )
        for pkg in forbidden:
            assert pkg not in non_comment, (
                f"{target.relative_to(ROOT)} still installs {pkg!r} on a "
                f"non-comment line. The rustls-everywhere refactor removed "
                f"openssl-sys from the build tree; this package is no "
                f"longer needed."
            )


def test_release_yml_does_not_install_openssl_or_perl_for_wheels() -> None:
    """With openssl-sys out of the build tree (verified by
    test_no_openssl_sys_in_wheel_build_tree), the previous
    before-script-linux that installed perl-IPC-Cmd / perl /
    perl-utils for the openssl-src vendored Configure script is
    obsolete. Removing it speeds the wheel build and keeps the
    Linux entry honest — every package install we keep here
    represents a hidden assumption about the manylinux container.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    non_comment = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))

    # No legacy install commands or env vars must appear on a non-comment
    # line. Each forbidden token represents an assumption about system
    # OpenSSL that the rustls refactor removed.
    forbidden = [
        "openssl-devel",
        "libssl-dev",
        "perl-IPC-Cmd",
        "libipc-cmd-perl",
        "OPENSSL_DIR",
    ]
    for token in forbidden:
        assert token not in non_comment, (
            f"release.yml build-wheels job still references {token!r} on "
            f"a non-comment line. The rustls-everywhere refactor removed "
            f"openssl-sys from the build tree; this command/env is now "
            f"obsolete."
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
