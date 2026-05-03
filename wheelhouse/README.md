# wheelhouse/

This directory is populated by the
`.github/actions/build-rust-core-wheel` composite action with the
locally-built `headroom-core-py-*.whl`.

The directory must EXIST in the source tree (with this README and the
sentinel `.gitkeep`) so that the `COPY wheelhouse ...` directive in
`e2e/wrap/Dockerfile` and `e2e/init/Dockerfile` doesn't fail on
clean checkouts where CI hasn't yet built a wheel.

For local development (when running `docker build` outside CI):

    pip install maturin
    maturin build --release -m crates/headroom-py/Cargo.toml --out wheelhouse
    docker build -f e2e/wrap/Dockerfile -t headroom-wrap-e2e .

Issue #355 / PR #357 — see those for full context.
