#!/usr/bin/env bash
# Build the Rust → Python extension (headroom._core) and link it into the
# in-tree `headroom/` package so `import headroom._core` resolves.
#
# Why a wrapper script: `maturin develop` builds the `.so` and installs it
# into the venv's site-packages, but the in-tree `headroom/` source
# directory (loaded via `pip install -e .`) shadows that on sys.path.
# Python finds `headroom/__init__.py` at the project root before reaching
# the maturin overlay, so `import headroom._core` fails. Symlinking the
# built `.so` into `headroom/` fixes the lookup with zero copies.
#
# Idempotent. Safe to run repeatedly. Requires `maturin` in PATH (i.e.
# inside the project venv).

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v maturin >/dev/null 2>&1; then
    echo "error: maturin not found. Activate the venv first:" >&2
    echo "  source .venv/bin/activate" >&2
    exit 1
fi

# Build the wheel + install into the venv site-packages.
maturin develop -m crates/headroom-py/Cargo.toml

# Locate the built `.so`. `maturin develop` writes it under
# `crates/headroom-py/python/headroom/_core.cpython-<ver>-<platform>.so`.
SO_FILE=$(find crates/headroom-py/python/headroom -maxdepth 1 \
    -name "_core.cpython-*.so" -o -name "_core.cpython-*.dylib" -o -name "_core.pyd" \
    2>/dev/null | head -1)

if [[ -z "$SO_FILE" ]]; then
    echo "error: maturin develop succeeded but produced no _core.* binary." >&2
    exit 1
fi

# Symlink into the in-tree package dir.
LINK_NAME="headroom/$(basename "$SO_FILE")"
ln -sf "$(pwd)/$SO_FILE" "$LINK_NAME"
echo "linked: $LINK_NAME -> $SO_FILE"

# Smoke-test the import to fail loudly if anything is misconfigured.
python -c "from headroom._core import DiffCompressor; print('headroom._core OK:', DiffCompressor)"
