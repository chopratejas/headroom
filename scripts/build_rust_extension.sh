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
# Hotfix-A0 (2026-05-02): the script now also runs an end-to-end import
# verification with the `hello()` marker so a partial / stale build is
# caught before the proxy is started. This mirrors the lifespan smoke
# test in `headroom.proxy.server._check_rust_core` so dev-time and
# deploy-time both catch the same class of failure.
#
# Idempotent. Safe to run repeatedly. Requires `maturin` in PATH (i.e.
# inside the project venv).

set -euo pipefail

# Move to repo root so all relative paths below are stable.
cd "$(dirname "$0")/.."

log() {
    printf '[build_rust_extension] %s\n' "$*" >&2
}

fail() {
    printf '[build_rust_extension] error: %s\n' "$*" >&2
    exit 1
}

# Step 1: pre-flight. Maturin must be on PATH and a venv must be active —
# `maturin develop` writes into site-packages, and we want that write to
# land in the same env the proxy will run in.
if ! command -v maturin >/dev/null 2>&1; then
    fail "maturin not found on PATH. Activate the venv first: source .venv/bin/activate"
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log "warning: VIRTUAL_ENV is unset; maturin will install into the system Python."
    log "         If that is not what you want, abort and 'source .venv/bin/activate' first."
fi

# Step 2: build + install via `maturin develop` (in-place editable install
# with C extensions). This produces a `.so` under
# crates/headroom-py/python/headroom/.
log "step 1/3: maturin develop"
maturin develop -m crates/headroom-py/Cargo.toml \
    || fail "maturin develop failed (see output above)"

# Step 3: locate the built artifact. `maturin develop` writes
# `_core.cpython-<ver>-<platform>.{so,dylib,pyd}` into the package dir.
SO_FILE=$(find crates/headroom-py/python/headroom -maxdepth 1 \
    \( -name "_core.cpython-*.so" -o -name "_core.cpython-*.dylib" -o -name "_core.pyd" \) \
    2>/dev/null | head -1 || true)

if [[ -z "${SO_FILE}" ]]; then
    fail "maturin develop succeeded but produced no _core.* binary in crates/headroom-py/python/headroom/"
fi

# Step 4: symlink into the in-tree package dir so the in-tree
# `headroom/__init__.py` resolves the `_core` submodule. Only the symlink
# style is supported; copy semantics drift on every rebuild.
LINK_NAME="headroom/$(basename "${SO_FILE}")"
ln -sf "$(pwd)/${SO_FILE}" "${LINK_NAME}" \
    || fail "failed to symlink ${SO_FILE} into ${LINK_NAME}"
log "step 2/3: linked ${LINK_NAME} -> ${SO_FILE}"

# Step 5: end-to-end import verification. This is the same check the
# proxy lifespan runs at startup. Failing here means the build produced
# something that can't be loaded — fix the build, don't fix the proxy.
log "step 3/3: verifying \`from headroom._core import hello\`"
python -c '
import sys
try:
    from headroom._core import hello, DiffCompressor
except Exception as exc:
    print(f"verify FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
marker = hello()
if marker != "headroom-core":
    print(f"verify FAILED: hello() returned {marker!r}, expected \"headroom-core\"", file=sys.stderr)
    sys.exit(1)
print(f"verify OK: hello()={marker!r}, DiffCompressor={DiffCompressor!r}")
' || fail "import verification failed (see above)"

log "headroom._core build + install + verify: OK"
