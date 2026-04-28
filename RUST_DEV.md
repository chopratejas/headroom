# Headroom Rust Rewrite — Developer Guide

This document covers the Rust port of Headroom. It is the only new top-level
doc created in Phase 0; longer-form design/plan writeups live elsewhere and
are not versioned in this repo.

## Workspace layout

```
Cargo.toml                       # workspace root
rust-toolchain.toml              # pins stable rustc with rustfmt+clippy
crates/
  headroom-core/                 # library: shared types + transform trait surface
  headroom-proxy/                # binary: axum /healthz (Phase 2 grows this)
  headroom-py/                   # PyO3 cdylib exposing `headroom._core`
  headroom-parity/               # lib + `parity-run` CLI for Python parity tests
tests/parity/
  fixtures/<transform>/*.json    # recorded Python outputs (Phase 1 ports match)
  recorder.py                    # Python-side fixture recorder
scripts/record_fixtures.py       # entry point for running the recorder
```

`cargo build --workspace` builds every crate. `default-members` drops
`headroom-py` from `cargo run`/bare-`cargo test` flows so that `cargo test
--workspace` does not try to execute the PyO3 cdylib standalone (it can't
find `libpython` without a Python interpreter hosting it).

## Common commands

`just` is not installed on dev boxes here; a `Makefile` at the repo root
exposes the same targets:

| Target | What it does |
| --- | --- |
| `make test` | `cargo test --workspace` |
| `make test-parity` | Builds `headroom-py` via maturin, runs `parity-run run` |
| `make bench` | `cargo bench --workspace` |
| `make build-proxy` | Release-builds `headroom-proxy`, strips, prints size |
| `make build-wheel` | `maturin build --release -m crates/headroom-py/pyproject.toml` |
| `make fmt` | `cargo fmt --all` |
| `make lint` | `cargo fmt --check` + `cargo clippy --workspace -- -D warnings` |

## Running the proxy

`headroom-proxy` is a transparent reverse proxy. Phase 1 forwards HTTP/1.1,
HTTP/2, SSE, and WebSocket traffic verbatim to a configured upstream — no
provider logic yet. The intent is that operators run the existing Python
proxy on a private port and put `headroom-proxy` on the public port pointed
at it; end users notice nothing.

```bash
# Build
make build-proxy
./target/release/headroom-proxy --help

# Run against a local upstream
./target/release/headroom-proxy \
    --listen 0.0.0.0:8787 \
    --upstream http://127.0.0.1:8788

# Health checks
curl -s http://127.0.0.1:8787/healthz            # => {"ok":true,...}
curl -s http://127.0.0.1:8787/healthz/upstream   # => 200 if upstream reachable
```

### Operator runbook (Phase 1 cutover)

```bash
# 1. Move the Python proxy to a private port (e.g. 8788)
HEADROOM_BIND=127.0.0.1:8788 python -m headroom.proxy &     # or your existing launcher

# 2. Run the Rust proxy on the previously-public port (8787) pointing at it
./target/release/headroom-proxy --listen 0.0.0.0:8787 --upstream http://127.0.0.1:8788 &

# 3. End users keep hitting :8787 unchanged.
# 4. Confirm passthrough:
curl -si http://127.0.0.1:8787/v1/models
# 5. Rollback = stop the Rust proxy and rebind Python back to 8787.
```

### Configuration flags

| Flag | Env var | Default | Notes |
| --- | --- | --- | --- |
| `--listen` | `HEADROOM_PROXY_LISTEN` | `0.0.0.0:8787` | bind address |
| `--upstream` | `HEADROOM_PROXY_UPSTREAM` | (required) | base URL the proxy forwards to |
| `--upstream-timeout` |  | `600s` | end-to-end request timeout (long for streams) |
| `--upstream-connect-timeout` |  | `10s` | TCP/TLS connect timeout |
| `--max-body-bytes` |  | `100MB` | for buffered cases; streams bypass |
| `--log-level` |  | `info` | `RUST_LOG`-style filter |
| `--rewrite-host` / `--no-rewrite-host` | | rewrite | rewrite Host to upstream (default) |
| `--graceful-shutdown-timeout` | | `30s` | wait for in-flight on SIGTERM/SIGINT |

### Reserved paths

`/healthz` and `/healthz/upstream` are intercepted by the Rust proxy and
**not** forwarded. Operators must not name a real upstream route either of
these. Everything else is a catch-all forward.

## Maturin + Python wiring

`headroom-py` is a PyO3 cdylib that exposes `headroom._core` in Python. The
`extension-module` feature is opt-in so plain `cargo build --workspace` does
not try to link against `libpython` on systems that don't have it.

### First-time setup (clean venv recommended)

```bash
python3.11 -m venv /tmp/hr-rust-venv
source /tmp/hr-rust-venv/bin/activate
pip install maturin
cd crates/headroom-py
maturin develop           # editable dev build, installs headroom._core
cd /tmp                   # IMPORTANT: step out of the repo root first
python -c "from headroom._core import hello; print(hello())"
# => headroom-core
```

> Why `cd /tmp`? The repo root also contains the Python `headroom/` package.
> Running the smoke import from the repo root makes Python resolve `headroom`
> to `./headroom/__init__.py` (the full SDK, which pulls in heavy deps) instead
> of the lightweight namespace package installed by maturin. Tests should
> either run outside the repo root, or ensure `headroom` is installed into
> the same venv (then the maturin-installed `_core.so` lands alongside it and
> both imports resolve).

### Release wheels

```bash
make build-wheel
# wheels land under target/wheels/
```

CI (`.github/workflows/rust.yml`) builds linux-x86_64, macos-arm64, and
macos-x86_64 wheels via `PyO3/maturin-action` and uploads them as artifacts.

## Parity harness

`crates/headroom-parity` owns the Rust-vs-Python oracle:

- JSON fixtures under `tests/parity/fixtures/<transform>/` (schema:
  `{ transform, input, config, output, recorded_at, input_sha256 }`).
- `TransformComparator` trait — one impl per transform. Phase 0 stubs return
  `Err(...)`; the harness flags those as `Skipped`, not panics.
- `parity-run` CLI: `cargo run -p headroom-parity -- run [--only TRANSFORM]`.
- Unit tests in `crates/headroom-parity/src/lib.rs` include a **negative
  test** (`harness_reports_diff_for_divergent_comparator`) proving the
  harness detects mismatched output before any real port lands.

### Recording fresh fixtures

```bash
source .venv/bin/activate           # the main Python SDK venv
python scripts/record_fixtures.py   # uses tests/parity/recorder.py
ls tests/parity/fixtures/*/ | sort | uniq -c
```

The recorder monkey-patches the in-process transform classes (see
`record_all()` in `tests/parity/recorder.py`). It does **not** modify any
file under `headroom/`.

## Phase 0 Blockers

These are known limitations for Phase 0. They are tracked here so Phase 1
doesn't rediscover them.

- **`cache_aligner` fixtures**: `CacheAligner.apply()` takes
  `(messages, tokenizer, **kwargs)` — a `Tokenizer` is provider-specific and
  its cheapest `NoopTokenCounter` / `TiktokenTokenCounter` construction still
  requires pulling `headroom.providers.*` which imports the full observability
  stack (opentelemetry, etc). The recorder records `cache_aligner` only if a
  usable tokenizer is cheaply available; otherwise it logs a blocker and
  skips. See `recorder.py::_build_cache_aligner_tokenizer`.
- **`ccr` is not a single class**: The repo has `CCRToolInjector`,
  `CCRResponseHandler`, `CCRToolCall`, `CCRToolResult` etc. rather than a
  single `CCR` class. The recorder targets the encoder-style entry point
  most analogous to the Rust port (`CCRToolInjector.inject_tool` and
  `CCRResponseHandler.parse_response`). If Phase 1 wants a different split
  it should update `recorder.py::record_all` accordingly.
- **Pre-commit hook noise**: `scripts/sync-plugin-versions.py` mutates
  `.claude-plugin/marketplace.json`, `.github/plugin/marketplace.json`, and
  `plugins/headroom-agent-hooks/**/plugin.json` on every commit. Those
  changes are harmless but each commit in Phase 0 picks them up. Phase 1
  does not need to do anything special — just let the hook run.
- **`rust-toolchain.toml`** pins `channel = "stable"` rather than a specific
  version so CI picks up the same toolchain the local box uses. Tighten to a
  pinned version (e.g. `1.78`) once the port stabilizes.
