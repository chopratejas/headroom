ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.6.17
# Pinned 2026-04-15. Update via Dependabot or: docker pull python:3.11-slim
ARG PYTHON_DIGEST=sha256:233de06753d30d120b1a3ce359d8d3be8bda78524cd8f520c99883bfe33964cf
# Pinned 2026-04-15. Update via Dependabot or: docker pull gcr.io/distroless/python3-debian13
ARG DISTROLESS_DIGEST=sha256:ed3a4beb46f8f8baac068743ba1b1f95ea3f793422129cf6dd23967f779b6018
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13
ARG PYTHON_SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages

# ---- Build stage: compile native extensions, build wheel ----
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST} AS builder

ARG UV_VERSION

# build-essential / g++ for any C extension wheels uv may need to build
# from source. curl + ca-certificates are required by the rustup
# bootstrap below. Hotfix-A0 (Finding #2) added the rust toolchain so the
# image actually carries `headroom._core`; previously the runtime image
# shipped without the Rust extension and every compressed request fell
# back to a Python-only path or no-op.
#
# `pkg-config` + `libssl-dev` are required because the workspace
# transitively pulls `openssl-sys` (via reqwest/native-tls in some
# dependency chain). Without them, `cargo` fails the maturin build with
# "Could not find openssl via pkg-config" — observed in PR #350 CI.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    patchelf \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==${UV_VERSION}

# Rust toolchain for the headroom._core extension build. Pinned via
# rust-toolchain.toml at the repo root so this matches what local devs
# build with. Installed as root before WORKDIR change so the env
# additions stick for every subsequent RUN.
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

WORKDIR /build

# Layer 1: install deps only (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
# Stub package so uv can resolve the local extras without full source
RUN mkdir -p headroom && touch headroom/__init__.py
ARG HEADROOM_EXTRAS=proxy,code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[${HEADROOM_EXTRAS}]"

# Layer 2 (Hotfix-A0): build and install the Rust extension wheel
# BEFORE installing headroom-ai source. Why this order:
#
#   * The headroom-core-py wheel includes a stub `headroom/__init__.py`
#     plus `headroom/_core.cpython-*.so` (maturin's `python-source`
#     layout — see `crates/headroom-py/pyproject.toml`).
#   * The headroom-ai install also writes files under `headroom/`.
#   * If headroom-ai is installed FIRST and the wheel goes second with
#     `--force-reinstall`, pip uninstalls the wheel's previously
#     installed files, deleting `headroom/__init__.py` (which the wheel
#     also claims). headroom-ai's __init__.py was already overwritten
#     by the wheel's empty stub at install-time, so the deletion leaves
#     no `__init__.py` at all — `from headroom._core import hello`
#     then fails with `ModuleNotFoundError: No module named
#     'headroom._core'`. Observed in PR #350 CI before this reorder.
#   * Installing the wheel FIRST means: wheel lays down stub
#     `__init__.py` + `_core.so`. Then headroom-ai install OVERWRITES
#     `__init__.py` with the real one and adds the rest of the
#     `headroom/` tree. `_core.so` survives because headroom-ai
#     doesn't claim ownership of it.
#
# uv already installed `maturin` as a transitive of the [proxy]/[code]
# extras; if it didn't, install it explicitly here so the build never
# silently skips.
COPY crates/ crates/
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/build/target \
    uv pip install --system maturin \
    && maturin build --release -m crates/headroom-py/Cargo.toml --out /build/wheels \
    && uv pip install --system --no-deps /build/wheels/headroom_core_py-*.whl

# Layer 3: copy real source, install headroom-ai (no deps). This
# overwrites the wheel's stub `headroom/__init__.py` with the real one
# and adds the full `headroom/` tree alongside the surviving
# `_core.so` from Layer 2.
COPY headroom/ headroom/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps --reinstall-package headroom-ai .

# Layer 4 (Hotfix-A0): verify the extension actually loads end-to-end
# inside the build image. If this fails, the runtime image would fail
# its lifespan smoke test on every restart — better to break the build
# loudly here than ship a broken image.
#
# IMPORTANT: run from `/tmp`, not from `/build`. `WORKDIR /build` puts
# `''` (cwd) at the front of `sys.path`, which makes `import headroom`
# resolve to `/build/headroom/` (the source tree we just COPY'd in)
# instead of `/usr/local/lib/python3.11/site-packages/headroom/` (where
# the wheel installed `_core.so`). The source tree has no `_core.so`,
# so the verify falsely fails. Production startup runs from a different
# cwd (the proxy's working directory or `/`), so this is a build-time-
# only quirk caused by `WORKDIR /build`. Anchoring the verify in `/tmp`
# matches the production import order: site-packages wins.
RUN cd /tmp && python -c "from headroom._core import hello; \
    marker = hello(); \
    assert marker == 'headroom-core', f'expected headroom-core, got {marker!r}'; \
    print(f'build-stage rust core verify OK: {marker}')"

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST} AS runtime-slim-base

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --link --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}
COPY --link --from=builder /usr/local/bin/headroom /usr/local/bin/headroom

RUN mkdir -p /home/nonroot /data && \
    if [ "$RUNTIME_USER" = "nonroot" ]; then \
      groupadd --gid 1000 nonroot && \
      useradd --uid 1000 --gid nonroot --create-home nonroot && \
      mkdir -p /home/nonroot/.headroom && \
      chown -R nonroot:nonroot /data /home/nonroot; \
    else \
      mkdir -p /root/.headroom; \
    fi

USER ${RUNTIME_USER}
WORKDIR /home/nonroot

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8787/readyz"]

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

FROM ${DISTROLESS_IMAGE}@${DISTROLESS_DIGEST} AS runtime-slim

ARG RUNTIME_USER=nonroot
ARG PYTHON_SITE_PACKAGES

COPY --link --from=builder ${PYTHON_SITE_PACKAGES} ${PYTHON_SITE_PACKAGES}

USER ${RUNTIME_USER}
WORKDIR /app

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHON_SITE_PACKAGES}

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/readyz', timeout=5)"]

ENTRYPOINT ["python3", "-m", "headroom.cli", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

# Default published image remains python-slim runtime
FROM runtime-slim-base AS runtime
