ARG VARIANT=3.12-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}

# Single-wheel architecture (post-#355): `uv sync` builds `headroom-ai`
# from the local pyproject.toml using maturin (declared in build-system).
# Maturin needs rust + cargo + pkg-config + libssl-dev (transitive
# openssl-sys via fastembed/hf-hub).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      pkg-config \
      libssl-dev && \
    rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0 --profile minimal && \
    rustc --version && cargo --version

RUN python -m pip install --no-cache-dir uv==0.6.17 'maturin>=1.5,<2.0'
