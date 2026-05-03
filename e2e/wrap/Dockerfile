FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    AIDER_CHAT_VERSION=0.86.2 \
    CODEX_VERSION=0.118.0 \
    OPENCLAW_VERSION=2026.4.7 \
    PATH="/root/.cargo/bin:/opt/headroom-venv/bin:/opt/aider-venv/bin:${PATH}" \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Single wheel architecture: `pip install -e .` invokes maturin under the
# hood which compiles the Rust extension. With rust + maturin available at
# build time, the image now ships the full Rust core — no
# `HEADROOM_REQUIRE_RUST_CORE=false` opt-out needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      pkg-config \
      libssl-dev \
      python3 \
      python3-pip \
      python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0 --profile minimal && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md uv.lock ./
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY headroom ./headroom
COPY crates ./crates
COPY sdk/typescript ./sdk/typescript
COPY plugins/openclaw ./plugins/openclaw

RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade pip && \
    /opt/headroom-venv/bin/python -m pip install -e ".[proxy]" && \
    /opt/headroom-venv/bin/python -c "from headroom._core import DiffCompressor; print('headroom._core OK')" && \
    python -m venv /opt/aider-venv && \
    /opt/aider-venv/bin/python -m pip install --upgrade pip && \
    /opt/aider-venv/bin/python -m pip install "aider-chat==${AIDER_CHAT_VERSION}" && \
    npm install -g --no-fund --no-audit "@openai/codex@${CODEX_VERSION}" "openclaw@${OPENCLAW_VERSION}"

COPY e2e/wrap ./e2e/wrap

CMD ["python", "e2e/wrap/run.py"]
