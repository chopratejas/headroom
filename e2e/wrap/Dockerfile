FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    AIDER_CHAT_VERSION=0.86.2 \
    CODEX_VERSION=0.118.0 \
    OPENCLAW_VERSION=2026.4.7 \
    PATH="/opt/headroom-venv/bin:/opt/aider-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      python3 \
      python3-pip \
      python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md uv.lock ./
COPY headroom ./headroom
COPY sdk/typescript ./sdk/typescript
COPY plugins/openclaw ./plugins/openclaw

RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade pip && \
    /opt/headroom-venv/bin/python -m pip install -e ".[proxy]" && \
    python -m venv /opt/aider-venv && \
    /opt/aider-venv/bin/python -m pip install --upgrade pip && \
    /opt/aider-venv/bin/python -m pip install "aider-chat==${AIDER_CHAT_VERSION}" && \
    npm install -g --no-fund --no-audit "@openai/codex@${CODEX_VERSION}" "openclaw@${OPENCLAW_VERSION}"

COPY e2e/wrap ./e2e/wrap

CMD ["python", "e2e/wrap/run.py"]
