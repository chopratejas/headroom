FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/headroom-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      python3 \
      python3-pip \
      python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md uv.lock ./
COPY headroom ./headroom
COPY .claude-plugin ./.claude-plugin
COPY .github/plugin ./.github/plugin
COPY plugins/headroom-agent-hooks ./plugins/headroom-agent-hooks
# The init e2e harness imports from e2e._lib; both directories must be
# present and each must contain an __init__.py so Python sees them as
# packages rooted at /workspace.
COPY e2e/__init__.py ./e2e/__init__.py
COPY e2e/_lib ./e2e/_lib
COPY e2e/init ./e2e/init

RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade "pip<25" && \
    /opt/headroom-venv/bin/python -m pip install -e ".[proxy]"

CMD ["python", "e2e/init/run.py"]
