# ---- Build stage: compile native extensions, build wheel ----
FROM python:3.11-slim AS builder

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Layer 1: install deps only (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
# Stub package so uv can resolve the local ".[proxy]" without full source
RUN mkdir -p headroom && touch headroom/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[proxy]"

# Layer 2: copy real source, reinstall only headroom-ai (no deps)
COPY headroom/ headroom/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps --reinstall-package headroom-ai .

# ---- Runtime stage: minimal image with only what's needed ----
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 headroom && \
    useradd --uid 1000 --gid headroom --create-home headroom

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom

RUN mkdir -p /data /home/headroom/.headroom && \
    chown -R headroom:headroom /data /home/headroom/.headroom

USER headroom
WORKDIR /home/headroom

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8787

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]
