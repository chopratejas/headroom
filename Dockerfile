ARG PYTHON_VERSION=3.11
ARG DISTROLESS_IMAGE=gcr.io/distroless/python3-debian13

# ---- Build stage: compile native extensions, build wheel ----
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Layer 1: install deps only (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
# Stub package so uv can resolve the local extras without full source
RUN mkdir -p headroom && touch headroom/__init__.py
ARG HEADROOM_EXTRAS=proxy,code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[${HEADROOM_EXTRAS}]"

# Layer 2: copy real source, reinstall only headroom-ai (no deps)
COPY headroom/ headroom/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps --reinstall-package headroom-ai .

# ---- Runtime stage (python-slim): supports root/nonroot via build arg ----
FROM python:${PYTHON_VERSION}-slim AS runtime-slim-base

ARG RUNTIME_USER=nonroot

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/headroom /usr/local/bin/headroom

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

ENTRYPOINT ["headroom", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

FROM ${DISTROLESS_IMAGE} AS runtime-slim

ARG RUNTIME_USER=nonroot

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

USER ${RUNTIME_USER}
WORKDIR /app

ENV HEADROOM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages

EXPOSE 8787

ENTRYPOINT ["python3", "-m", "headroom.cli", "proxy"]
CMD ["--host", "0.0.0.0", "--port", "8787"]

# Default published image remains python-slim runtime
FROM runtime-slim-base AS runtime
