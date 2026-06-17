"""Runtime helpers for OpenCode-facing integrations."""

from __future__ import annotations


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"
