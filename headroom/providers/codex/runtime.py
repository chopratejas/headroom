"""Runtime helpers for Codex/OpenAI-facing integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.proxy.helpers import WS_COMPRESSION_FAIL_OPEN_ENV

DEFAULT_API_URL = "https://api.openai.com"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenAI-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Codex through the local proxy."""
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env["OPENAI_BASE_URL"] = base_url
    # Auto-enable fail-open on WebSocket 1009 (Message Too Big / compression
    # timeout) so Codex never crashes with "Connection refused" on large
    # messages.  The proxy will pass the message through instead of dropping
    # the connection.
    env[WS_COMPRESSION_FAIL_OPEN_ENV] = "1"
    return env, [
        f"OPENAI_BASE_URL={base_url}",
        f"{WS_COMPRESSION_FAIL_OPEN_ENV}=1 (fail-open on WS 1009 compression timeout)",
    ]
