"""Regression tests for lightweight package bootstrap."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_headroom_import_stays_lazy() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom

        print(json.dumps({
            "version": headroom.__version__,
            "cache_loaded": "headroom.cache" in sys.modules,
            "models_registry_loaded": "headroom.models.registry" in sys.modules,
            "memory_loaded": "headroom.memory" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    # Version is a non-empty string; don't hardcode a specific value.
    assert isinstance(data["version"], str) and data["version"]
    assert data["cache_loaded"] is False
    assert data["models_registry_loaded"] is False
    assert data["memory_loaded"] is False


def test_proxy_server_import_skips_litellm_backend() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.proxy.server

        print(json.dumps({
            "litellm_backend_loaded": "headroom.backends.litellm" in sys.modules,
            "anyllm_backend_loaded": "headroom.backends.anyllm" in sys.modules,
            "litellm_loaded": "litellm" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    assert data["litellm_backend_loaded"] is False
    assert data["anyllm_backend_loaded"] is False
    assert data["litellm_loaded"] is False
