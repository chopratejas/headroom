from __future__ import annotations

from headroom.providers.aider.install import build_install_env
from headroom.providers.aider.runtime import build_launch_env


def test_aider_build_launch_env_sets_proxy_urls_without_mutating_input() -> None:
    # Arrange
    source_env = {"EXISTING": "value"}

    # Act
    env, lines = build_launch_env(port=9999, environ=source_env)

    # Assert
    assert source_env == {"EXISTING": "value"}
    assert env["EXISTING"] == "value"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9999/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    assert lines == [
        "OPENAI_API_BASE=http://127.0.0.1:9999/v1",
        "ANTHROPIC_BASE_URL=http://127.0.0.1:9999",
    ]


def test_aider_build_install_env_returns_only_persistent_proxy_variables() -> None:
    # Arrange / Act
    env = build_install_env(port=8787, backend="ignored")

    # Assert
    assert env == {
        "OPENAI_API_BASE": "http://127.0.0.1:8787/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    }
