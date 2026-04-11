from __future__ import annotations

from pathlib import Path

from headroom.install.models import DeploymentManifest
from headroom.install.runtime import build_runtime_command


def test_build_runtime_command_for_docker_includes_deployment_env(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "HEADROOM_DEPLOYMENT_PROFILE=default" in joined
    assert "HEADROOM_DEPLOYMENT_PRESET=persistent-docker" in joined
    assert "127.0.0.1:8787:8787" in joined
    assert "ghcr.io/chopratejas/headroom:latest" in command


def test_build_runtime_command_for_docker_matches_wrapper_parity(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    assert (tmp_path / ".headroom").is_dir()
    assert (tmp_path / ".claude").is_dir()
    assert (tmp_path / ".codex").is_dir()
    assert (tmp_path / ".gemini").is_dir()
    assert "--env" in command
    joined = " ".join(command)
    assert "ANTHROPIC_API_KEY" in joined
    assert "OPENAI_API_KEY" in joined
