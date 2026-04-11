from __future__ import annotations

import json
from pathlib import Path

from headroom.install.models import DeploymentManifest
from headroom.install.providers import (
    _apply_claude_provider_scope,
    _apply_codex_provider_scope,
    _apply_windows_env_scope,
    _remove_windows_env_scope,
    _revert_claude_provider_scope,
    _revert_codex_provider_scope,
)


def _manifest(tmp_path: Path) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="provider",
        provider_mode="manual",
        targets=["claude", "codex"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={
            "claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"},
            "codex": {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"},
        },
    )


def test_apply_and_revert_claude_provider_scope(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "keep", "ANTHROPIC_BASE_URL": "https://old"}})
    )
    monkeypatch.setattr("headroom.install.providers.claude_settings_path", lambda: settings_path)
    manifest = _manifest(tmp_path)

    mutation = _apply_claude_provider_scope(manifest)
    payload = json.loads(settings_path.read_text())
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert payload["env"]["ANTHROPIC_API_KEY"] == "keep"

    _revert_claude_provider_scope(mutation, manifest.tool_envs["claude"])
    reverted = json.loads(settings_path.read_text())
    assert reverted["env"]["ANTHROPIC_BASE_URL"] == "https://old"
    assert reverted["env"]["ANTHROPIC_API_KEY"] == "keep"


def test_apply_and_revert_codex_provider_scope(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.install.providers.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    mutation = _apply_codex_provider_scope(manifest)
    content = config_path.read_text()
    assert 'model_provider = "headroom"' in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content

    _revert_codex_provider_scope(mutation)
    reverted = config_path.read_text()
    assert 'model_provider = "headroom"' not in reverted
    assert reverted.strip() == 'model = "gpt-4o"'


def test_apply_openclaw_provider_scope_uses_manifest_port(monkeypatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr("headroom.install.providers.shutil_which", lambda name: "openclaw")
    monkeypatch.setattr(
        "headroom.install.providers.resolve_headroom_command",
        lambda: ["headroom"],
    )
    monkeypatch.setattr(
        "headroom.install.providers._invoke_openclaw",
        lambda command: recorded.append(command),
    )
    monkeypatch.setattr(
        "headroom.install.providers.openclaw_config_path",
        lambda: tmp_path / "openclaw.json",
    )
    manifest = _manifest(tmp_path)
    manifest.port = 9999

    from headroom.install.providers import _apply_openclaw_provider_scope

    _apply_openclaw_provider_scope(manifest)

    assert recorded == [["headroom", "wrap", "openclaw", "--no-auto-start", "--proxy-port", "9999"]]


def test_windows_env_scope_restores_previous_values(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.scope = "user"
    manifest.targets = ["claude"]
    manifest.base_env = {"HEADROOM_PORT": "8787"}
    manifest.tool_envs = {"claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}

    calls: list[list[str]] = []
    previous_values = {
        "HEADROOM_PORT": "7777",
        "ANTHROPIC_BASE_URL": "https://old",
    }

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        script = command[-1]
        if "GetEnvironmentVariable" in script:
            name = script.split("GetEnvironmentVariable('", 1)[1].split("'", 1)[0]
            value = previous_values.get(name, "__HEADROOM_UNSET__")
            return Result(stdout=value)
        return Result()

    monkeypatch.setattr("headroom.install.providers.subprocess.run", fake_run)

    mutations = _apply_windows_env_scope(manifest)
    _remove_windows_env_scope(mutations)

    previous_by_name = {mutation.data["name"]: mutation.data["previous"] for mutation in mutations}
    assert previous_by_name["HEADROOM_PORT"] == "7777"
    assert previous_by_name["ANTHROPIC_BASE_URL"] == "https://old"
    assert any(
        "[Environment]::SetEnvironmentVariable('HEADROOM_PORT','7777','User')" in command[-1]
        for command in calls
    )
    assert any(
        "[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL','https://old','User')"
        in command[-1]
        for command in calls
    )
