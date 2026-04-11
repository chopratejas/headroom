"""Tool-target configuration for persistent deployments."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import click

from .models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from .paths import (
    claude_settings_path,
    codex_config_path,
    openclaw_config_path,
    unix_system_env_targets,
    unix_user_env_targets,
)
from .runtime import resolve_headroom_command

_ENV_MARKER_START = "# >>> headroom persistent env >>>"
_ENV_MARKER_END = "# <<< headroom persistent env <<<"
_ENV_PATTERN = re.compile(
    re.escape(_ENV_MARKER_START) + r".*?" + re.escape(_ENV_MARKER_END),
    re.DOTALL,
)
_CODEX_MARKER_START = "# --- Headroom persistent provider ---"
_CODEX_MARKER_END = "# --- end Headroom persistent provider ---"
_CODEX_PATTERN = re.compile(
    re.escape(_CODEX_MARKER_START) + r".*?" + re.escape(_CODEX_MARKER_END),
    re.DOTALL,
)


def _merge_marker_block(file_path: Path, block: str, pattern: re.Pattern[str], marker: str) -> str:
    if file_path.exists():
        existing = file_path.read_text()
        if marker in existing:
            return pattern.sub(block, existing)
        return existing.rstrip() + "\n\n" + block + "\n"
    return block + "\n"


def _env_block(values: dict[str, str]) -> str:
    lines = [_ENV_MARKER_START]
    for name, value in values.items():
        lines.append(f'export {name}="{value}"')
    lines.append(_ENV_MARKER_END)
    return "\n".join(lines)


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _unix_scope_values(manifest: DeploymentManifest) -> dict[str, str]:
    merged = dict(manifest.base_env)
    for env_map in manifest.tool_envs.values():
        merged.update(env_map)
    return merged


def _apply_unix_env_scope(manifest: DeploymentManifest) -> list[ManagedMutation]:
    values = _unix_scope_values(manifest)
    block = _env_block(values)
    if manifest.scope == ConfigScope.USER.value:
        targets = unix_user_env_targets()
    else:
        targets = unix_system_env_targets()
    mutations: list[ManagedMutation] = []
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = _merge_marker_block(path, block, _ENV_PATTERN, _ENV_MARKER_START)
        path.write_text(merged)
        mutations.append(ManagedMutation(target="env", kind="shell-block", path=str(path)))
    return mutations


def _remove_unix_env_scope(mutations: list[ManagedMutation]) -> None:
    for mutation in mutations:
        if mutation.kind != "shell-block" or not mutation.path:
            continue
        path = Path(mutation.path)
        if not path.exists():
            continue
        content = path.read_text()
        if _ENV_MARKER_START not in content:
            continue
        path.write_text(_ENV_PATTERN.sub("", content).strip() + "\n")


def _apply_windows_env_scope(manifest: DeploymentManifest) -> list[ManagedMutation]:
    scope_name = "Machine" if manifest.scope == ConfigScope.SYSTEM.value else "User"
    merged = _unix_scope_values(manifest)
    mutations: list[ManagedMutation] = []
    for name, value in merged.items():
        previous = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$value = [Environment]::GetEnvironmentVariable({_powershell_literal(name)},{_powershell_literal(scope_name)}); "
                "if ($null -eq $value) { '__HEADROOM_UNSET__' } else { $value }",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[Environment]::SetEnvironmentVariable({_powershell_literal(name)},{_powershell_literal(value)},{_powershell_literal(scope_name)})",
        ]
        subprocess.run(command, check=True)
        mutations.append(
            ManagedMutation(
                target="env",
                kind="windows-env",
                data={
                    "name": name,
                    "scope": scope_name,
                    "previous": None if previous == "__HEADROOM_UNSET__" else previous,
                },
            )
        )
    return mutations


def _remove_windows_env_scope(mutations: list[ManagedMutation]) -> None:
    for mutation in mutations:
        if mutation.kind != "windows-env":
            continue
        name = mutation.data.get("name")
        if not isinstance(name, str):
            raise ValueError("Windows environment mutation is missing a variable name")
        scope_name = mutation.data.get("scope", "User")
        if not isinstance(scope_name, str):
            raise ValueError("Windows environment mutation is missing a valid scope")
        previous = mutation.data.get("previous")
        if previous is None:
            value_literal = "$null"
        else:
            value_literal = _powershell_literal(previous)
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[Environment]::SetEnvironmentVariable({_powershell_literal(name)},{value_literal},{_powershell_literal(scope_name)})",
        ]
        subprocess.run(command, check=True)


def _apply_claude_provider_scope(manifest: DeploymentManifest) -> ManagedMutation:
    path = claude_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text())
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    previous = {
        name: env_map.get(name) for name in manifest.tool_envs.get(ToolTarget.CLAUDE.value, {})
    }
    env_map.update(manifest.tool_envs[ToolTarget.CLAUDE.value])
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return ManagedMutation(
        target=ToolTarget.CLAUDE.value,
        kind="json-env",
        path=str(path),
        data={"previous": previous},
    )


def _revert_claude_provider_scope(mutation: ManagedMutation, values: dict[str, str]) -> None:
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    previous: dict[str, object] = mutation.data.get("previous", {})
    for name in values:
        if previous.get(name) is None:
            env_map.pop(name, None)
        else:
            env_map[name] = previous[name]
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _apply_codex_provider_scope(manifest: DeploymentManifest) -> ManagedMutation:
    path = codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    section = (
        f"{_CODEX_MARKER_START}\n"
        'model_provider = "headroom"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom persistent proxy"\n'
        f'base_url = "http://127.0.0.1:{manifest.port}/v1"\n'
        'env_key = "OPENAI_API_KEY"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = true\n"
        f"{_CODEX_MARKER_END}\n"
    )
    merged = _merge_marker_block(path, section, _CODEX_PATTERN, _CODEX_MARKER_START)
    path.write_text(merged)
    return ManagedMutation(target=ToolTarget.CODEX.value, kind="toml-block", path=str(path))


def _revert_codex_provider_scope(mutation: ManagedMutation) -> None:
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return
    content = path.read_text()
    if _CODEX_MARKER_START not in content:
        return
    path.write_text(_CODEX_PATTERN.sub("", content).strip() + "\n")


def _invoke_openclaw(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _apply_openclaw_provider_scope(manifest: DeploymentManifest) -> ManagedMutation:
    if not shutil_which("openclaw"):
        raise click.ClickException("openclaw not found in PATH; cannot apply provider scope.")
    command = [
        *resolve_headroom_command(),
        "wrap",
        "openclaw",
        "--no-auto-start",
        "--proxy-port",
        str(manifest.port),
    ]
    _invoke_openclaw(command)
    return ManagedMutation(
        target=ToolTarget.OPENCLAW.value, kind="openclaw-wrap", path=str(openclaw_config_path())
    )


def _revert_openclaw_provider_scope() -> None:
    if not shutil_which("openclaw"):
        return
    command = [*resolve_headroom_command(), "unwrap", "openclaw"]
    _invoke_openclaw(command)


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def apply_mutations(manifest: DeploymentManifest) -> list[ManagedMutation]:
    """Apply provider/user/system configuration for a deployment."""

    mutations: list[ManagedMutation] = []
    if manifest.scope in {ConfigScope.USER.value, ConfigScope.SYSTEM.value}:
        if os.name == "nt":
            mutations.extend(_apply_windows_env_scope(manifest))
        else:
            mutations.extend(_apply_unix_env_scope(manifest))
        if ToolTarget.OPENCLAW.value in manifest.targets:
            mutations.append(_apply_openclaw_provider_scope(manifest))
        return mutations

    if ToolTarget.CLAUDE.value in manifest.targets:
        mutations.append(_apply_claude_provider_scope(manifest))
    if ToolTarget.CODEX.value in manifest.targets:
        mutations.append(_apply_codex_provider_scope(manifest))
    if ToolTarget.OPENCLAW.value in manifest.targets:
        mutations.append(_apply_openclaw_provider_scope(manifest))
    return mutations


def revert_mutations(manifest: DeploymentManifest) -> None:
    """Undo the stored mutations for a deployment."""

    if manifest.scope in {ConfigScope.USER.value, ConfigScope.SYSTEM.value}:
        shell_mutations = [m for m in manifest.mutations if m.target == "env"]
        if os.name == "nt":
            _remove_windows_env_scope(shell_mutations)
        else:
            _remove_unix_env_scope(shell_mutations)

    for mutation in manifest.mutations:
        if mutation.target == ToolTarget.CLAUDE.value and mutation.kind == "json-env":
            _revert_claude_provider_scope(
                mutation, manifest.tool_envs.get(ToolTarget.CLAUDE.value, {})
            )
        elif mutation.target == ToolTarget.CODEX.value and mutation.kind == "toml-block":
            _revert_codex_provider_scope(mutation)
        elif mutation.target == ToolTarget.OPENCLAW.value and mutation.kind == "openclaw-wrap":
            _revert_openclaw_provider_scope()
