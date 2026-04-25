"""Codex install-time helpers."""

from __future__ import annotations

import re
from pathlib import Path

from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import codex_config_path

from .runtime import proxy_base_url

_CODEX_MARKER_START = "# --- Headroom persistent provider ---"
_CODEX_MARKER_END = "# --- end Headroom persistent provider ---"
_CODEX_PATTERN = re.compile(
    re.escape(_CODEX_MARKER_START) + r".*?" + re.escape(_CODEX_MARKER_END),
    re.DOTALL,
)


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Codex."""
    del backend
    return {"OPENAI_BASE_URL": proxy_base_url(port)}


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Apply Codex provider-scope configuration when requested."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    path = codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    section = (
        f"{_CODEX_MARKER_START}\n"
        'model_provider = "headroom"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom persistent proxy"\n'
        f'base_url = "{proxy_base_url(manifest.port)}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = true\n"
        f"{_CODEX_MARKER_END}\n"
    )
    if path.exists():
        existing = path.read_text()
        if _CODEX_MARKER_START in existing:
            merged = _CODEX_PATTERN.sub(section, existing)
        else:
            merged = existing.rstrip() + "\n\n" + section + "\n"
    else:
        merged = section + "\n"
    path.write_text(merged)
    return ManagedMutation(target=ToolTarget.CODEX.value, kind="toml-block", path=str(path))


def revert_provider_scope(mutation: ManagedMutation, manifest: DeploymentManifest) -> None:
    """Revert Codex provider-scope configuration."""
    del manifest
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return
    content = path.read_text()
    if _CODEX_MARKER_START not in content:
        return
    path.write_text(_CODEX_PATTERN.sub("", content).strip() + "\n")
