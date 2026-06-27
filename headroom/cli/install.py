"""Persistent install / deployment CLI commands."""

from __future__ import annotations

import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass

import click

from headroom._subprocess import run
from headroom.install.health import probe_json, probe_ready
from headroom.install.models import (
    ConfigScope,
    DeploymentManifest,
    InstallPreset,
    ProviderSelectionMode,
    RuntimeKind,
    SupervisorKind,
)
from headroom.install.planner import build_manifest
from headroom.install.providers import apply_mutations, revert_mutations
from headroom.install.runtime import (
    acquire_runtime_start_lock,
    run_foreground,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)
from headroom.install.state import delete_manifest, load_manifest, save_manifest
from headroom.install.supervisors import (
    install_supervisor,
    remove_supervisor,
    start_supervisor,
    stop_supervisor,
)

from .main import main


@dataclass(frozen=True)
class TurnkeyPlan:
    """Resolved runtime strategy for one-line deployments."""

    preset: str
    runtime: str
    supervisor_kind: str | None
    reason: str
    base_env: dict[str, str] | None = None


@main.group()
def install() -> None:
    """Install and manage persistent Headroom deployments."""


def _require_manifest(profile: str) -> DeploymentManifest:
    manifest = load_manifest(profile)
    if manifest is None:
        raise click.ClickException(f"No deployment profile named '{profile}' is installed.")
    return manifest


def _start_deployment(manifest: DeploymentManifest) -> None:
    if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
        start_persistent_docker(manifest)
    elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        start_supervisor(manifest)
    else:
        start_detached_agent(manifest.profile)

    if not wait_ready(manifest, timeout_seconds=45):
        raise click.ClickException(
            f"Deployment '{manifest.profile}' did not become ready after start."
        )


def _stop_deployment(manifest: DeploymentManifest) -> None:
    if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        stop_supervisor(manifest)
    stop_runtime(manifest)


def _remove_deployment(manifest: DeploymentManifest) -> None:
    try:
        _stop_deployment(manifest)
    except Exception:
        pass
    try:
        remove_supervisor(manifest)
    except Exception:
        pass
    try:
        revert_mutations(manifest)
    except Exception:
        pass
    delete_manifest(manifest.profile)


def _restore_deployment(manifest: DeploymentManifest) -> None:
    restored = deepcopy(manifest)
    restored.mutations = apply_mutations(restored)
    restored.artifacts = install_supervisor(restored)
    save_manifest(restored)
    _start_deployment(restored)


def _reject_task_lifecycle(manifest: DeploymentManifest, action: str) -> None:
    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        raise click.ClickException(
            f"Deployment '{manifest.profile}' uses persistent-task scheduling; "
            f"`headroom install {action}` is not supported for task deployments."
        )


def _command_available(command: str) -> bool:
    return shutil.which(command) is not None


def _probe_command(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return None


def _detect_nvidia_gpu_names() -> list[str]:
    if not _command_available("nvidia-smi"):
        return []
    result = _probe_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _docker_supports_nvidia_gpus() -> bool:
    if not _command_available("docker"):
        return False
    result = _probe_command(["docker", "info", "--format", "{{json .Runtimes}}"])
    if result is None or result.returncode != 0:
        return False
    return "nvidia" in result.stdout.lower()


def _select_turnkey_plan(*, prefer_docker: bool = True) -> TurnkeyPlan:
    """Choose the most portable available deployment strategy for this host."""

    gpu_names = _detect_nvidia_gpu_names()
    if prefer_docker and gpu_names and _docker_supports_nvidia_gpus():
        gpu_summary = ", ".join(gpu_names[:2])
        if len(gpu_names) > 2:
            gpu_summary += f", +{len(gpu_names) - 2} more"
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_DOCKER.value,
            runtime=RuntimeKind.DOCKER.value,
            supervisor_kind=SupervisorKind.NONE.value,
            reason=f"NVIDIA GPU available ({gpu_summary}); using Docker GPU passthrough.",
            base_env={"HEADROOM_DOCKER_GPUS": "all"},
        )

    if prefer_docker and _command_available("docker"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_DOCKER.value,
            runtime=RuntimeKind.DOCKER.value,
            supervisor_kind=SupervisorKind.NONE.value,
            reason="Docker is available, so Headroom can run in a restartable container.",
        )

    if sys.platform == "darwin" and _command_available("launchctl"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="launchd is available for scheduled health recovery.",
        )

    if sys.platform.startswith("win") and _command_available("schtasks"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="Windows Task Scheduler is available for scheduled health recovery.",
        )

    if sys.platform.startswith("linux") and _command_available("crontab"):
        return TurnkeyPlan(
            preset=InstallPreset.PERSISTENT_TASK.value,
            runtime=RuntimeKind.PYTHON.value,
            supervisor_kind=SupervisorKind.TASK.value,
            reason="cron is available for scheduled health recovery.",
        )

    return TurnkeyPlan(
        preset=InstallPreset.PERSISTENT_TASK.value,
        runtime=RuntimeKind.PYTHON.value,
        supervisor_kind=SupervisorKind.NONE.value,
        reason="No supported supervisor was detected; Headroom will start a managed detached runtime.",
    )


def _build_deployment_manifest(
    *,
    preset: str,
    runtime: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    image: str,
    supervisor_kind: str | None = None,
    extra_base_env: dict[str, str] | None = None,
) -> DeploymentManifest:
    manifest = build_manifest(
        profile=profile,
        preset=preset,
        runtime_kind=runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=list(targets),
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory_enabled=memory,
        telemetry_enabled=telemetry and not no_telemetry,
        image=image,
    )
    if supervisor_kind is not None:
        manifest.supervisor_kind = supervisor_kind
    if extra_base_env:
        manifest.base_env.update(extra_base_env)
    return manifest


def _apply_manifest(manifest: DeploymentManifest) -> None:
    existing = load_manifest(manifest.profile)
    if existing is not None:
        click.echo(f"Updating existing deployment profile '{manifest.profile}'...")
        _remove_deployment(existing)

    try:
        manifest.mutations = apply_mutations(manifest)
        manifest.artifacts = install_supervisor(manifest)
        save_manifest(manifest)
        _start_deployment(manifest)
    except Exception:
        _remove_deployment(manifest)
        if existing is not None:
            click.echo(f"Restoring previous deployment '{manifest.profile}'...")
            _restore_deployment(existing)
        raise


def _echo_installed(manifest: DeploymentManifest, *, prefix: str = "Installed persistent") -> None:
    click.echo(
        f"{prefix} deployment '{manifest.profile}' "
        f"({manifest.preset}, runtime={manifest.runtime_kind}, scope={manifest.scope})."
    )
    click.echo(f"Health: {manifest.health_url}")
    if manifest.targets:
        click.echo(f"Targets: {', '.join(manifest.targets)}")


@install.command("apply")
@click.option(
    "--preset",
    type=click.Choice([preset.value for preset in InstallPreset]),
    default=InstallPreset.PERSISTENT_SERVICE.value,
    show_default=True,
    help="Persistent runtime preset to install.",
)
@click.option(
    "--runtime",
    type=click.Choice([runtime.value for runtime in RuntimeKind]),
    default=RuntimeKind.PYTHON.value,
    show_default=True,
    help="Runtime used to execute Headroom for service/task modes.",
)
@click.option(
    "--scope",
    type=click.Choice([scope.value for scope in ConfigScope]),
    default=ConfigScope.USER.value,
    show_default=True,
    help="Where to apply persistent configuration.",
)
@click.option(
    "--providers",
    "provider_mode",
    type=click.Choice([mode.value for mode in ProviderSelectionMode]),
    default=ProviderSelectionMode.AUTO.value,
    show_default=True,
    help="Target selection mode for direct tool configuration.",
)
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(["claude", "copilot", "codex", "aider", "cursor", "openclaw", "opencode"]),
    help="Tool target to configure when --providers manual is used.",
)
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
@click.option(
    "--port", "-p", default=8787, type=int, show_default=True, help="Persistent proxy port."
)
@click.option(
    "--backend",
    default="anthropic",
    show_default=True,
    help="Proxy backend for the persistent runtime.",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backends when --backend anyllm is used.",
)
@click.option("--region", default=None, help="Cloud region for Bedrock / Vertex style backends.")
@click.option(
    "--mode", "proxy_mode", default="token", show_default=True, help="Proxy optimization mode."
)
@click.option("--memory", is_flag=True, help="Enable persistent memory in the proxy runtime.")
@click.option(
    "--telemetry",
    is_flag=True,
    help="Opt in to anonymous telemetry in the runtime (off by default).",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Force anonymous telemetry off in the runtime (already the default).",
)
@click.option(
    "--image",
    default="ghcr.io/chopratejas/headroom:latest",
    show_default=True,
    help="Docker image to use when runtime=docker or preset=persistent-docker.",
)
def install_apply(
    preset: str,
    runtime: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    image: str,
) -> None:
    """Install a persistent Headroom deployment."""

    if preset == InstallPreset.PERSISTENT_DOCKER.value:
        runtime = RuntimeKind.DOCKER.value

    manifest = _build_deployment_manifest(
        profile=profile,
        preset=preset,
        runtime=runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory=memory,
        telemetry=telemetry,
        no_telemetry=no_telemetry,
        image=image,
    )

    _apply_manifest(manifest)
    _echo_installed(manifest)


@main.command("deploy")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
@click.option(
    "--port", "-p", default=8787, type=int, show_default=True, help="Persistent proxy port."
)
@click.option(
    "--backend",
    default="anthropic",
    show_default=True,
    help="Proxy backend for the persistent runtime.",
)
@click.option(
    "--anyllm-provider",
    default=None,
    help="Provider for any-llm backends when --backend anyllm is used.",
)
@click.option("--region", default=None, help="Cloud region for Bedrock / Vertex style backends.")
@click.option(
    "--mode", "proxy_mode", default="token", show_default=True, help="Proxy optimization mode."
)
@click.option(
    "--scope",
    type=click.Choice([scope.value for scope in ConfigScope]),
    default=ConfigScope.USER.value,
    show_default=True,
    help="Where to apply persistent configuration.",
)
@click.option(
    "--providers",
    "provider_mode",
    type=click.Choice([mode.value for mode in ProviderSelectionMode]),
    default=ProviderSelectionMode.AUTO.value,
    show_default=True,
    help="Target selection mode for direct tool configuration.",
)
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(["claude", "copilot", "codex", "aider", "cursor", "openclaw", "opencode"]),
    help="Tool target to configure when --providers manual is used.",
)
@click.option("--memory", is_flag=True, help="Enable persistent memory in the proxy runtime.")
@click.option(
    "--telemetry",
    is_flag=True,
    help="Opt in to anonymous telemetry in the runtime (off by default).",
)
@click.option(
    "--no-telemetry",
    is_flag=True,
    help="Force anonymous telemetry off in the runtime (already the default).",
)
@click.option(
    "--image",
    default="ghcr.io/chopratejas/headroom:latest",
    show_default=True,
    help="Docker image to use when Docker is selected.",
)
@click.option(
    "--no-docker",
    is_flag=True,
    help="Use the native Python runtime even when Docker is installed.",
)
def deploy(
    profile: str,
    port: int,
    backend: str,
    anyllm_provider: str | None,
    region: str | None,
    proxy_mode: str,
    scope: str,
    provider_mode: str,
    targets: tuple[str, ...],
    memory: bool,
    telemetry: bool,
    no_telemetry: bool,
    image: str,
    no_docker: bool,
) -> None:
    """Deploy a turnkey local Headroom proxy and configure detected tools."""

    plan = _select_turnkey_plan(prefer_docker=not no_docker)
    click.echo(f"Selected {plan.preset} ({plan.runtime}): {plan.reason}")
    manifest = _build_deployment_manifest(
        profile=profile,
        preset=plan.preset,
        runtime=plan.runtime,
        scope=scope,
        provider_mode=provider_mode,
        targets=targets,
        port=port,
        backend=backend,
        anyllm_provider=anyllm_provider,
        region=region,
        proxy_mode=proxy_mode,
        memory=memory,
        telemetry=telemetry,
        no_telemetry=no_telemetry,
        image=image,
        supervisor_kind=plan.supervisor_kind,
        extra_base_env=plan.base_env,
    )
    _apply_manifest(manifest)
    _echo_installed(manifest, prefix="Deployed turnkey")


@install.command("status")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_status(profile: str) -> None:
    """Show persistent deployment status."""

    manifest = _require_manifest(profile)
    payload = probe_json(manifest.health_url.replace("/readyz", "/health"))
    click.echo(f"Profile:    {manifest.profile}")
    click.echo(f"Preset:     {manifest.preset}")
    click.echo(f"Runtime:    {manifest.runtime_kind}")
    click.echo(f"Supervisor: {manifest.supervisor_kind}")
    click.echo(f"Scope:      {manifest.scope}")
    click.echo(f"Port:       {manifest.port}")
    click.echo(f"Status:     {runtime_status(manifest)}")
    click.echo(f"Healthy:    {'yes' if probe_ready(manifest.health_url) else 'no'}")
    if payload and isinstance(payload, dict):
        click.echo(f"Health URL: {manifest.health_url.replace('/readyz', '/health')}")
        click.echo(f"Backend:    {payload.get('config', {}).get('backend', manifest.backend)}")


@install.command("start")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_start(profile: str) -> None:
    """Start a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "start")
    _start_deployment(manifest)
    click.echo(f"Started deployment '{profile}'.")


@install.command("stop")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_stop(profile: str) -> None:
    """Stop a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "stop")
    _stop_deployment(manifest)
    click.echo(f"Stopped deployment '{profile}'.")


@install.command("restart")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_restart(profile: str) -> None:
    """Restart a persistent deployment."""

    manifest = _require_manifest(profile)
    _reject_task_lifecycle(manifest, "restart")
    _stop_deployment(manifest)
    _start_deployment(manifest)
    click.echo(f"Restarted deployment '{profile}'.")


@install.command("remove")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_remove(profile: str) -> None:
    """Remove a persistent deployment and undo managed config."""

    manifest = _require_manifest(profile)
    try:
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            stop_supervisor(manifest)
    except Exception:
        pass
    try:
        stop_runtime(manifest)
    except Exception:
        pass
    try:
        remove_supervisor(manifest)
    except Exception:
        pass
    revert_mutations(manifest)
    delete_manifest(profile)
    click.echo(f"Removed deployment '{profile}'.")


@install.group("agent", hidden=True)
def install_agent() -> None:
    """Hidden runtime helpers used by persistent supervisors."""


@install_agent.command("run")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_agent_run(profile: str) -> None:
    """Run the persistent runtime in the foreground."""

    manifest = _require_manifest(profile)
    raise SystemExit(run_foreground(manifest))


_STARTUP_READY_TIMEOUT_SECONDS = 15


@install_agent.command("ensure")
@click.option("--profile", default="default", show_default=True, help="Deployment profile name.")
def install_agent_ensure(profile: str) -> None:
    """Ensure a persistent deployment is healthy, starting it when needed."""

    manifest = _require_manifest(profile)
    if probe_ready(manifest.health_url):
        click.echo(f"Deployment '{profile}' is already healthy.")
        return
    with acquire_runtime_start_lock(manifest.profile) as acquired:
        if not acquired:
            click.echo(f"Deployment '{profile}' start is already in progress.")
            return
        # Double-check after acquiring the lock — another ensure may have
        # started the runtime while we waited for the lock.
        if probe_ready(manifest.health_url):
            click.echo(f"Deployment '{profile}' is already healthy.")
            return
        if runtime_status(manifest) == "running":
            # Runtime exists but isn't ready yet — give it a grace period
            # before deciding it's wedged and restarting.
            if wait_ready(manifest, timeout_seconds=_STARTUP_READY_TIMEOUT_SECONDS):
                click.echo(f"Deployment '{profile}' is healthy.")
                return
            stop_runtime(manifest)
        _start_deployment(manifest)
    click.echo(f"Deployment '{profile}' is healthy.")
