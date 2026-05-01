from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import click

import headroom
import headroom.cli.wrap as wrap_cli


class _Manifest:
    profile = "default"
    preset = "persistent-service"
    supervisor_kind = "service"
    health_url = "http://127.0.0.1:8787/readyz"


def test_ensure_proxy_recovers_matching_persistent_deployment(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_recovers_persistent_deployment_when_socket_is_bound(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_rejects_unhealthy_persistent_deployment(monkeypatch) -> None:
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: False)

    try:
        wrap_cli._ensure_proxy(8787, False)
    except click.ClickException as exc:
        assert "is not healthy" in str(exc)
    else:
        raise AssertionError("expected unhealthy persistent deployment to raise")


def test_ensure_proxy_falls_back_when_persistent_manifest_is_stale(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_start_proxy", lambda *args, **kwargs: calls.append("start"))

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start"]


def test_find_persistent_manifest_prefers_default_profile(monkeypatch) -> None:
    class DefaultManifest:
        profile = "default"
        port = 8787

    class OtherManifest:
        profile = "custom"
        port = 8787

    monkeypatch.setattr(
        "headroom.install.state.list_manifests",
        lambda: [OtherManifest(), DefaultManifest()],
    )

    manifest = wrap_cli._find_persistent_manifest(8787)

    assert manifest.profile == "default"


def test_recover_persistent_proxy_reuses_healthy_deployment(monkeypatch) -> None:
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)

    assert wrap_cli._recover_persistent_proxy(8787) is True


def test_recover_persistent_proxy_warns_for_task_deployment(monkeypatch) -> None:
    class TaskManifest(_Manifest):
        supervisor_kind = "task"

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: TaskManifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)

    assert wrap_cli._recover_persistent_proxy(8787) is False


def test_kill_proxy_by_pid_handles_permission_errors_and_escalation(monkeypatch) -> None:
    signals: list[int] = []
    proxy_checks = iter([True] * 50 + [False])

    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: next(proxy_checks))

    def fake_kill(pid: int, sig: int) -> None:
        signals.append(sig)

    monkeypatch.setattr(wrap_cli.os, "kill", fake_kill)

    assert wrap_cli._kill_proxy_by_pid(1234, 8787) is True
    assert signals == [
        wrap_cli.signal.SIGTERM,
        getattr(wrap_cli.signal, "SIGKILL", wrap_cli.signal.SIGTERM),
    ]

    def raise_permission(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(wrap_cli.os, "kill", raise_permission)
    assert wrap_cli._kill_proxy_by_pid(4321, 8787) is False


def test_recover_persistent_proxy_handles_docker_success_and_start_failures(monkeypatch) -> None:
    class DockerManifest(_Manifest):
        preset = "persistent-docker"
        supervisor_kind = "docker"

    starts: list[str] = []
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: DockerManifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.runtime.start_persistent_docker",
        lambda manifest: starts.append(manifest.profile),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )

    assert wrap_cli._recover_persistent_proxy(8787) is True
    assert starts == ["default"]

    monkeypatch.setattr(
        "headroom.install.runtime.start_persistent_docker",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert wrap_cli._recover_persistent_proxy(8787) is False


def test_make_cleanup_skips_shared_proxy_and_kills_stuck_process(monkeypatch) -> None:
    class FakeProc:
        def __init__(self, wait_raises: bool = False) -> None:
            self.terminated = False
            self.killed = False
            self._wait_raises = wait_raises

        def poll(self):  # noqa: ANN201
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> None:
            if self._wait_raises:
                raise subprocess.TimeoutExpired(cmd="proxy", timeout=timeout)

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{wrap_cli.os.getpid()}\n9999\n"),
    )
    shared_proc = FakeProc()
    wrap_cli._make_cleanup([shared_proc], 8787)()
    assert shared_proc.terminated is False

    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{wrap_cli.os.getpid()}\n"),
    )
    stuck_proc = FakeProc(wait_raises=True)
    wrap_cli._make_cleanup([stuck_proc], 8787)()
    assert stuck_proc.terminated is True
    assert stuck_proc.killed is True


def test_launch_tool_runs_cleanup_and_exits_with_subprocess_code(monkeypatch) -> None:
    cleanup_calls: list[str] = []
    signal_calls: list[tuple[int, object]] = []

    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: "proxy-proc")
    monkeypatch.setattr(wrap_cli, "_setup_code_graph", lambda verbose=False: True)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port=8787: lambda signum=None, frame=None: cleanup_calls.append("cleanup"),
    )
    monkeypatch.setattr(
        wrap_cli.signal,
        "signal",
        lambda sig, handler: signal_calls.append((sig, handler)),
    )
    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda cmd, env=None: SimpleNamespace(returncode=7),
    )

    try:
        wrap_cli._launch_tool(
            binary="codex",
            args=("--model", "gpt-5.4"),
            env={"PATH": "x"},
            port=8787,
            no_proxy=False,
            tool_label="Codex",
            env_vars_display=["OPENAI_BASE_URL=http://127.0.0.1:8787"],
            code_graph=True,
        )
    except SystemExit as exc:
        assert exc.code == 7
    else:
        raise AssertionError("expected launch to exit with wrapped subprocess return code")

    assert len(signal_calls) == 2
    assert cleanup_calls == ["cleanup"]


def test_launch_tool_converts_runtime_errors_into_exit_code_one(monkeypatch) -> None:
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_cli,
        "_make_cleanup",
        lambda holder, port=8787: lambda signum=None, frame=None: None,
    )
    monkeypatch.setattr(wrap_cli.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda cmd, env=None: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )

    try:
        wrap_cli._launch_tool(
            binary="codex",
            args=(),
            env={},
            port=8787,
            no_proxy=False,
            tool_label="Codex",
            env_vars_display=[],
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected launch failure to exit with code 1")


def test_misc_wrap_helpers_cover_telemetry_health_check_and_log_path(monkeypatch, tmp_path) -> None:
    telemetry_module = ModuleType("headroom.telemetry.beacon")
    telemetry_module.format_telemetry_notice = lambda prefix="": f"{prefix}notice"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.telemetry.beacon", telemetry_module)

    echoed: list[str] = []
    monkeypatch.setattr(wrap_cli.click, "echo", lambda message="": echoed.append(message))
    wrap_cli._print_telemetry_notice()
    assert echoed == ["  notice"]

    class GoodSocket:
        def __enter__(self):  # noqa: ANN201
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def settimeout(self, timeout: int) -> None:
            self.timeout = timeout

        def connect(self, address) -> None:  # noqa: ANN001
            self.address = address

    monkeypatch.setattr(wrap_cli.socket, "socket", lambda *args, **kwargs: GoodSocket())
    assert wrap_cli._check_proxy(8787) is True

    class BadSocket(GoodSocket):
        def connect(self, address) -> None:  # noqa: ANN001
            raise ConnectionRefusedError

    monkeypatch.setattr(wrap_cli.socket, "socket", lambda *args, **kwargs: BadSocket())
    assert wrap_cli._check_proxy(8787) is False

    paths_module = ModuleType("headroom.paths")
    paths_module.log_dir = lambda: tmp_path / "logs"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.paths", paths_module)
    monkeypatch.setattr(headroom, "paths", paths_module, raising=False)
    log_path = wrap_cli._get_log_path()
    assert log_path == tmp_path / "logs" / "proxy.log"
    assert log_path.parent.exists()


def test_start_proxy_success_failure_and_timeout(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "proxy.log"
    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda: log_path)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("HEADROOM_MODE", "token")
    monkeypatch.setenv("HEADROOM_BACKEND", "anyllm")
    monkeypatch.setenv("HEADROOM_ANYLLM_PROVIDER", "groq")
    monkeypatch.setenv("HEADROOM_REGION", "us-central1")

    class FakeProc:
        def __init__(self, poll_values=None, returncode=0) -> None:
            self._poll_values = iter(poll_values or [None, None])
            self.returncode = returncode
            self.killed = False

        def poll(self):  # noqa: ANN201
            return next(self._poll_values, None)

        def kill(self) -> None:
            self.killed = True

    popen_calls: list[dict[str, object]] = []

    def fake_popen(cmd, stdout=None, stderr=None, env=None):  # noqa: ANN001, ANN201
        popen_calls.append({"cmd": cmd, "stdout": stdout, "stderr": stderr, "env": env})
        return FakeProc()

    checks = iter([False, True])
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: next(checks))
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", fake_popen)
    proc = wrap_cli._start_proxy(
        8787,
        learn=True,
        memory=True,
        code_graph=True,
        agent_type="codex",
        openai_api_url="https://example.test/v1",
    )
    assert isinstance(proc, FakeProc)
    launched = popen_calls[0]
    assert "--mode" in launched["cmd"]
    assert "--learn" in launched["cmd"]
    assert "--memory" in launched["cmd"]
    assert "--code-graph" in launched["cmd"]
    assert "--backend" in launched["cmd"]
    assert "--anyllm-provider" in launched["cmd"]
    assert "--region" in launched["cmd"]
    assert "--openai-api-url" in launched["cmd"]
    assert launched["env"]["PYTHONIOENCODING"] == "utf-8"
    assert launched["env"]["HEADROOM_AGENT_TYPE"] == "codex"
    assert launched["env"]["HEADROOM_STACK"] == "wrap_codex"

    log_path.write_text("proxy crashed hard")
    crashing_proc = FakeProc(poll_values=[None, 1], returncode=1)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", lambda *args, **kwargs: crashing_proc)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    try:
        wrap_cli._start_proxy(8787)
    except RuntimeError as exc:
        assert "Proxy exited with code 1" in str(exc)
        assert "proxy crashed hard" in str(exc)
    else:
        raise AssertionError("expected crash path to raise")

    timed_out_proc = FakeProc(poll_values=[None] * 50)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", lambda *args, **kwargs: timed_out_proc)
    try:
        wrap_cli._start_proxy(8787)
    except RuntimeError as exc:
        assert "Proxy failed to start" in str(exc)
    else:
        raise AssertionError("expected timeout path to raise")
    assert timed_out_proc.killed is True


def test_ensure_proxy_upgrades_running_proxy_and_handles_restart_edge_cases(monkeypatch) -> None:
    started: list[dict[str, object]] = []

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_config",
        lambda port: {"memory": False, "learn": True, "code_graph": False, "pid": 1234},
    )
    monkeypatch.setattr(wrap_cli, "_kill_proxy_by_pid", lambda pid, port: True)
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda port, **kwargs: started.append({"port": port, **kwargs}) or "started-proxy",
    )

    result = wrap_cli._ensure_proxy(8787, False, memory=True, code_graph=True)
    assert result == "started-proxy"
    assert started == [
        {
            "port": 8787,
            "learn": True,
            "memory": True,
            "agent_type": "unknown",
            "code_graph": True,
            "backend": None,
            "anyllm_provider": None,
            "region": None,
            "openai_api_url": None,
        }
    ]

    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_config",
        lambda port: {"memory": False, "learn": False, "code_graph": False, "pid": None},
    )
    assert wrap_cli._ensure_proxy(8787, False, memory=True) is None

    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_config",
        lambda port: {"memory": False, "learn": False, "code_graph": False, "pid": 999},
    )
    monkeypatch.setattr(wrap_cli, "_kill_proxy_by_pid", lambda pid, port: False)
    try:
        wrap_cli._ensure_proxy(8787, False, memory=True)
    except click.ClickException as exc:
        assert "Failed to stop existing proxy" in str(exc)
    else:
        raise AssertionError("expected failed kill to raise")

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    assert wrap_cli._ensure_proxy(8787, True) is None


def test_setup_rtk_and_code_graph_paths(monkeypatch, tmp_path) -> None:
    rtk_module = ModuleType("headroom.rtk")
    installer_module = ModuleType("headroom.rtk.installer")
    paths = iter([None, tmp_path / "rtk", tmp_path / "rtk2"])
    rtk_module.get_rtk_path = lambda: next(paths)  # type: ignore[attr-defined]
    installer_module.ensure_rtk = lambda: tmp_path / "rtk"  # type: ignore[attr-defined]
    installer_module.register_claude_hooks = lambda path: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.rtk", rtk_module)
    monkeypatch.setitem(sys.modules, "headroom.rtk.installer", installer_module)
    assert wrap_cli._setup_rtk(verbose=True) == tmp_path / "rtk"

    installer_module.register_claude_hooks = lambda path: False  # type: ignore[attr-defined]
    assert wrap_cli._setup_rtk(verbose=False) == tmp_path / "rtk"

    installer_module.ensure_rtk = lambda: None  # type: ignore[attr-defined]
    rtk_module.get_rtk_path = lambda: None  # type: ignore[attr-defined]
    assert wrap_cli._setup_rtk(verbose=False) is None

    graph_module = ModuleType("headroom.graph.installer")
    graph_module.get_cbm_path = lambda: None  # type: ignore[attr-defined]
    graph_module.ensure_cbm = lambda: tmp_path / "cbm.exe"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.graph.installer", graph_module)

    registered: list[str] = []
    monkeypatch.setattr(
        wrap_cli, "_register_cbm_mcp_server", lambda cbm_bin: registered.append(cbm_bin)
    )
    monkeypatch.setattr(wrap_cli.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"nodes":12,"edges":34}\n',
            stderr="",
        ),
    )
    assert wrap_cli._setup_code_graph(verbose=True) is True
    assert registered == [str(tmp_path / "cbm.exe")]

    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="cbm", timeout=30)
        ),
    )
    assert wrap_cli._setup_code_graph(verbose=False) is False

    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="nope"),
    )
    assert wrap_cli._setup_code_graph(verbose=True) is False


def test_register_cbm_mcp_server_skips_existing_and_registers_new(monkeypatch) -> None:
    calls: list[list[str]] = []
    echoed: list[str] = []

    monkeypatch.setattr(
        wrap_cli.shutil, "which", lambda name: "claude" if name == "claude" else None
    )
    monkeypatch.setattr(wrap_cli.click, "echo", lambda message="": echoed.append(message))

    def fake_run(cmd, capture_output=True, text=True):  # noqa: ANN001, ANN201
        calls.append(cmd)
        if cmd[:3] == ["claude", "mcp", "get"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    wrap_cli._register_cbm_mcp_server("cbm-bin")
    assert calls[0][:3] == ["claude", "mcp", "get"]
    assert calls[1][:3] == ["claude", "mcp", "add"]
    assert any("registered codebase-memory-mcp MCP server" in message for message in echoed)

    calls.clear()
    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda cmd, capture_output=True, text=True: (
            calls.append(cmd) or SimpleNamespace(returncode=0)
        ),
    )
    wrap_cli._register_cbm_mcp_server("cbm-bin")
    assert len(calls) == 1

    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: None)
    wrap_cli._register_cbm_mcp_server("cbm-bin")
