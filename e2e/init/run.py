from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from headroom.cli import init as init_cli

REPO_ROOT = Path("/workspace")
HEADROOM = "headroom"


def log(message: str) -> None:
    print(f"[init-e2e] {message}", flush=True)


def run(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_agent_shims(shim_dir: Path, log_path: Path) -> None:
    shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        from pathlib import Path

        record = {
            "tool": Path(sys.argv[0]).name,
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
        }
        log_path = Path(os.environ["HEADROOM_INIT_E2E_LOG"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        print(f"{record['tool']} shim executed")
        raise SystemExit(0)
        """
    )
    shim_dir.mkdir(parents=True, exist_ok=True)
    for name in ("claude", "copilot"):
        write_executable(shim_dir / name, shim)


def expect_hook_command(command: str, profile: str) -> None:
    assert_true("init hook ensure" in command, f"missing init hook ensure in: {command}")
    assert_true(f"--profile {profile}" in command, f"missing profile {profile} in: {command}")


def read_manifest(home_dir: Path, profile: str) -> dict[str, object]:
    path = home_dir / ".headroom" / "deploy" / profile / "manifest.json"
    assert_true(path.exists(), f"Expected manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_claude_local(home_dir: Path, project_dir: Path, shim_log: Path) -> None:
    settings = json.loads(
        (project_dir / ".claude" / "settings.local.json").read_text(encoding="utf-8")
    )
    assert_true(
        settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9011",
        "Claude local settings should point at the requested proxy port",
    )
    session_start = settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    pre_tool = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    profile = init_cli._local_profile(project_dir)
    expect_hook_command(session_start, profile)
    expect_hook_command(pre_tool, profile)

    manifest = read_manifest(home_dir, profile)
    assert_true("claude" in manifest["targets"], "Claude init should register the claude target")

    claude_calls = [record["argv"] for record in read_jsonl(shim_log) if record["tool"] == "claude"]
    assert_true(
        claude_calls
        == [
            ["plugin", "marketplace", "add", str(REPO_ROOT)],
            ["plugin", "install", "headroom@headroom-marketplace", "--scope", "local"],
        ],
        f"Unexpected Claude install commands: {claude_calls}",
    )


def verify_copilot_global(home_dir: Path, shim_log: Path) -> None:
    config = json.loads((home_dir / ".copilot" / "config.json").read_text(encoding="utf-8"))
    assert_true(
        "SessionStart" in config["hooks"], "Copilot config should include SessionStart hooks"
    )
    assert_true("PreToolUse" in config["hooks"], "Copilot config should include PreToolUse hooks")
    session_start = config["hooks"]["SessionStart"][0]["command"]
    expect_hook_command(session_start, "init-user")

    for shell_file in (home_dir / ".bashrc", home_dir / ".zshrc", home_dir / ".profile"):
        content = shell_file.read_text(encoding="utf-8")
        assert_true(
            'export COPILOT_PROVIDER_TYPE="openai"' in content,
            f"{shell_file.name} should contain the Copilot provider type",
        )
        assert_true(
            'export COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:9005/v1"' in content,
            f"{shell_file.name} should contain the Copilot provider base URL",
        )
        assert_true(
            'export COPILOT_PROVIDER_WIRE_API="completions"' in content,
            f"{shell_file.name} should contain the Copilot wire API",
        )

    copilot_calls = [
        record["argv"] for record in read_jsonl(shim_log) if record["tool"] == "copilot"
    ]
    assert_true(
        copilot_calls
        == [
            ["plugin", "marketplace", "add", str(REPO_ROOT)],
            ["plugin", "install", "headroom@headroom-marketplace"],
        ],
        f"Unexpected Copilot install commands: {copilot_calls}",
    )


def verify_codex_local(home_dir: Path, project_dir: Path) -> None:
    config_path = project_dir / ".codex" / "config.toml"
    hooks_path = project_dir / ".codex" / "hooks.json"
    config = config_path.read_text(encoding="utf-8")
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    profile = init_cli._local_profile(project_dir)

    assert_true(
        'base_url = "http://127.0.0.1:9012/v1"' in config,
        "Codex config should point at the requested proxy port",
    )
    assert_true(
        config.count("[features]") == 1, "Codex config should keep a single [features] table"
    )
    assert_true("codex_hooks = true" in config, "Codex config should enable codex_hooks")
    command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    expect_hook_command(command, profile)

    manifest = read_manifest(home_dir, profile)
    targets = manifest["targets"]
    assert_true(set(targets) == {"claude", "codex"}, f"Unexpected merged targets: {targets}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="headroom-init-e2e-") as temp_root_raw:
        temp_root = Path(temp_root_raw)
        home_dir = temp_root / "home"
        project_dir = temp_root / "project"
        shim_dir = temp_root / "bin"
        shim_log = temp_root / "shim-log.jsonl"
        home_dir.mkdir(parents=True)
        project_dir.mkdir(parents=True)
        create_agent_shims(shim_dir, shim_log)

        env = os.environ.copy()
        env["HOME"] = str(home_dir)
        env["USERPROFILE"] = str(home_dir)
        env["HEADROOM_INIT_E2E_LOG"] = str(shim_log)
        env["PATH"] = f"{shim_dir}:{env['PATH']}"

        run([HEADROOM, "init", "--port", "9011", "claude"], env=env, cwd=project_dir)
        verify_claude_local(home_dir, project_dir, shim_log)

        run(
            [
                HEADROOM,
                "init",
                "-g",
                "--port",
                "9005",
                "--backend",
                "openai",
                "copilot",
            ],
            env=env,
            cwd=project_dir,
        )
        verify_copilot_global(home_dir, shim_log)

        run([HEADROOM, "init", "--port", "9012", "codex"], env=env, cwd=project_dir)
        verify_codex_local(home_dir, project_dir)

        log("Init e2e completed successfully")


if __name__ == "__main__":
    main()
