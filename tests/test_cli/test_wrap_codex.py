"""Tests for `headroom wrap codex` and `headroom unwrap codex`.

These exercise the Codex-specific ``config.toml`` injection and restoration
helpers that route Codex through the Headroom proxy.  They are deliberately
end-to-end-ish: the unit tests call the helpers directly against a temp
``$HOME``, and the integration tests invoke the real Click commands the same
way a user would from the shell.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Unit tests: helpers operating on ~/.codex/config.toml
# ---------------------------------------------------------------------------


class TestStripCodexHeadroomBlocks:
    """Tests for the regex-based cleanup helper."""

    def test_empty_content_returns_empty(self) -> None:
        assert wrap_mod._strip_codex_headroom_blocks("") == ""

    def test_returns_content_unchanged_when_no_markers(self) -> None:
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        cleaned = wrap_mod._strip_codex_headroom_blocks(original)
        # Trailing whitespace normalization only — semantic content preserved.
        assert 'model = "gpt-4o"' in cleaned
        assert "[profiles.default]" in cleaned

    def test_removes_complete_headroom_block(self) -> None:
        wrapped = (
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            "\n"
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )
        assert wrap_mod._strip_codex_headroom_blocks(wrapped) == ""

    def test_preserves_user_content_around_block(self) -> None:
        user_pre = '[profiles.default]\nmodel = "gpt-4o"\n'
        user_post = '[mcp_servers.foo]\ncommand = "echo"\n'
        wrapped = (
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n" + user_pre + "\n"
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n" + user_post
        )
        cleaned = wrap_mod._strip_codex_headroom_blocks(wrapped)
        assert wrap_mod._CODEX_TOP_LEVEL_MARKER not in cleaned
        assert wrap_mod._CODEX_END_MARKER not in cleaned
        assert 'model = "gpt-4o"' in cleaned
        assert "[mcp_servers.foo]" in cleaned

    def test_removes_stray_top_level_model_provider_line(self) -> None:
        # Old wrap versions left `model_provider = "headroom"` outside markers.
        content = 'foo = 1\nmodel_provider = "headroom"\nbar = 2\n'
        cleaned = wrap_mod._strip_codex_headroom_blocks(content)
        assert 'model_provider = "headroom"' not in cleaned
        assert "foo = 1" in cleaned
        assert "bar = 2" in cleaned


class TestSnapshotCodexConfig:
    """Tests for ``_snapshot_codex_config_if_unwrapped``."""

    def test_creates_backup_on_first_call(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text('model = "gpt-4o"\n')

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert backup_file.exists()
        assert backup_file.read_text() == 'model = "gpt-4o"\n'

    def test_does_not_overwrite_existing_backup(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text("second-wrap content\n")
        backup_file.write_text("original-pre-wrap content\n")

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Backup must still contain the *original* pre-wrap content.
        assert backup_file.read_text() == "original-pre-wrap content\n"

    def test_no_backup_when_config_missing(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert not backup_file.exists()

    def test_no_backup_when_config_already_wrapped(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text(
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Pre-wrap snapshot must never snapshot an already-wrapped file.
        assert not backup_file.exists()


class TestInjectAndRestoreRoundTrip:
    """End-to-end wrap → unwrap cycle operating directly on a temp $HOME."""

    def test_wrap_unwrap_restores_empty_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_file = tmp_path / ".codex" / "config.toml"

        wrap_mod._inject_codex_provider_config(8787)
        assert config_file.exists()
        assert 'model_provider = "headroom"' in config_file.read_text()

        status, _ = wrap_mod._restore_codex_provider_config()
        # No prior config existed → the injected file is fully removed.
        assert status == "removed"
        assert not config_file.exists()
        assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()

    def test_wrap_unwrap_restores_prior_model_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = (
            'model_provider = "openai"\n'
            "\n"
            "[model_providers.openai]\n"
            'name = "OpenAI"\n'
            'base_url = "https://api.openai.com/v1"\n'
        )
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        wrapped = config_file.read_text()
        assert 'model_provider = "headroom"' in wrapped
        assert "[model_providers.headroom]" in wrapped

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text() == original
        assert not (config_dir / "config.toml.headroom-backup").exists()

    def test_wrap_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)  # port change

        content = config_file.read_text()
        # Exactly two Headroom blocks — a top-level-key block and the
        # provider-table block.  Re-wrapping must not duplicate them.
        assert content.count(wrap_mod._CODEX_TOP_LEVEL_MARKER) == 2
        assert content.count(wrap_mod._CODEX_END_MARKER) == 2
        # Latest port is honoured.
        assert 'base_url = "http://127.0.0.1:9999/v1"' in content
        assert 'base_url = "http://127.0.0.1:8787/v1"' not in content
        # User's original content is preserved.
        assert 'model = "gpt-4o"' in content

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text() == original

    def test_unwrap_is_noop_when_never_wrapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "noop"

    def test_unwrap_cleans_block_without_backup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Handles crash-case where wrap injected but backup was wiped."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        user_content = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(
            user_content + f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n\n'
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "cleaned"
        cleaned = config_file.read_text()
        assert wrap_mod._CODEX_TOP_LEVEL_MARKER not in cleaned
        assert wrap_mod._CODEX_END_MARKER not in cleaned
        assert 'model_provider = "headroom"' not in cleaned
        assert 'model = "gpt-4o"' in cleaned

    def test_unwrap_handles_malformed_prior_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unwrap preserves backup content verbatim — TOML validity isn't required."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        malformed = 'this is not valid toml ][ "" \x00\n'
        config_file.write_text(malformed)

        wrap_mod._inject_codex_provider_config(8787)
        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "restored"
        assert config_file.read_text() == malformed


# ---------------------------------------------------------------------------
# Integration tests: full `headroom wrap codex` / `headroom unwrap codex`
# ---------------------------------------------------------------------------


def test_wrap_codex_prepare_only_creates_backup_and_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = 'model_provider = "openai"\n'
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    assert 'model_provider = "headroom"' in config_file.read_text()
    backup = tmp_path / ".codex" / "config.toml.headroom-backup"
    assert backup.exists()
    assert backup.read_text() == original


def test_unwrap_codex_restores_prior_config_end_to_end(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bug report, reproduced: wrap → unwrap must round-trip cleanly."""
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = (
        "[profiles.default]\n"
        'model = "gpt-4o"\n'
        "\n"
        "[model_providers.openai]\n"
        'base_url = "https://api.openai.com/v1"\n'
    )
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output
    assert 'model_provider = "headroom"' in config_file.read_text()

    unwrap_result = runner.invoke(main, ["unwrap", "codex"])
    assert unwrap_result.exit_code == 0, unwrap_result.output

    # Config must be byte-for-byte what the user had before wrap, and the
    # injected block must be gone — no more "Missing OPENAI_API_KEY" when the
    # proxy is stopped.
    assert config_file.read_text() == original
    assert 'model_provider = "headroom"' not in config_file.read_text()
    assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()


def test_unwrap_codex_is_safe_noop_with_no_prior_wrap(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    result = runner.invoke(main, ["unwrap", "codex"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_unwrap_codex_removes_headroom_only_config_file(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output

    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()

    unwrap_result = runner.invoke(main, ["unwrap", "codex"])
    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert not config_file.exists()


def test_unwrap_codex_preserves_unrelated_sections(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    # A config with an MCP server the user configured by hand.
    original = '[mcp_servers.local_thing]\ncommand = "/usr/local/bin/thing"\nargs = ["--serve"]\n'
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    result = runner.invoke(main, ["unwrap", "codex"])
    assert result.exit_code == 0, result.output
    restored = config_file.read_text()
    assert restored == original


def test_inject_memory_mcp_config_creates_and_replaces_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        '[profiles.default]\nmodel = "gpt-5.4"\n'
        f"\n{wrap_mod._MEMORY_MCP_MARKER}\n"
        "[mcp_servers.headroom_memory]\n"
        'command = "python"\n'
        f"{wrap_mod._MEMORY_MCP_END}\n"
    )

    wrap_mod._inject_memory_mcp_config(r"C:\memory\db.sqlite", "user-123")

    content = config_file.read_text()
    assert content.count(wrap_mod._MEMORY_MCP_MARKER) == 1
    assert "headroom.memory.mcp_server" in content
    assert "C:/memory/db.sqlite" in content
    assert "user-123" in content


def test_inject_memory_agents_md_and_rtk_instructions_are_idempotent(tmp_path: Path) -> None:
    agents_file = tmp_path / "AGENTS.md"

    assert wrap_mod._inject_memory_agents_md(agents_file) is True
    first_agents = agents_file.read_text()
    assert wrap_mod._MEMORY_AGENTS_MARKER in first_agents

    assert wrap_mod._inject_memory_agents_md(agents_file) is True
    assert agents_file.read_text() == first_agents

    cursorrules = tmp_path / ".cursorrules"
    assert wrap_mod._inject_rtk_instructions(cursorrules, verbose=True) is True
    first_rtk = cursorrules.read_text()
    assert wrap_mod._RTK_MARKER in first_rtk

    assert wrap_mod._inject_rtk_instructions(cursorrules, verbose=True) is True
    assert cursorrules.read_text() == first_rtk


def test_memory_and_rtk_injection_append_existing_content_and_create_missing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    existing_config = tmp_path / ".codex" / "config.toml"
    existing_config.parent.mkdir(parents=True)
    existing_config.write_text('[profiles.default]\nmodel = "gpt-5.4"\n')
    wrap_mod._inject_memory_mcp_config(r"C:\db.sqlite", "user-1")
    appended = existing_config.read_text()
    assert "[profiles.default]" in appended
    assert wrap_mod._MEMORY_MCP_MARKER in appended

    missing_home = tmp_path / "second-home"
    _set_test_home(monkeypatch, missing_home)
    wrap_mod._inject_memory_mcp_config(r"C:\fresh.sqlite", "user-2")
    created_config = missing_home / ".codex" / "config.toml"
    assert created_config.exists()
    assert "fresh.sqlite" in created_config.read_text()

    agents_file = tmp_path / "nested" / "AGENTS.md"
    agents_file.parent.mkdir(parents=True)
    agents_file.write_text("# existing\n")
    assert wrap_mod._inject_memory_agents_md(agents_file) is True
    content = agents_file.read_text()
    assert content.startswith("# existing")
    assert wrap_mod._MEMORY_AGENTS_MARKER in content

    echoed: list[str] = []
    monkeypatch.setattr(wrap_mod.click, "echo", lambda message="": echoed.append(message))
    rules_file = tmp_path / ".cursorrules"
    rules_file.write_text("existing rules\n")
    assert wrap_mod._inject_rtk_instructions(rules_file, verbose=False) is True
    assert "existing rules" in rules_file.read_text()
    assert wrap_mod._RTK_MARKER in rules_file.read_text()

    assert wrap_mod._inject_rtk_instructions(rules_file, verbose=True) is True
    assert any("rtk instructions already" in message for message in echoed)


def test_wrap_codex_prepare_only_with_memory_imports_claude_memories(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "codex-user")
    events: list[object] = []

    fake_local = types.ModuleType("headroom.memory.backends.local")

    class FakeLocalBackendConfig:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    class FakeLocalBackend:
        def __init__(self, config: FakeLocalBackendConfig) -> None:
            events.append(("backend", config.db_path))

        async def _ensure_initialized(self) -> None:
            events.append("initialized")

        async def close(self) -> None:
            events.append("closed")

    fake_local.LocalBackend = FakeLocalBackend
    fake_local.LocalBackendConfig = FakeLocalBackendConfig

    fake_sync = types.ModuleType("headroom.memory.sync")

    async def fake_sync_import(backend, adapter, user):  # noqa: ANN001, ANN201
        events.append(("sync_import", adapter.memory_dir, user))
        return 4

    fake_sync.sync_import = fake_sync_import

    fake_claude = types.ModuleType("headroom.memory.sync_adapters.claude_code")

    class FakeClaudeCodeAdapter:
        def __init__(self, memory_dir: Path) -> None:
            self.memory_dir = memory_dir

    fake_claude.ClaudeCodeAdapter = FakeClaudeCodeAdapter
    fake_claude.get_claude_memory_dir = lambda: tmp_path / ".claude"

    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", fake_local)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync", fake_sync)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync_adapters.claude_code", fake_claude)

    with (
        patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")),
        patch("headroom.cli.wrap._inject_rtk_instructions") as inject_rtk,
        patch("headroom.cli.wrap._inject_memory_mcp_config") as inject_memory_mcp,
        patch("headroom.cli.wrap._inject_memory_agents_md") as inject_memory_agents,
        patch("headroom.cli.wrap._inject_codex_provider_config") as inject_provider,
    ):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--memory"])

    assert result.exit_code == 0, result.output
    inject_rtk.assert_any_call(tmp_path / "AGENTS.md", verbose=False)
    inject_rtk.assert_any_call(tmp_path / ".codex" / "AGENTS.md", verbose=False)
    inject_memory_mcp.assert_called_once_with(
        str(tmp_path / ".headroom" / "memory.db"), "codex-user"
    )
    inject_memory_agents.assert_called_once_with(tmp_path / "AGENTS.md")
    inject_provider.assert_called_once_with(8787)
    assert ("backend", str(tmp_path / ".headroom" / "memory.db")) in events
    assert ("sync_import", tmp_path / ".claude", "codex-user") in events
    assert "initialized" in events
    assert "closed" in events
    assert "Memory: imported 4 memories from Claude" in result.output


def test_wrap_codex_prepare_only_warns_when_memory_import_fails(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    fake_local = types.ModuleType("headroom.memory.backends.local")
    fake_local.LocalBackend = object
    fake_local.LocalBackendConfig = object

    fake_sync = types.ModuleType("headroom.memory.sync")
    fake_sync.sync_import = object()

    fake_claude = types.ModuleType("headroom.memory.sync_adapters.claude_code")
    fake_claude.ClaudeCodeAdapter = object

    def raise_memory_dir() -> Path:
        raise RuntimeError("claude import failed")

    fake_claude.get_claude_memory_dir = raise_memory_dir

    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", fake_local)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync", fake_sync)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync_adapters.claude_code", fake_claude)

    with (
        patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None),
        patch("headroom.cli.wrap._inject_memory_mcp_config"),
        patch("headroom.cli.wrap._inject_memory_agents_md"),
        patch("headroom.cli.wrap._inject_codex_provider_config"),
    ):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--memory", "--no-rtk"])

    assert result.exit_code == 0, result.output
    assert "Warning: Claude memory import failed: claude import failed" in result.output


def test_wrap_codex_reports_missing_binary_when_not_prepare_only(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--no-rtk"])

    assert result.exit_code == 1
    assert "Error: 'codex' not found in PATH." in result.output


def test_wrap_codex_launches_with_memory_reinjection(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "codex-user")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None),
        patch("headroom.cli.wrap.shutil.which", return_value="codex"),
        patch(
            "headroom.cli.wrap._build_codex_launch_env",
            return_value=(
                {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"},
                ["OPENAI_BASE_URL=http://127.0.0.1:8787/v1"],
            ),
        ),
        patch("headroom.cli.wrap._inject_codex_provider_config") as inject_provider,
        patch("headroom.cli.wrap._inject_memory_mcp_config") as inject_memory_mcp,
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main, ["wrap", "codex", "--memory", "--no-rtk", "--", "--model", "gpt-5.4"]
        )

    assert result.exit_code == 0, result.output
    inject_provider.assert_called_once_with(8787)
    inject_memory_mcp.assert_any_call(str(tmp_path / ".headroom" / "memory.db"), "codex-user")
    assert captured["binary"] == "codex"
    assert captured["args"] == ("--model", "gpt-5.4")
    assert captured["tool_label"] == "CODEX"
    assert captured["agent_type"] == "codex"


def test_wrap_codex_prepare_only_memory_without_imported_entries_is_quiet(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)

    fake_local = types.ModuleType("headroom.memory.backends.local")

    class FakeLocalBackendConfig:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    class FakeLocalBackend:
        def __init__(self, config: FakeLocalBackendConfig) -> None:
            self.config = config

        async def _ensure_initialized(self) -> None:
            return None

        async def close(self) -> None:
            return None

    fake_local.LocalBackend = FakeLocalBackend
    fake_local.LocalBackendConfig = FakeLocalBackendConfig

    fake_sync = types.ModuleType("headroom.memory.sync")

    async def fake_sync_import(backend, adapter, user):  # noqa: ANN001, ANN201
        return 0

    fake_sync.sync_import = fake_sync_import

    fake_claude = types.ModuleType("headroom.memory.sync_adapters.claude_code")
    fake_claude.ClaudeCodeAdapter = lambda memory_dir: object()
    fake_claude.get_claude_memory_dir = lambda: tmp_path / ".claude"

    monkeypatch.setitem(sys.modules, "headroom.memory.backends.local", fake_local)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync", fake_sync)
    monkeypatch.setitem(sys.modules, "headroom.memory.sync_adapters.claude_code", fake_claude)

    with (
        patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None),
        patch("headroom.cli.wrap._inject_memory_mcp_config"),
        patch("headroom.cli.wrap._inject_memory_agents_md"),
        patch("headroom.cli.wrap._inject_codex_provider_config"),
    ):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--memory", "--no-rtk"])

    assert result.exit_code == 0, result.output
    assert "Memory: imported" not in result.output
