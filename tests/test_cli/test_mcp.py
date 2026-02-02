"""Integration tests for MCP CLI commands.

These are real tests that:
- Actually write/read config files
- Test actual CLI behavior
- Test MCP server initialization
"""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.cli.mcp import (
    get_headroom_command,
    load_mcp_config,
    save_mcp_config,
)


@pytest.fixture
def temp_claude_dir(tmp_path):
    """Create a temporary .claude directory for testing."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    return claude_dir


@pytest.fixture
def mock_claude_config_path(temp_claude_dir):
    """Patch the MCP config path to use temp directory."""
    config_path = temp_claude_dir / "mcp.json"
    with patch("headroom.cli.mcp.MCP_CONFIG_PATH", config_path):
        with patch("headroom.cli.mcp.CLAUDE_CONFIG_DIR", temp_claude_dir):
            yield config_path


class TestMCPConfigFunctions:
    """Test config file handling functions."""

    def test_get_headroom_command_returns_list(self):
        """Command should be a list suitable for subprocess."""
        cmd = get_headroom_command()
        assert isinstance(cmd, list)
        assert len(cmd) >= 1
        # Should end with mcp serve args
        assert "mcp" in cmd or "-m" in cmd

    def test_load_mcp_config_empty_when_no_file(self, mock_claude_config_path):
        """Loading non-existent config returns empty structure."""
        config = load_mcp_config()
        assert config == {"mcpServers": {}}

    def test_save_and_load_config(self, mock_claude_config_path):
        """Config can be saved and loaded back."""
        test_config = {
            "mcpServers": {
                "headroom": {
                    "command": "headroom",
                    "args": ["mcp", "serve"],
                }
            }
        }
        save_mcp_config(test_config)

        # File should exist
        assert mock_claude_config_path.exists()

        # Load it back
        loaded = load_mcp_config()
        assert loaded == test_config

    def test_save_config_creates_directory(self, tmp_path):
        """save_mcp_config creates parent directory if needed."""
        claude_dir = tmp_path / "new_dir" / ".claude"
        config_path = claude_dir / "mcp.json"

        with patch("headroom.cli.mcp.MCP_CONFIG_PATH", config_path):
            with patch("headroom.cli.mcp.CLAUDE_CONFIG_DIR", claude_dir):
                save_mcp_config({"mcpServers": {}})

        assert config_path.exists()

    def test_load_config_preserves_other_servers(self, mock_claude_config_path):
        """Loading preserves other MCP servers in config."""
        # Write config with another server
        existing_config = {
            "mcpServers": {
                "other-server": {"command": "other", "args": []},
            }
        }
        mock_claude_config_path.write_text(json.dumps(existing_config))

        loaded = load_mcp_config()
        assert "other-server" in loaded["mcpServers"]


class TestMCPInstallCommand:
    """Test 'headroom mcp install' command."""

    def test_install_creates_config(self, mock_claude_config_path):
        """Install creates MCP config file."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "install"])

        assert result.exit_code == 0
        assert "installed" in result.output.lower()
        assert mock_claude_config_path.exists()

        # Verify config content
        config = json.loads(mock_claude_config_path.read_text())
        assert "headroom" in config["mcpServers"]
        assert config["mcpServers"]["headroom"]["command"] == "headroom"
        assert "mcp" in config["mcpServers"]["headroom"]["args"]
        assert "serve" in config["mcpServers"]["headroom"]["args"]

    def test_install_preserves_other_servers(self, mock_claude_config_path):
        """Install preserves existing MCP servers."""
        # Create config with another server
        existing_config = {
            "mcpServers": {
                "github": {"command": "github-mcp", "args": []},
            }
        }
        mock_claude_config_path.write_text(json.dumps(existing_config))

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "install"])

        assert result.exit_code == 0

        # Both servers should exist
        config = json.loads(mock_claude_config_path.read_text())
        assert "github" in config["mcpServers"]
        assert "headroom" in config["mcpServers"]

    def test_install_with_custom_proxy_url(self, mock_claude_config_path):
        """Install with custom proxy URL sets env var."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "install", "--proxy-url", "http://localhost:9000"])

        assert result.exit_code == 0

        config = json.loads(mock_claude_config_path.read_text())
        assert (
            config["mcpServers"]["headroom"]["env"]["HEADROOM_PROXY_URL"] == "http://localhost:9000"
        )

    def test_install_default_proxy_url_no_env(self, mock_claude_config_path):
        """Install with default proxy URL doesn't set env var."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "install"])

        assert result.exit_code == 0

        config = json.loads(mock_claude_config_path.read_text())
        # No env section for default URL
        assert "env" not in config["mcpServers"]["headroom"]

    def test_install_already_configured_no_force(self, mock_claude_config_path):
        """Install without --force when already configured exits cleanly."""
        # First install
        runner = CliRunner()
        runner.invoke(main, ["mcp", "install"])

        # Second install without force
        result = runner.invoke(main, ["mcp", "install"])

        assert result.exit_code == 0
        assert "already configured" in result.output.lower()

    def test_install_force_overwrites(self, mock_claude_config_path):
        """Install with --force overwrites existing config."""
        runner = CliRunner()
        runner.invoke(main, ["mcp", "install", "--proxy-url", "http://old:8787"])

        # Force install with new URL
        result = runner.invoke(
            main, ["mcp", "install", "--force", "--proxy-url", "http://new:9000"]
        )

        assert result.exit_code == 0
        assert "installed" in result.output.lower()

        config = json.loads(mock_claude_config_path.read_text())
        assert config["mcpServers"]["headroom"]["env"]["HEADROOM_PROXY_URL"] == "http://new:9000"


class TestMCPUninstallCommand:
    """Test 'headroom mcp uninstall' command."""

    def test_uninstall_removes_headroom(self, mock_claude_config_path):
        """Uninstall removes headroom from config."""
        # First install
        runner = CliRunner()
        runner.invoke(main, ["mcp", "install"])

        # Then uninstall
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower()

        config = json.loads(mock_claude_config_path.read_text())
        assert "headroom" not in config["mcpServers"]

    def test_uninstall_preserves_other_servers(self, mock_claude_config_path):
        """Uninstall preserves other MCP servers."""
        # Create config with headroom and another server
        config = {
            "mcpServers": {
                "headroom": {"command": "headroom", "args": ["mcp", "serve"]},
                "github": {"command": "github-mcp", "args": []},
            }
        }
        mock_claude_config_path.write_text(json.dumps(config))

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0

        config = json.loads(mock_claude_config_path.read_text())
        assert "headroom" not in config["mcpServers"]
        assert "github" in config["mcpServers"]

    def test_uninstall_no_config_file(self, mock_claude_config_path):
        """Uninstall with no config file exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "nothing to uninstall" in result.output.lower()

    def test_uninstall_not_configured(self, mock_claude_config_path):
        """Uninstall when headroom not in config exits cleanly."""
        # Create config without headroom
        config = {"mcpServers": {"other": {"command": "other"}}}
        mock_claude_config_path.write_text(json.dumps(config))

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "not configured" in result.output.lower()


class TestMCPStatusCommand:
    """Test 'headroom mcp status' command."""

    def test_status_not_configured(self, mock_claude_config_path):
        """Status shows not configured when no config."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "status"])

        assert result.exit_code == 0
        assert "MCP SDK" in result.output
        # Should show not configured
        assert (
            "✗" in result.output
            or "Not configured" in result.output.lower()
            or "No config" in result.output
        )

    def test_status_configured(self, mock_claude_config_path):
        """Status shows configured when installed."""
        runner = CliRunner()
        runner.invoke(main, ["mcp", "install"])

        result = runner.invoke(main, ["mcp", "status"])

        assert result.exit_code == 0
        assert "✓ Configured" in result.output


class TestMCPServeCommand:
    """Test 'headroom mcp serve' command."""

    def test_serve_help(self):
        """Serve command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "serve", "--help"])

        assert result.exit_code == 0
        assert "proxy-url" in result.output
        assert "debug" in result.output


class TestMCPServerInitialization:
    """Test actual MCP server creation."""

    def test_mcp_server_can_be_created(self):
        """MCP server can be instantiated."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server

        server = create_ccr_mcp_server()
        assert server is not None
        assert server.proxy_url == "http://127.0.0.1:8787"

    def test_mcp_server_with_custom_url(self):
        """MCP server accepts custom proxy URL."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server

        server = create_ccr_mcp_server(proxy_url="http://custom:9000")
        assert server.proxy_url == "http://custom:9000"

    def test_mcp_server_has_correct_tool_name(self):
        """MCP server is configured for headroom_retrieve tool."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server
        from headroom.ccr.tool_injection import CCR_TOOL_NAME

        server = create_ccr_mcp_server()

        # Verify the server was created with correct configuration
        assert server.server is not None
        assert server.server.name == "headroom-ccr"
        # The tool name should be headroom_retrieve
        assert CCR_TOOL_NAME == "headroom_retrieve"


class TestEndToEndFlow:
    """Test complete install -> status -> uninstall flow."""

    def test_full_lifecycle(self, mock_claude_config_path):
        """Test complete lifecycle of MCP configuration."""
        runner = CliRunner()

        # Initially not configured
        result = runner.invoke(main, ["mcp", "status"])
        assert "No config" in result.output or "Not configured" in result.output.lower()

        # Install
        result = runner.invoke(main, ["mcp", "install"])
        assert result.exit_code == 0
        assert "installed" in result.output.lower()

        # Status shows configured
        result = runner.invoke(main, ["mcp", "status"])
        assert "✓ Configured" in result.output

        # Config file has correct content
        config = json.loads(mock_claude_config_path.read_text())
        assert config["mcpServers"]["headroom"]["command"] == "headroom"

        # Uninstall
        result = runner.invoke(main, ["mcp", "uninstall"])
        assert result.exit_code == 0
        assert "removed" in result.output.lower()

        # Status shows not configured
        result = runner.invoke(main, ["mcp", "status"])
        assert "headroom" not in result.output.lower() or "not configured" in result.output.lower()
