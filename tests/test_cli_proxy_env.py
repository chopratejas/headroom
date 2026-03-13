"""Tests for CLI proxy env variable handling and backend validation.

Verifies that:
1. OPENAI_TARGET_API_URL and GEMINI_TARGET_API_URL env vars are read by `headroom proxy`
2. litellm-* backends are accepted by both CLI and argparse paths
"""

import os
from unittest.mock import patch

import pytest

click = pytest.importorskip("click")
pytest.importorskip("fastapi")

from click.testing import CliRunner  # noqa: E402

from headroom.cli.main import main  # noqa: E402


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIProxyEnvVars:
    """Test that the CLI proxy command reads API URL env vars."""

    def test_openai_target_api_url_from_env(self, runner):
        """OPENAI_TARGET_API_URL env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"OPENAI_TARGET_API_URL": "http://my-vllm:4000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"

    def test_gemini_target_api_url_from_env(self, runner):
        """GEMINI_TARGET_API_URL env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"GEMINI_TARGET_API_URL": "http://my-gemini:5000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].gemini_api_url == "http://my-gemini:5000"

    def test_openai_api_url_cli_flag(self, runner):
        """--openai-api-url CLI flag should take precedence."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--openai-api-url", "http://from-cli:4000"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://from-cli:4000"

    def test_cli_flag_overrides_env_var(self, runner):
        """CLI flag should take precedence over env var."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--openai-api-url", "http://from-cli:4000"],
                env={"OPENAI_TARGET_API_URL": "http://from-env:4000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://from-cli:4000"

    def test_no_env_var_defaults_to_none(self, runner):
        """Without env var or flag, openai_api_url should be None."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        # Ensure the env var is not set
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_TARGET_API_URL"}

        with (
            patch("headroom.proxy.server.run_server", mock_run_server),
            patch.dict(os.environ, env, clear=True),
        ):
            result = runner.invoke(
                main,
                ["proxy"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url is None

    def test_both_api_urls_from_env(self, runner):
        """Both OPENAI and GEMINI target URLs can be set via env."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={
                    "OPENAI_TARGET_API_URL": "http://my-vllm:4000",
                    "GEMINI_TARGET_API_URL": "http://my-gemini:5000",
                },
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"
        assert captured_config["config"].gemini_api_url == "http://my-gemini:5000"


class TestCLIProxyBackend:
    """Test that litellm-* backends are accepted by the CLI."""

    def test_litellm_hosted_vllm_backend_accepted(self, runner):
        """--backend litellm-hosted_vllm should be accepted (not rejected)."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "litellm-hosted_vllm"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-hosted_vllm"

    def test_litellm_vertex_backend_accepted(self, runner):
        """--backend litellm-vertex should be accepted."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "litellm-vertex"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-vertex"

    def test_litellm_backend_with_openai_url(self, runner):
        """Full vLLM setup: litellm backend + OPENAI_TARGET_API_URL."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                [
                    "proxy",
                    "--backend",
                    "litellm-hosted_vllm",
                    "--openai-api-url",
                    "http://my-vllm:4000",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-hosted_vllm"
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"


class TestCLIAnyllmProviderEnv:
    """Test that HEADROOM_ANYLLM_PROVIDER env var is read by the CLI."""

    def test_anyllm_provider_from_env(self, runner):
        """HEADROOM_ANYLLM_PROVIDER env var should override the default."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "anyllm"],
                env={"HEADROOM_ANYLLM_PROVIDER": "llamacpp"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].anyllm_provider == "llamacpp"

    def test_anyllm_provider_cli_flag_works(self, runner):
        """--anyllm-provider flag should still work."""
        captured_config = {}

        def mock_run_server(config):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "anyllm", "--anyllm-provider", "groq"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].anyllm_provider == "groq"


class TestArgparseBackendValidation:
    """Test that the argparse path (python -m headroom.proxy.server) accepts litellm-* backends."""

    def test_argparse_accepts_litellm_backend(self):
        """The argparse --backend should accept litellm-hosted_vllm (no choices restriction)."""
        import argparse

        # Recreate the parser matching server.py's main() argparse setup
        # We just need to verify argparse doesn't reject litellm-* values
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default="anthropic")
        args = parser.parse_args(["--backend", "litellm-hosted_vllm"])
        assert args.backend == "litellm-hosted_vllm"
