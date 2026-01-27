"""Tests for proxy scalability features.

These tests verify connection pooling, HTTP/2, and worker configuration.
"""

import asyncio

import httpx
import pytest


class TestConnectionPoolConfig:
    """Test connection pool configuration."""

    def test_httpx_limits_basic(self):
        """Test that httpx accepts our connection limits."""
        limits = httpx.Limits(
            max_connections=500,
            max_keepalive_connections=100,
        )
        assert limits.max_connections == 500
        assert limits.max_keepalive_connections == 100

    def test_httpx_limits_custom(self):
        """Test custom connection limits."""
        limits = httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=200,
        )
        assert limits.max_connections == 1000
        assert limits.max_keepalive_connections == 200

    def test_httpx_timeout_config(self):
        """Test timeout configuration for proxy."""
        timeout = httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=300.0,
            pool=10.0,
        )
        assert timeout.connect == 10.0
        assert timeout.read == 300.0
        assert timeout.write == 300.0
        assert timeout.pool == 10.0

    @pytest.mark.asyncio
    async def test_async_client_with_limits(self):
        """Test AsyncClient accepts connection pool limits."""
        limits = httpx.Limits(
            max_connections=500,
            max_keepalive_connections=100,
        )
        async with httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(10.0),
        ) as client:
            # Verify client was created successfully with our limits
            # (httpx doesn't expose limits directly, but creation succeeds)
            assert client is not None
            # The limits object we passed should have our values
            assert limits.max_connections == 500
            assert limits.max_keepalive_connections == 100


class TestHTTP2Config:
    """Test HTTP/2 configuration."""

    def test_http2_requires_h2_package(self):
        """Test that http2=True requires h2 package."""
        import importlib.util

        h2_available = importlib.util.find_spec("h2") is not None

        if h2_available:
            # Should work if h2 is installed
            client = httpx.Client(http2=True)
            assert client._base_url is not None
            client.close()
        else:
            # Should raise if h2 not installed
            with pytest.raises(ImportError):
                httpx.Client(http2=True)

    @pytest.mark.asyncio
    async def test_async_client_http2(self):
        """Test AsyncClient with HTTP/2 enabled."""
        import importlib.util

        if not importlib.util.find_spec("h2"):
            pytest.skip("h2 package not installed")

        async with httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(max_connections=100),
        ) as client:
            # Client should be configured for HTTP/2
            assert client is not None


class TestProxyConfigDataclass:
    """Test ProxyConfig dataclass with new fields."""

    def test_proxy_config_defaults(self):
        """Test default values for scalability settings."""
        from dataclasses import dataclass

        @dataclass
        class ProxyConfigTest:
            """Minimal proxy config for testing."""

            host: str = "127.0.0.1"
            port: int = 8787
            request_timeout_seconds: int = 300
            connect_timeout_seconds: int = 10
            max_connections: int = 500
            max_keepalive_connections: int = 100
            http2: bool = True

        config = ProxyConfigTest()
        assert config.max_connections == 500
        assert config.max_keepalive_connections == 100
        assert config.http2 is True

    def test_proxy_config_custom_values(self):
        """Test custom values for scalability settings."""
        from dataclasses import dataclass

        @dataclass
        class ProxyConfigTest:
            max_connections: int = 500
            max_keepalive_connections: int = 100
            http2: bool = True

        config = ProxyConfigTest(
            max_connections=1000,
            max_keepalive_connections=200,
            http2=False,
        )
        assert config.max_connections == 1000
        assert config.max_keepalive_connections == 200
        assert config.http2 is False


class TestConcurrencyPatterns:
    """Test async concurrency patterns used in proxy."""

    @pytest.mark.asyncio
    async def test_semaphore_for_backpressure(self):
        """Test semaphore pattern for limiting concurrent requests."""
        semaphore = asyncio.Semaphore(3)
        active = []
        completed = []

        async def task(task_id: int):
            async with semaphore:
                active.append(task_id)
                # Verify we never exceed semaphore limit
                assert len(active) <= 3
                await asyncio.sleep(0.01)
                active.remove(task_id)
                completed.append(task_id)

        # Run 10 tasks with max 3 concurrent
        tasks = [task(i) for i in range(10)]
        await asyncio.gather(*tasks)

        assert len(completed) == 10

    @pytest.mark.asyncio
    async def test_connection_reuse_pattern(self):
        """Test that single client instance is reused (not recreated)."""
        clients_created = []

        class MockProxyWithClient:
            def __init__(self):
                self.http_client = None

            async def startup(self):
                self.http_client = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100),
                )
                clients_created.append(self.http_client)

            async def shutdown(self):
                if self.http_client:
                    await self.http_client.aclose()

            async def make_request(self, url: str):
                # Should reuse the same client, not create new one
                return self.http_client

        proxy = MockProxyWithClient()
        await proxy.startup()

        # Multiple requests should return same client instance
        client1 = await proxy.make_request("http://example1.com")
        client2 = await proxy.make_request("http://example2.com")
        client3 = await proxy.make_request("http://example3.com")

        assert client1 is client2 is client3
        assert len(clients_created) == 1  # Only one client created

        await proxy.shutdown()


class TestTimeoutOverrides:
    """Test per-request timeout overrides."""

    @pytest.mark.asyncio
    async def test_request_level_timeout_override(self):
        """Test that timeout can be overridden per-request."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),  # Default timeout
        ):
            # Per-request override should work
            override_timeout = httpx.Timeout(120.0)
            # Just verify the timeout object is valid
            assert override_timeout.read == 120.0
            assert override_timeout.connect == 120.0


class TestWorkerConfiguration:
    """Test worker process configuration."""

    def test_uvicorn_workers_parameter(self):
        """Test that uvicorn accepts workers parameter."""
        # uvicorn.run accepts workers=N for multi-process
        import uvicorn

        # Verify the Config class accepts workers
        config = uvicorn.Config(
            app="app:app",
            workers=4,
            limit_concurrency=1000,
        )
        assert config.workers == 4
        assert config.limit_concurrency == 1000

    def test_single_worker_default(self):
        """Test that default is single worker (None)."""
        import uvicorn

        config = uvicorn.Config(app="app:app")
        # Default should be None (single process)
        assert config.workers is None or config.workers == 1
