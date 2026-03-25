"""Anonymous usage telemetry beacon for Headroom.

Sends aggregate-only stats (tokens saved, compression ratios, cache hit rates)
to help improve Headroom. No prompts, no content, no PII.

On by default. Opt out with:
    HEADROOM_TELEMETRY=off headroom proxy
    headroom proxy --no-telemetry
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
import time

logger = logging.getLogger(__name__)

# Supabase endpoint (write-only: anon key can INSERT, not read/update/delete)
_SUPABASE_URL = "https://dtlllcsudcoasebbamcq.supabase.co"
_SUPABASE_KEY = "sb_publishable_kHcSIX2Ip0_m0C3WuwZlaQ_33my7qya"
_TABLE = "proxy_telemetry"
_ENDPOINT = f"{_SUPABASE_URL}/rest/v1/{_TABLE}"

# Report every 5 minutes
_INTERVAL_SECONDS = 300


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled (on by default, opt out with env var)."""
    val = os.environ.get("HEADROOM_TELEMETRY", "on").lower().strip()
    return val not in ("off", "false", "0", "no", "disable", "disabled")


class TelemetryBeacon:
    """Periodically sends anonymous aggregate stats to Supabase."""

    def __init__(self, port: int = 8787, sdk: str = "proxy", backend: str = "anthropic") -> None:
        self._port = port
        self._sdk = sdk
        self._backend = backend
        self._task: asyncio.Task[None] | None = None
        self._start_time = time.time()

    async def start(self) -> None:
        """Start the periodic beacon. Call from proxy startup."""
        if not is_telemetry_enabled():
            logger.debug("Telemetry disabled (HEADROOM_TELEMETRY=off)")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Telemetry: ENABLED (anonymous aggregate stats, opt out: HEADROOM_TELEMETRY=off)"
        )

    async def stop(self) -> None:
        """Stop and send one final report. Call from proxy shutdown."""
        if self._task:
            self._task.cancel()
            self._task = None
        # Final report
        if is_telemetry_enabled():
            await self._report()

    async def _loop(self) -> None:
        """Background loop: wait, report, repeat."""
        # Wait 60 seconds before first report
        await asyncio.sleep(60)
        while True:
            try:
                await self._report()
            except Exception:
                pass  # Never crash the proxy for telemetry
            await asyncio.sleep(_INTERVAL_SECONDS)

    async def _report(self) -> None:
        """Fetch stats from local /stats endpoint and POST to Supabase."""
        try:
            import httpx
        except ImportError:
            return

        # Fetch stats from our own proxy
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://127.0.0.1:{self._port}/stats")
                if resp.status_code != 200:
                    return
                stats = resp.json()
        except Exception:
            return

        tokens = stats.get("tokens", {})
        requests = stats.get("requests", {})
        cache = stats.get("prefix_cache", {}).get("totals", {})
        cost = stats.get("cost", {})
        models_by = requests.get("by_model", {})
        models = [m for m in models_by.keys() if not m.startswith("passthrough:")]

        # Don't send empty stats — no point reporting zeros
        total_requests = requests.get("total", 0)
        if total_requests == 0:
            return

        session_minutes = max(1, int((time.time() - self._start_time) / 60))

        try:
            from headroom import __version__ as headroom_version
        except Exception:
            headroom_version = "unknown"

        payload = {
            "headroom_version": headroom_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "os": f"{platform.system()} {platform.machine()}",
            "sdk": self._sdk,
            "backend": self._backend,
            "tokens_saved": tokens.get("saved", 0),
            "requests": requests.get("total", 0),
            "compression_percent": tokens.get("savings_percent", 0),
            "cache_hit_rate": cache.get("hit_rate", 0),
            "cost_saved_usd": cost.get("savings_usd", 0),
            "cache_saved_usd": cache.get("cache_savings_usd", 0),
            "session_minutes": session_minutes,
            "models_used": models,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    _ENDPOINT,
                    json=payload,
                    headers={
                        "apikey": _SUPABASE_KEY,
                        "Authorization": f"Bearer {_SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                )
        except Exception:
            pass  # Fire and forget
