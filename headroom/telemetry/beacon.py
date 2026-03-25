"""Anonymous usage telemetry beacon for Headroom.

Sends aggregate-only stats (tokens saved, compression ratios, cache hit rates,
performance overhead) to help improve Headroom.  No prompts, no content, no PII.

On by default. Opt out with:
    HEADROOM_TELEMETRY=off headroom proxy
    headroom proxy --no-telemetry
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import platform
import sys
import time
import uuid

logger = logging.getLogger(__name__)

# Supabase endpoint (anon key can INSERT+UPDATE for upsert, not read/delete)
# NOTE: Table requires a UNIQUE constraint on session_id for upsert to work.
#       RLS policy must allow UPDATE (in addition to INSERT) for the anon role.
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
        # Unique per proxy run — used as upsert key so each session produces 1 row
        self._session_id = uuid.uuid4().hex
        # Stable across restarts — anonymous machine fingerprint (SHA256 of hostname)
        self._instance_id = hashlib.sha256(platform.node().encode()).hexdigest()[:16]

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
        """Fetch stats from local /stats endpoint and POST to Supabase.

        Wrapped in multiple try/except layers so that:
        1. A missing httpx import silently skips.
        2. A failed /stats fetch silently skips.
        3. Extraction of any stats section is independent — one bad key
           never blocks the others.
        4. A failed Supabase POST silently skips (fire-and-forget).
        The proxy NEVER crashes or slows down because of telemetry.
        """
        try:
            import httpx
        except ImportError:
            return

        # ---- Fetch stats from our own proxy ----
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://127.0.0.1:{self._port}/stats")
                if resp.status_code != 200:
                    return
                stats = resp.json()
        except Exception:
            return

        # Don't send empty stats — no point reporting zeros
        try:
            total_requests = stats.get("requests", {}).get("total", 0)
            if total_requests == 0:
                return
        except Exception:
            return

        # ---- Build payload — each section guarded independently ----
        session_minutes = max(1, int((time.time() - self._start_time) / 60))

        try:
            from headroom import __version__ as headroom_version
        except Exception:
            headroom_version = "unknown"

        # Core identity (always present)
        payload: dict = {
            "session_id": self._session_id,
            "instance_id": self._instance_id,
            "headroom_version": headroom_version,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "os": f"{platform.system()} {platform.machine()}",
            "sdk": self._sdk,
            "backend": self._backend,
            "session_minutes": session_minutes,
        }

        # --- Effectiveness metrics ---
        try:
            tokens = stats.get("tokens", {})
            requests_stats = stats.get("requests", {})
            cache = stats.get("prefix_cache", {}).get("totals", {})
            cost = stats.get("cost", {})
            models_by = requests_stats.get("by_model", {})

            payload.update(
                {
                    "tokens_saved": tokens.get("saved", 0),
                    "requests": requests_stats.get("total", 0),
                    "compression_percent": tokens.get("savings_percent", 0),
                    "cache_hit_rate": cache.get("hit_rate", 0),
                    "cost_saved_usd": cost.get("savings_usd", 0),
                    "cache_saved_usd": cost.get("cache_savings_usd", 0),
                    "models_used": [
                        m for m in models_by.keys() if not m.startswith("passthrough:")
                    ],
                }
            )
        except Exception:
            logger.debug("Beacon: failed to extract effectiveness metrics", exc_info=True)

        # --- Performance overhead (how much latency Headroom adds) ---
        try:
            overhead = stats.get("overhead", {})
            payload.update(
                {
                    "overhead_avg_ms": round(overhead.get("average_ms", 0), 2),
                    "overhead_max_ms": round(overhead.get("max_ms", 0), 2),
                }
            )
        except Exception:
            logger.debug("Beacon: failed to extract overhead metrics", exc_info=True)

        # --- TTFB (time to first byte — what the user feels) ---
        try:
            ttfb = stats.get("ttfb", {})
            payload["ttfb_avg_ms"] = round(ttfb.get("average_ms", 0), 2)
        except Exception:
            logger.debug("Beacon: failed to extract TTFB metrics", exc_info=True)

        # --- Pipeline timing breakdown (where is time spent?) ---
        # Stored as JSONB — variable-shape dict of transform_name → avg_ms.
        # This is the most valuable data for optimising Headroom internals.
        try:
            raw_timing = stats.get("pipeline_timing", {})
            if raw_timing:
                # Flatten to {name: avg_ms} for compact storage
                pipeline_timing = {
                    name: round(info.get("average_ms", 0), 2)
                    for name, info in raw_timing.items()
                    if isinstance(info, dict)
                }
                payload["pipeline_timing"] = pipeline_timing
        except Exception:
            logger.debug("Beacon: failed to extract pipeline timing", exc_info=True)

        # --- Request patterns (how big are conversations?) ---
        try:
            tokens = stats.get("tokens", {})
            total_req = stats.get("requests", {}).get("total", 1)
            tokens_before = tokens.get("original", 0)
            tokens_after = tokens.get("optimized", 0)
            payload.update(
                {
                    "avg_tokens_before": round(tokens_before / max(total_req, 1)),
                    "avg_tokens_after": round(tokens_after / max(total_req, 1)),
                }
            )
        except Exception:
            logger.debug("Beacon: failed to extract request patterns", exc_info=True)

        # --- Compression cache effectiveness ---
        try:
            cc = stats.get("compression_cache", {})
            if cc:
                payload["compression_cache"] = {
                    "hit_rate": cc.get("hit_rate", 0),
                    "entries": cc.get("entries", 0),
                    "avg_lookup_ns": cc.get("avg_lookup_ns", 0),
                }
        except Exception:
            logger.debug("Beacon: failed to extract cache stats", exc_info=True)

        # --- CCR (Compress-Cache-Retrieve) usage ---
        try:
            ccr = stats.get("compression", {})
            if ccr.get("ccr_entries", 0) > 0:
                payload["ccr"] = {
                    "entries": ccr.get("ccr_entries", 0),
                    "retrievals": ccr.get("ccr_retrievals", 0),
                }
        except Exception:
            logger.debug("Beacon: failed to extract CCR stats", exc_info=True)

        # --- Waste signals (what patterns of waste do we see?) ---
        try:
            waste = stats.get("waste_signals", {})
            if waste:
                payload["waste_signals"] = waste
        except Exception:
            logger.debug("Beacon: failed to extract waste signals", exc_info=True)

        # ---- Send to Supabase (fire-and-forget) ----
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    _ENDPOINT,
                    json=payload,
                    headers={
                        "apikey": _SUPABASE_KEY,
                        "Authorization": f"Bearer {_SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        # Upsert: on conflict with session_id, merge (overwrite) the row
                        "Prefer": "resolution=merge-duplicates,return=minimal",
                    },
                )
        except Exception:
            # No internet, DNS failure, timeout, Supabase down — all fine.
            # Headroom continues working perfectly without telemetry.
            logger.debug("Beacon: failed to send telemetry", exc_info=True)
