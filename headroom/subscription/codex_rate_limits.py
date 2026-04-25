"""Passive tracking of OpenAI Codex rate-limit window data from response headers.

Codex (OpenAI) embeds rate-limit data directly in API response headers
(``x-codex-primary-used-percent``, ``x-codex-primary-window-minutes``, etc.)
rather than exposing a dedicated usage endpoint.  This module captures those
headers from responses that headroom proxies and makes them available in
``/stats`` and the dashboard.

Header schema (parsed by codex-rs ``rate_limits.rs``):
    x-codex-primary-used-percent      float 0-100
    x-codex-primary-window-minutes    int   window size
    x-codex-primary-reset-at          int   Unix timestamp (seconds)
    x-codex-secondary-used-percent    float 0-100   (optional)
    x-codex-secondary-window-minutes  int            (optional)
    x-codex-secondary-reset-at        int            (optional)
    x-codex-credits-has-credits       bool
    x-codex-credits-unlimited         bool
    x-codex-credits-balance           str   e.g. "$5.00"
    x-codex-promo-message             str   server announcement
    x-codex-limit-name                str   e.g. "gpt-5.2-codex-sonic"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

from headroom.subscription.base import QuotaTracker


@dataclass
class CodexRateLimitWindow:
    """Usage data for a single rolling rate-limit window."""

    used_percent: float
    window_minutes: int | None = None
    resets_at: int | None = None  # Unix timestamp (seconds)

    @property
    def window_label(self) -> str:
        if self.window_minutes is None:
            return "unknown"
        if self.window_minutes < 60:
            return f"{self.window_minutes}m"
        hours = self.window_minutes // 60
        mins = self.window_minutes % 60
        return f"{hours}h{mins:02d}m" if mins else f"{hours}h"

    @property
    def seconds_until_reset(self) -> int | None:
        if self.resets_at is None:
            return None
        return max(0, self.resets_at - int(time.time()))

    def to_dict(self) -> dict:
        return {
            "used_percent": self.used_percent,
            "window_minutes": self.window_minutes,
            "window_label": self.window_label,
            "resets_at": self.resets_at,
            "seconds_until_reset": self.seconds_until_reset,
        }


@dataclass
class CodexCreditsSnapshot:
    """OpenAI credits balance for Codex."""

    has_credits: bool
    unlimited: bool
    balance: str | None = None

    def to_dict(self) -> dict:
        return {
            "has_credits": self.has_credits,
            "unlimited": self.unlimited,
            "balance": self.balance,
        }


@dataclass
class CodexRateLimitSnapshot:
    """Full rate-limit snapshot parsed from a single Codex API response."""

    limit_id: str = "codex"
    limit_name: str | None = None
    primary: CodexRateLimitWindow | None = None
    secondary: CodexRateLimitWindow | None = None
    credits: CodexCreditsSnapshot | None = None
    promo_message: str | None = None
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "limit_id": self.limit_id,
            "limit_name": self.limit_name,
            "primary": self.primary.to_dict() if self.primary else None,
            "secondary": self.secondary.to_dict() if self.secondary else None,
            "credits": self.credits.to_dict() if self.credits else None,
            "promo_message": self.promo_message,
            "captured_at": self.captured_at,
        }


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------


def _parse_float(headers: dict[str, str], name: str) -> float | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        v = float(raw)
        return v if v == v else None  # NaN guard
    except (ValueError, TypeError):
        return None


def _parse_int(headers: dict[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _parse_bool(headers: dict[str, str], name: str) -> bool | None:
    raw = headers.get(name)
    if raw is None:
        return None
    if raw.lower() in ("true", "1"):
        return True
    if raw.lower() in ("false", "0"):
        return False
    return None


def _parse_window(headers: dict[str, str], prefix: str, which: str) -> CodexRateLimitWindow | None:
    used_pct = _parse_float(headers, f"{prefix}-{which}-used-percent")
    if used_pct is None:
        return None
    return CodexRateLimitWindow(
        used_percent=used_pct,
        window_minutes=_parse_int(headers, f"{prefix}-{which}-window-minutes"),
        resets_at=_parse_int(headers, f"{prefix}-{which}-reset-at"),
    )


def _parse_credits(headers: dict[str, str]) -> CodexCreditsSnapshot | None:
    has_credits = _parse_bool(headers, "x-codex-credits-has-credits")
    if has_credits is None:
        return None
    unlimited = _parse_bool(headers, "x-codex-credits-unlimited") or False
    raw_balance = headers.get("x-codex-credits-balance", "").strip()
    return CodexCreditsSnapshot(
        has_credits=has_credits,
        unlimited=unlimited,
        balance=raw_balance or None,
    )


def parse_codex_rate_limits(headers: dict[str, str]) -> CodexRateLimitSnapshot | None:
    """Parse a :class:`CodexRateLimitSnapshot` from a dict of HTTP response headers.

    Returns ``None`` when no Codex rate-limit headers are present (e.g. the
    response came from a non-Codex OpenAI endpoint or a cached reply).
    """
    prefix = "x-codex"
    primary = _parse_window(headers, prefix, "primary")
    secondary = _parse_window(headers, prefix, "secondary")
    credits = _parse_credits(headers)
    raw_promo = headers.get("x-codex-promo-message", "").strip()
    promo = raw_promo or None
    raw_limit_name = headers.get("x-codex-limit-name", "").strip()
    limit_name = raw_limit_name or None

    if primary is None and secondary is None and credits is None and promo is None:
        return None  # Not a Codex response with rate-limit headers

    return CodexRateLimitSnapshot(
        limit_id="codex",
        limit_name=limit_name,
        primary=primary,
        secondary=secondary,
        credits=credits,
        promo_message=promo,
    )


# ---------------------------------------------------------------------------
# Singleton state store
# ---------------------------------------------------------------------------


class CodexRateLimitState(QuotaTracker):
    """Thread-safe store for the latest Codex rate-limit snapshot.

    Implements :class:`~headroom.subscription.base.QuotaTracker` so it can
    be registered with the :class:`~headroom.subscription.base.QuotaTrackerRegistry`.
    This tracker is *passive* — it is updated by the OpenAI proxy handler
    each time a response containing ``x-codex-*`` headers passes through
    headroom, so :meth:`start` and :meth:`stop` are no-ops.
    """

    # QuotaTracker identity
    key = "codex_rate_limits"
    label = "OpenAI Codex"

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: CodexRateLimitSnapshot | None = None

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update state from a response header dict (no-op if no Codex headers)."""
        snapshot = parse_codex_rate_limits(headers)
        if snapshot is None:
            return
        with self._lock:
            self._latest = snapshot

    @property
    def latest(self) -> CodexRateLimitSnapshot | None:
        with self._lock:
            return self._latest

    def get_stats(self) -> dict | None:
        snap = self.latest
        return snap.to_dict() if snap is not None else None


_state: CodexRateLimitState | None = None
_state_lock = Lock()


def get_codex_rate_limit_state() -> CodexRateLimitState:
    """Return the process-global :class:`CodexRateLimitState` singleton."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = CodexRateLimitState()
    return _state
