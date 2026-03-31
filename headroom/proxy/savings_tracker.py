"""Durable proxy savings history tracking.

Persists cumulative proxy compression savings to a local JSON file so
historical charts survive proxy restarts and can be shared by multiple
Headroom frontends.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from csv import DictWriter
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HEADROOM_SAVINGS_PATH_ENV_VAR = "HEADROOM_SAVINGS_PATH"
DEFAULT_SAVINGS_DIR = ".headroom"
DEFAULT_SAVINGS_FILE = "proxy_savings.json"
SCHEMA_VERSION = 1
DEFAULT_MAX_HISTORY_POINTS = 5000
DEFAULT_MAX_HISTORY_AGE_DAYS = 365

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def get_default_savings_storage_path() -> str:
    """Return the configured savings storage path."""
    env_path = os.environ.get(HEADROOM_SAVINGS_PATH_ENV_VAR, "").strip()
    if env_path:
        return env_path
    return str(Path.home() / DEFAULT_SAVINGS_DIR / DEFAULT_SAVINGS_FILE)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_start(timestamp: datetime, bucket: str) -> datetime:
    if bucket == "hour":
        return timestamp.replace(minute=0, second=0, microsecond=0)
    if bucket == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "week":
        day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start - timedelta(days=day_start.weekday())
    if bucket == "month":
        return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported savings history bucket: {bucket}")


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return default


def _resolve_litellm_model(model: str) -> str:
    """Resolve model name to one LiteLLM recognizes."""
    if not LITELLM_AVAILABLE:
        return model

    try:
        litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=0)
        return model
    except Exception:
        pass

    prefixes = {
        "claude-": "anthropic/",
        "gpt-": "openai/",
        "o1-": "openai/",
        "o3-": "openai/",
        "o4-": "openai/",
        "gemini-": "google/",
    }
    for pattern, prefix in prefixes.items():
        if model.startswith(pattern):
            candidate = f"{prefix}{model}"
            try:
                litellm.cost_per_token(model=candidate, prompt_tokens=1, completion_tokens=0)
                return candidate
            except Exception:
                break

    return model


def _estimate_compression_savings_usd(model: str, tokens_saved: int) -> float:
    """Estimate compression savings in USD from saved input tokens."""
    if tokens_saved <= 0 or not LITELLM_AVAILABLE:
        return 0.0

    try:
        resolved = _resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved, {})
        input_cost_per_token = info.get("input_cost_per_token")
        if not input_cost_per_token:
            return 0.0
        return float(tokens_saved) * float(input_cost_per_token)
    except Exception:
        return 0.0


def _normalize_history_entry(entry: Any) -> dict[str, Any] | None:
    """Normalize persisted history entries across schema shapes."""
    timestamp: datetime | None = None
    total_tokens_saved = 0
    compression_savings_usd = 0.0

    if isinstance(entry, dict):
        timestamp = _parse_timestamp(entry.get("timestamp"))
        total_tokens_saved = _coerce_int(entry.get("total_tokens_saved"))
        compression_savings_usd = _coerce_float(entry.get("compression_savings_usd"))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        timestamp = _parse_timestamp(entry[0])
        total_tokens_saved = _coerce_int(entry[1])
        if len(entry) >= 3:
            compression_savings_usd = _coerce_float(entry[2])
    else:
        return None

    if timestamp is None:
        return None

    return {
        "timestamp": _to_utc_iso(timestamp),
        "total_tokens_saved": total_tokens_saved,
        "compression_savings_usd": round(compression_savings_usd, 6),
    }


class SavingsTracker:
    """Persist bounded proxy compression savings history."""

    def __init__(
        self,
        path: str | None = None,
        max_history_points: int = DEFAULT_MAX_HISTORY_POINTS,
        max_history_age_days: int = DEFAULT_MAX_HISTORY_AGE_DAYS,
    ) -> None:
        self._path = Path(path or get_default_savings_storage_path())
        self._max_history_points = max_history_points
        self._max_history_age_days = max_history_age_days
        self._lock = threading.Lock()
        self._state = self._load_state()

    @property
    def storage_path(self) -> str:
        return str(self._path)

    def record_compression_savings(
        self,
        *,
        model: str,
        tokens_saved: int,
        timestamp: datetime | str | None = None,
    ) -> bool:
        """Persist a cumulative savings checkpoint when compression changed totals."""
        delta_tokens = _coerce_int(tokens_saved)
        if delta_tokens <= 0:
            return False

        timestamp_dt = (
            _parse_timestamp(timestamp)
            if isinstance(timestamp, str)
            else timestamp.astimezone(timezone.utc)
            if isinstance(timestamp, datetime)
            else _utc_now()
        )
        if timestamp_dt is None:
            timestamp_dt = _utc_now()

        delta_usd = _estimate_compression_savings_usd(model, delta_tokens)

        with self._lock:
            lifetime = self._state["lifetime"]
            lifetime["tokens_saved"] += delta_tokens
            lifetime["compression_savings_usd"] = round(
                lifetime["compression_savings_usd"] + delta_usd, 6
            )

            self._state["history"].append(
                {
                    "timestamp": _to_utc_iso(timestamp_dt),
                    "total_tokens_saved": lifetime["tokens_saved"],
                    "compression_savings_usd": lifetime["compression_savings_usd"],
                }
            )
            self._trim_history_locked(reference_time=timestamp_dt)
            self._save_locked()
            return True

    def stats_preview(self, recent_points: int = 20) -> dict[str, Any]:
        """Return a compact preview for `/stats`."""
        snapshot = self.snapshot()
        return {
            "schema_version": snapshot["schema_version"],
            "storage_path": snapshot["storage_path"],
            "lifetime": snapshot["lifetime"],
            "history_points": len(snapshot["history"]),
            "recent_history": snapshot["history"][-recent_points:],
            "retention": snapshot["retention"],
        }

    def history_response(self) -> dict[str, Any]:
        """Return frontend-friendly historical data for `/stats-history`."""
        snapshot = self.snapshot()
        history = snapshot["history"]
        series = {
            "hourly": self._build_rollup(history, bucket="hour"),
            "daily": self._build_rollup(history, bucket="day"),
            "weekly": self._build_rollup(history, bucket="week"),
            "monthly": self._build_rollup(history, bucket="month"),
        }
        return {
            "schema_version": snapshot["schema_version"],
            "generated_at": _to_utc_iso(_utc_now()),
            "storage_path": snapshot["storage_path"],
            "lifetime": snapshot["lifetime"],
            "history": history,
            "series": series,
            "exports": {
                "default_format": "json",
                "available_formats": ["json", "csv"],
                "available_series": ["history", *series.keys()],
            },
            "retention": snapshot["retention"],
        }

    def export_rows(self, series: str = "history") -> list[dict[str, Any]]:
        """Return export rows for history or a rollup series."""
        response = self.history_response()
        if series == "history":
            return [dict(item) for item in response["history"]]
        return [dict(item) for item in response["series"].get(series, [])]

    def export_csv(self, series: str = "history") -> str:
        """Export history or rollup series as CSV."""
        rows = self.export_rows(series=series)
        if series == "history":
            fieldnames = ["timestamp", "total_tokens_saved", "compression_savings_usd"]
        else:
            fieldnames = [
                "timestamp",
                "tokens_saved",
                "compression_savings_usd_delta",
                "total_tokens_saved",
                "compression_savings_usd",
            ]

        buffer = StringIO()
        writer = DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        return buffer.getvalue()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            history = [dict(item) for item in self._state["history"]]
            return {
                "schema_version": SCHEMA_VERSION,
                "storage_path": str(self._path),
                "lifetime": dict(self._state["lifetime"]),
                "history": history,
                "retention": {
                    "max_history_points": self._max_history_points,
                    "max_history_age_days": self._max_history_age_days,
                },
            }

    def _default_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "lifetime": {"tokens_saved": 0, "compression_savings_usd": 0.0},
            "history": [],
        }

    def _load_state(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._default_state()

        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load savings history from %s: %s", self._path, e)
            return self._default_state()

        return self._sanitize_state(raw)

    def _sanitize_state(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return self._default_state()

        history_raw = raw.get("history", [])
        normalized_history = []
        if isinstance(history_raw, list):
            for item in history_raw:
                normalized = _normalize_history_entry(item)
                if normalized is not None:
                    normalized_history.append(normalized)

        normalized_history.sort(key=lambda item: item["timestamp"])

        lifetime_raw = raw.get("lifetime", {})
        lifetime_tokens_saved = 0
        lifetime_savings_usd = 0.0
        if isinstance(lifetime_raw, dict):
            lifetime_tokens_saved = _coerce_int(lifetime_raw.get("tokens_saved"))
            lifetime_savings_usd = _coerce_float(lifetime_raw.get("compression_savings_usd"))

        if normalized_history:
            last = normalized_history[-1]
            lifetime_tokens_saved = max(lifetime_tokens_saved, last["total_tokens_saved"])
            lifetime_savings_usd = max(
                lifetime_savings_usd,
                _coerce_float(last["compression_savings_usd"]),
            )

        state = {
            "schema_version": SCHEMA_VERSION,
            "lifetime": {
                "tokens_saved": lifetime_tokens_saved,
                "compression_savings_usd": round(lifetime_savings_usd, 6),
            },
            "history": normalized_history,
        }

        if normalized_history:
            reference_time = _parse_timestamp(normalized_history[-1]["timestamp"]) or _utc_now()
            original_state = self._state if hasattr(self, "_state") else None
            self._state = state
            try:
                self._trim_history_locked(reference_time=reference_time)
                state = self._state
            finally:
                if original_state is not None:
                    self._state = original_state

        return state

    def _trim_history_locked(self, reference_time: datetime | None = None) -> None:
        history = self._state["history"]
        if not history:
            return

        if self._max_history_age_days > 0:
            cutoff = (reference_time or _utc_now()) - timedelta(days=self._max_history_age_days)
            filtered = [
                item
                for item in history
                if (_parse_timestamp(item["timestamp"]) or _utc_now()) >= cutoff
            ]
            if not filtered:
                filtered = [history[-1]]
            history = filtered

        if self._max_history_points > 0 and len(history) > self._max_history_points:
            history = history[-self._max_history_points :]

        self._state["history"] = history

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "lifetime": self._state["lifetime"],
                "history": self._state["history"],
            }
            json_data = json.dumps(payload, indent=2)

            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".proxy_savings_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json_data)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp_path).replace(self._path)
            except Exception:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Failed to save savings history to %s: %s", self._path, e)

    def _build_rollup(self, history: list[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
        if not history:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        prev_total_tokens = 0
        prev_total_usd = 0.0

        for point in history:
            timestamp = _parse_timestamp(point["timestamp"])
            if timestamp is None:
                continue

            bucket_start = _bucket_start(timestamp, bucket)

            bucket_key = _to_utc_iso(bucket_start)
            total_tokens_saved = _coerce_int(point.get("total_tokens_saved"))
            total_usd = _coerce_float(point.get("compression_savings_usd"))
            delta_tokens = max(total_tokens_saved - prev_total_tokens, 0)
            delta_usd = max(total_usd - prev_total_usd, 0.0)

            prev_total_tokens = total_tokens_saved
            prev_total_usd = total_usd

            entry = aggregated.setdefault(
                bucket_key,
                {
                    "timestamp": bucket_key,
                    "tokens_saved": 0,
                    "compression_savings_usd_delta": 0.0,
                    "total_tokens_saved": total_tokens_saved,
                    "compression_savings_usd": total_usd,
                },
            )
            entry["tokens_saved"] += delta_tokens
            entry["compression_savings_usd_delta"] = round(
                entry["compression_savings_usd_delta"] + delta_usd,
                6,
            )
            entry["total_tokens_saved"] = total_tokens_saved
            entry["compression_savings_usd"] = round(total_usd, 6)

        return list(aggregated.values())
