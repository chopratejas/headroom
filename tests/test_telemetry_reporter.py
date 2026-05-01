from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from headroom.telemetry.reporter import (
    GRACE_PERIOD_SECONDS,
    LicenseInfo,
    UsageReporter,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = responses or []
        self.posts: list[dict[str, object]] = []
        self.closed = False

    async def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True


def test_license_info_round_trip_and_cache_helpers(tmp_path: Path) -> None:
    validated_at = datetime.now(timezone.utc)
    trial_expires = validated_at + timedelta(days=30)
    info = LicenseInfo(
        status="trial",
        org_id="org-1",
        org_name="Headroom",
        plan="pro",
        quota_tokens=123,
        trial_expires_at=trial_expires,
        validated_at=validated_at,
    )
    restored = LicenseInfo.from_dict(info.to_dict())
    assert restored == info

    cache_path = tmp_path / "license.json"
    reporter = UsageReporter("hlk_test", cache_path=cache_path)
    reporter._license_info = info
    reporter._save_cache()
    assert json.loads(cache_path.read_text())["status"] == "trial"

    loaded = reporter._load_cache_or_default()
    assert loaded.status == "trial"
    assert reporter.should_compress is True
    assert reporter.is_active is True

    stale = LicenseInfo(
        status="active",
        validated_at=datetime.now(timezone.utc) - timedelta(seconds=GRACE_PERIOD_SECONDS + 60),
    )
    cache_path.write_text(json.dumps(stale.to_dict()))
    expired = reporter._load_cache_or_default()
    assert expired.status == "expired"

    cache_path.write_text("{bad json")
    fallback = reporter._load_cache_or_default()
    assert fallback.status == "expired"


@pytest.mark.asyncio
async def test_validate_license_success_failure_and_http_client_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "license.json"
    reporter = UsageReporter("hlk_test", cloud_url="https://cloud.test/", cache_path=cache_path)

    client = _FakeClient(
        [
            _FakeResponse(
                200,
                {
                    "status": "active",
                    "org_id": "org-1",
                    "org_name": "Headroom",
                    "plan": "enterprise",
                    "quota_tokens": 999,
                    "trial_expires_at": "2024-03-01T00:00:00+00:00",
                },
            )
        ]
    )
    reporter._get_client = AsyncMock(return_value=client)
    validated = await reporter.validate_license()
    assert validated.status == "active"
    assert validated.plan == "enterprise"
    assert cache_path.exists()

    fallback_cached = LicenseInfo(
        status="trial",
        validated_at=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    cache_path.write_text(json.dumps(fallback_cached.to_dict()))
    reporter._get_client = AsyncMock(return_value=_FakeClient([RuntimeError("network down")]))
    cached = await reporter.validate_license()
    assert cached.status == "trial"

    reporter = UsageReporter("hlk_test", cache_path=cache_path)
    client = await reporter._get_client()
    assert client is await reporter._get_client()


@pytest.mark.asyncio
async def test_start_stop_and_license_state_properties(tmp_path: Path) -> None:
    reporter = UsageReporter("hlk_test", report_interval=3600, cache_path=tmp_path / "license.json")
    reporter.validate_license = AsyncMock(
        return_value=LicenseInfo(
            status="expired",
            validated_at=datetime.now(timezone.utc) - timedelta(seconds=GRACE_PERIOD_SECONDS + 1),
        )
    )
    reporter._report_loop = AsyncMock(side_effect=lambda: asyncio.sleep(3600))
    proxy = type(
        "Proxy",
        (),
        {
            "cost_tracker": type(
                "CostTracker",
                (),
                {
                    "_tokens_saved_by_model": {},
                    "_tokens_sent_by_model": {},
                    "_requests_by_model": {},
                },
            )()
        },
    )()

    await reporter.start(proxy)
    assert reporter._proxy is proxy
    assert reporter._task is not None

    reporter._license_info = LicenseInfo(status="expired", validated_at=datetime.now(timezone.utc))
    assert reporter.should_compress is True
    reporter._license_info = LicenseInfo(
        status="expired",
        validated_at=datetime.now(timezone.utc) - timedelta(seconds=GRACE_PERIOD_SECONDS + 1),
    )
    assert reporter.should_compress is False
    reporter._license_info = LicenseInfo(status="invalid")
    assert reporter.should_compress is True
    assert reporter.is_active is False

    await reporter.stop()
    assert reporter._stopped is True


@pytest.mark.asyncio
async def test_report_usage_and_snapshot_paths(tmp_path: Path) -> None:
    reporter = UsageReporter(
        "hlk_test", cloud_url="https://cloud.test", cache_path=tmp_path / "license.json"
    )
    reporter._license_info = LicenseInfo(status="active")
    reporter._proxy = type(
        "Proxy",
        (),
        {
            "cost_tracker": type(
                "CostTracker",
                (),
                {
                    "_tokens_saved_by_model": {"gpt-4o": 15, "claude": 3},
                    "_tokens_sent_by_model": {"gpt-4o": 20, "claude": 7},
                    "_requests_by_model": {"gpt-4o": 2, "claude": 1},
                },
            )()
        },
    )()
    reporter._last_tokens_saved_by_model = {"gpt-4o": 5}
    reporter._last_tokens_sent_by_model = {"gpt-4o": 10}
    reporter._last_requests_by_model = {"gpt-4o": 1}
    reporter._last_report_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    client = _FakeClient([_FakeResponse(200, {"status": "expired", "message": "quota exceeded"})])
    reporter._get_client = AsyncMock(return_value=client)
    await reporter._report_usage()
    assert client.posts[0]["url"] == "https://cloud.test/v1/license/usage"
    assert client.posts[0]["json"]["requests"] == 2
    assert client.posts[0]["json"]["tokens_saved"] == 13
    assert reporter._license_info.status == "expired"
    assert reporter._last_requests_by_model == {"gpt-4o": 2, "claude": 1}

    client = _FakeClient([_FakeResponse(503, {})])
    reporter._last_tokens_saved_by_model = {}
    reporter._last_tokens_sent_by_model = {}
    reporter._last_requests_by_model = {}
    reporter._get_client = AsyncMock(return_value=client)
    await reporter._report_usage()
    assert client.posts


@pytest.mark.asyncio
async def test_report_usage_empty_and_exception_paths(tmp_path: Path) -> None:
    reporter = UsageReporter("hlk_test", cache_path=tmp_path / "license.json")
    reporter._proxy = type(
        "Proxy",
        (),
        {
            "cost_tracker": type(
                "CostTracker",
                (),
                {
                    "_tokens_saved_by_model": {},
                    "_tokens_sent_by_model": {},
                    "_requests_by_model": {},
                },
            )()
        },
    )()
    reporter._last_report_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    await reporter._report_usage()
    assert reporter._last_report_time is not None

    reporter._proxy = None
    await reporter._report_usage()

    reporter._proxy = type("Proxy", (), {"cost_tracker": None})()
    reporter._snapshot_metrics()
    assert reporter._last_tokens_saved_by_model == {}

    reporter = UsageReporter("hlk_test", cache_path=tmp_path / "license.json")
    reporter._proxy = type(
        "Proxy",
        (),
        {
            "cost_tracker": type(
                "CostTracker",
                (),
                {
                    "_tokens_saved_by_model": {"gpt-4o": 10},
                    "_tokens_sent_by_model": {"gpt-4o": 5},
                    "_requests_by_model": {"gpt-4o": 1},
                },
            )()
        },
    )()
    reporter._get_client = AsyncMock(return_value=_FakeClient([RuntimeError("boom")]))
    await reporter._report_usage()
