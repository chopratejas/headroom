"""GitHub Copilot OAuth discovery and API-token exchange helpers."""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.githubcopilot.com"
DEFAULT_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_GITHUB_HOST = "github.com"
_TOKEN_EXPIRY_BUFFER_S = 60
_DEFAULT_EDITOR_VERSION = "vscode/1.104.1"
_DEFAULT_USER_AGENT = "GitHubCopilotChat/0.1"

_API_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_API_TOKEN",
    "COPILOT_PROVIDER_BEARER_TOKEN",
)
_OAUTH_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "GITHUB_TOKEN",
    "COPILOT_GITHUB_TOKEN",
)
_OAUTH_TOKEN_KEYS = (
    "oauth_token",
    "oauthToken",
    "token",
    "access_token",
    "accessToken",
)
_EXPIRY_KEYS = ("expires_at", "expiresAt", "expiry", "expires")


@dataclass(frozen=True)
class CopilotAPIToken:
    """Short-lived API token exchanged from a GitHub OAuth token."""

    token: str
    expires_at: float
    api_url: str = DEFAULT_API_URL
    refresh_in: int | None = None
    sku: str | None = None

    @property
    def is_valid(self) -> bool:
        return time.time() < (self.expires_at - _TOKEN_EXPIRY_BUFFER_S)


def _github_host() -> str:
    return (os.environ.get("GITHUB_COPILOT_HOST") or DEFAULT_GITHUB_HOST).strip().lower()


def _token_exchange_url() -> str:
    return os.environ.get("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", DEFAULT_TOKEN_EXCHANGE_URL).strip()


def _should_exchange_oauth_token() -> bool:
    raw = os.environ.get("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_token_file_paths() -> list[Path]:
    override = os.environ.get("GITHUB_COPILOT_TOKEN_FILE", "").strip()
    if override:
        return [Path(override).expanduser()]

    paths: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        base = Path(local_appdata) / "github-copilot"
        paths.extend([base / "apps.json", base / "hosts.json"])

    config_base = Path.home() / ".config" / "github-copilot"
    paths.extend([config_base / "apps.json", config_base / "hosts.json"])
    return paths


def _read_gh_cli_oauth_token() -> str | None:
    gh_bin = os.environ.get("GH_PATH", "").strip() or "gh"
    command = [gh_bin, "auth", "token"]
    host = _github_host()
    if host and host != DEFAULT_GITHUB_HOST:
        command.extend(["--hostname", host])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        logger.debug("Unable to invoke GitHub CLI for Copilot auth discovery: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug("GitHub CLI auth token lookup failed with exit code %s", result.returncode)
        return None

    token = result.stdout.strip()
    return token or None


def _read_windows_copilot_cli_oauth_token() -> str | None:
    if os.name != "nt":
        return None

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", wintypes.LPVOID),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    cred_ptr = ctypes.POINTER(CREDENTIAL)
    credentials = ctypes.POINTER(cred_ptr)()
    count = wintypes.DWORD()
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None

    advapi32 = win_dll("Advapi32.dll")
    advapi32.CredEnumerateW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(ctypes.POINTER(cred_ptr)),
    ]
    advapi32.CredEnumerateW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [wintypes.LPVOID]

    try:
        if not advapi32.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(credentials)):
            return None
    except OSError as exc:
        logger.debug("Unable to enumerate Windows credentials for Copilot auth discovery: %s", exc)
        return None

    host = _github_host().lower()
    service_prefixes = [f"copilot-cli/{host}:"]
    if "://" not in host:
        service_prefixes.append(f"copilot-cli/https://{host}:")

    try:
        for idx in range(count.value):
            credential = credentials[idx].contents
            target = (credential.TargetName or "").strip().lower()
            if not any(target.startswith(prefix) for prefix in service_prefixes):
                continue
            if credential.CredentialBlobSize <= 0 or not credential.CredentialBlob:
                continue
            blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            token = blob.decode("utf-8", errors="replace").strip()
            if token:
                return token
    finally:
        if credentials:
            advapi32.CredFree(credentials)

    return None


def _parse_expiry(value: Any) -> float | None:
    if value in (None, ""):
        return None

    if isinstance(value, int | float):
        number = float(value)
        if number > 10_000_000_000:
            return number / 1000.0
        return number

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return _parse_expiry(int(raw))
        try:
            normalized = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return None

    return None


def _entry_expired(entry: dict[str, Any]) -> bool:
    for key in _EXPIRY_KEYS:
        expiry = _parse_expiry(entry.get(key))
        if expiry is None:
            continue
        return time.time() >= (expiry - _TOKEN_EXPIRY_BUFFER_S)
    return False


def _extract_oauth_token(entry: dict[str, Any]) -> str | None:
    if _entry_expired(entry):
        return None

    for key in _OAUTH_TOKEN_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in entry.values():
        if isinstance(value, dict):
            nested = _extract_oauth_token(value)
            if nested:
                return nested

    return None


def _iter_file_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                entries.append((str(key), value))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                key = str(value.get("host") or value.get("githubHost") or idx)
                entries.append((key, value))
    return entries


def read_cached_oauth_token() -> str | None:
    """Return a GitHub OAuth token for Copilot, if one is available."""

    for env_var in _OAUTH_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            return token

    windows_copilot_token = _read_windows_copilot_cli_oauth_token()
    if windows_copilot_token:
        return windows_copilot_token

    gh_token = _read_gh_cli_oauth_token()
    if gh_token:
        return gh_token

    host = _github_host()
    for path in _resolve_token_file_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug("Unable to read Copilot credentials file %s: %s", path, exc)
            continue

        for key, entry in _iter_file_entries(payload):
            if host not in key.lower():
                continue
            cached_token = _extract_oauth_token(entry)
            if cached_token:
                return cached_token

    return None


def resolve_client_bearer_token() -> str | None:
    """Return a bearer token suitable for satisfying Copilot provider auth checks."""

    for env_var in _API_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            return token
    return read_cached_oauth_token()


def has_oauth_auth() -> bool:
    """Return True when existing Copilot auth can be reused."""

    return resolve_client_bearer_token() is not None


def is_copilot_api_url(url: str | None) -> bool:
    """Return True when the upstream URL points at GitHub Copilot."""

    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower() or parsed.path.lower()
    return "githubcopilot.com" in host


def build_copilot_upstream_url(base_url: str, path: str) -> str:
    """Build an upstream URL, normalizing GitHub Copilot's non-/v1 path layout."""

    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if is_copilot_api_url(normalized_base) and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[3:]
    return f"{normalized_base}{normalized_path}"


class CopilotTokenProvider:
    """Resolve and cache short-lived Copilot API tokens."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: CopilotAPIToken | None = None

    async def get_api_token(self) -> CopilotAPIToken:
        explicit_api_token = os.environ.get("GITHUB_COPILOT_API_TOKEN", "").strip()
        if explicit_api_token:
            return CopilotAPIToken(
                token=explicit_api_token,
                expires_at=time.time() + 3600,
                api_url=os.environ.get("GITHUB_COPILOT_API_URL", DEFAULT_API_URL).strip()
                or DEFAULT_API_URL,
            )

        cached = self._cached
        if cached is not None and cached.is_valid:
            return cached

        async with self._lock:
            cached = self._cached
            if cached is not None and cached.is_valid:
                return cached

            oauth_token = read_cached_oauth_token()
            if not oauth_token:
                raise RuntimeError("No GitHub Copilot OAuth token is available.")

            if not _should_exchange_oauth_token():
                direct_token = CopilotAPIToken(
                    token=oauth_token,
                    expires_at=time.time() + 3600,
                    api_url=os.environ.get("GITHUB_COPILOT_API_URL", DEFAULT_API_URL).strip()
                    or DEFAULT_API_URL,
                )
                self._cached = direct_token
                return direct_token

            exchanged = await self._exchange_token(oauth_token)
            self._cached = exchanged
            return exchanged

    async def _exchange_token(self, oauth_token: str) -> CopilotAPIToken:
        headers = {
            "Authorization": f"token {oauth_token}",
            "Accept": "application/json",
            "Editor-Version": os.environ.get(
                "GITHUB_COPILOT_EDITOR_VERSION", _DEFAULT_EDITOR_VERSION
            ),
            "User-Agent": _DEFAULT_USER_AGENT,
        }
        payload = await asyncio.to_thread(self._exchange_token_sync, headers)
        token = str(payload.get("token") or "").strip()
        if not token:
            raise RuntimeError("Copilot token exchange returned an empty token.")

        expires_at = _parse_expiry(payload.get("expires_at")) or (time.time() + 1800)
        raw_endpoints = payload.get("endpoints")
        endpoints: dict[str, Any] = raw_endpoints if isinstance(raw_endpoints, dict) else {}
        api_url = str(endpoints.get("api") or DEFAULT_API_URL).strip() or DEFAULT_API_URL
        refresh_in = payload.get("refresh_in")
        sku = payload.get("sku")
        return CopilotAPIToken(
            token=token,
            expires_at=expires_at,
            api_url=api_url,
            refresh_in=int(refresh_in) if isinstance(refresh_in, int | float) else None,
            sku=str(sku) if isinstance(sku, str) and sku.strip() else None,
        )

    @staticmethod
    def _exchange_token_sync(headers: dict[str, str]) -> dict[str, Any]:
        request = urllib_request.Request(_token_exchange_url(), headers=headers, method="GET")
        try:
            with urllib_request.urlopen(request, timeout=10.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Copilot token exchange failed with HTTP {exc.code}: {body}"
            ) from exc


_provider: CopilotTokenProvider | None = None


def get_copilot_token_provider() -> CopilotTokenProvider:
    """Return the shared Copilot token provider."""

    global _provider
    if _provider is None:
        _provider = CopilotTokenProvider()
    return _provider


async def apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
    """Replace Authorization with a fresh Copilot API token when targeting Copilot."""

    resolved = dict(headers)
    if not is_copilot_api_url(url):
        return resolved

    token = await get_copilot_token_provider().get_api_token()
    for key in list(resolved):
        if key.lower() == "authorization":
            resolved.pop(key)
    resolved["Authorization"] = f"Bearer {token.token}"
    return resolved
