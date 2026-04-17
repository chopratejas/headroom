"""Loopback-only access guard for /debug/* endpoints.

Unit 5 of the Codex-proxy resilience plan. A FastAPI dependency that
raises :class:`fastapi.HTTPException` with status 404 — *not* 403 — for
any request whose client address is not the loopback interface. 404 is
deliberate: debug endpoints should be invisible to external scanners,
not merely forbidden.

The guard is a ``Depends(...)``-friendly function (rather than a
middleware) because:

* FastAPI's dependency injection makes the guard explicit on each
  route, so ``ruff``/reviewers can see which endpoints are guarded.
* ``TestClient`` lets us override a dependency with
  ``app.dependency_overrides``, which is the cleanest way to simulate
  a non-loopback client in tests.
* The set of debug endpoints is small and co-located; a middleware
  would be disproportionate.
"""

try:
    from fastapi import HTTPException, Request
except ImportError:  # pragma: no cover - fastapi is a hard dep in practice
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]


__all__ = [
    "LOOPBACK_HOSTS",
    "is_loopback_host",
    "require_loopback",
]


# Accepted loopback hostnames. ``None`` covers ``TestClient`` which does
# not populate ``request.client`` by default; we treat that as loopback
# because it means the call did not originate from a real socket.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


def is_loopback_host(host: str | None) -> bool:
    """Return True if ``host`` represents a loopback interface.

    ``None`` is treated as loopback — this covers ``TestClient``
    requests where FastAPI does not populate ``request.client``.
    """
    if host is None:
        return True
    return host in LOOPBACK_HOSTS


def require_loopback(request: Request) -> None:  # type: ignore[valid-type]
    """FastAPI dependency: 404 any non-loopback caller.

    Usage::

        @app.get("/debug/tasks", dependencies=[Depends(require_loopback)])
        async def debug_tasks() -> list[dict]:
            ...

    Returning 404 (not 403) keeps debug endpoints invisible to
    external scanners — indistinguishable from "no such route".
    """
    if HTTPException is None:  # pragma: no cover - defensive
        raise RuntimeError("FastAPI is required for the loopback guard")

    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not is_loopback_host(host):
        # No body: minimal FastAPI default, behaves like "no route".
        raise HTTPException(status_code=404)
