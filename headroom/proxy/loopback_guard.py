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

from __future__ import annotations

import ipaddress
import os as _os

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


# Legacy canonical loopback literal set. Retained for backwards
# compatibility with callers/tests that still import it; the real check
# now goes through :func:`ipaddress.ip_address(...).is_loopback` so we
# also accept IPv6-mapped IPv4 (``::ffff:127.0.0.1``) and other valid
# loopback literals on dual-stack sockets.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

# One-time warning guard: set to True after the first None-client warning
# is emitted so we don't flood logs on every request in affected deployments.
_WARNED_NONE_CLIENT: bool = False


def is_loopback_host(host: str | None) -> bool:
    """Return True if ``host`` represents a loopback interface.

    ``None`` is treated as loopback — this covers ``ASGITransport`` /
    UDS-style requests where FastAPI does not populate
    ``request.client``.

    ``"localhost"`` is special-cased as a string since it is not a
    valid IP literal. Every other host is parsed with
    :func:`ipaddress.ip_address`; this accepts IPv6-mapped IPv4
    (``::ffff:127.0.0.1``) which Linux dual-stack sockets emit by
    default. Malformed input returns ``False``.

    Production note: the guard only inspects ``scope["client"]``, which
    uvicorn / hypercorn populate from the TCP peer address of the socket.
    Reverse-proxy ``X-Forwarded-For`` headers are deliberately ignored —
    enabling that would let a network attacker bypass the guard with a
    spoofed forwarded address. Operators terminating TLS in front of the
    proxy must place trusted middleware between the proxy and untrusted
    networks.

    Note for tests: Starlette's ``TestClient`` sets ``scope["client"]``
    to the sentinel ``("testclient", 50000)``. That sentinel is **not**
    treated as loopback here — tests must instead bypass the guard via
    ``app.dependency_overrides[require_loopback] = lambda: None`` (the
    same pattern used for ``/v1/responses`` integration tests). Adding
    ``"testclient"`` to the trusted set would silently expand the
    attack surface if a deployment ever wired the test transport into
    a real ASGI server.
    """
    global _WARNED_NONE_CLIENT
    if host is None:
        if (
            not _WARNED_NONE_CLIENT
            and not _os.environ.get("PYTEST_CURRENT_TEST")
            and not _os.environ.get("HEADROOM_ALLOW_NONE_CLIENT")
        ):
            import logging as _logging

            _logging.getLogger("headroom.loopback_guard").warning(
                "Trusting request with no client address (None host). "
                "Set HEADROOM_ALLOW_NONE_CLIENT=1 to suppress this warning."
            )
            _WARNED_NONE_CLIENT = True
        return True
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


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
