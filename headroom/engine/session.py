from __future__ import annotations

import hashlib
import hmac


def derive_session_key(
    *, credential: str | None, conversation_scope: str | None, salt: bytes
) -> str:
    """Pinned session_key derivation. Every host MUST call this so the proxy and any
    embedded adapter key CCR/session/drift state identically. The raw credential is
    hashed, never stored. A falsy ``conversation_scope`` (None or "") yields a
    tenant-scoped key (session == tenant)."""
    tenant_principal = hashlib.pbkdf2_hmac(
        "sha256",
        (credential or "").encode(),
        salt,
        100_000,
    ).hex()
    scope = conversation_scope or tenant_principal
    return hmac.new(salt, f"{tenant_principal}:{scope}".encode(), hashlib.sha256).hexdigest()
