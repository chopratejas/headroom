"""Centralised environment variable lookup for Headroom.

All ``HEADROOM_*`` env vars should be read through :func:`get_hr_env` so that:

* The canonical ``HR_*`` prefix is always checked first.
* The legacy ``HEADROOM_*`` prefix continues to work but emits a
  :class:`DeprecationWarning` once per process.

Usage::

    from headroom.env import get_hr_env

    api_key = get_hr_env("API_KEY")          # reads HR_API_KEY or HEADROOM_API_KEY
    mode = get_hr_env("MODE", "token")       # with a default
"""

from __future__ import annotations

import logging
import os
import warnings

_env_logger = logging.getLogger("headroom.env")

# Track which old keys have already been warned so we only warn once per
# process (not on every env read in hot paths).
_deprecation_warned: set[str] = set()


def get_hr_env(suffix: str, default: str | None = None) -> str | None:
    """Read ``HR_<suffix>``, falling back to ``HEADROOM_<suffix>`` with a deprecation warning.

    Parameters
    ----------
    suffix:
        The part after the prefix, e.g. ``"API_KEY"`` reads ``HR_API_KEY``
        and falls back to ``HEADROOM_API_KEY``.
    default:
        Value returned when neither variable is set.  Mirrors the
        ``os.environ.get(key, default)`` signature.

    Returns
    -------
    str | None
        The resolved value, or *default* if neither variable is present.
    """
    new_key = f"HR_{suffix}"
    old_key = f"HEADROOM_{suffix}"

    val = os.environ.get(new_key)
    if val is not None:
        return val

    val = os.environ.get(old_key)
    if val is not None:
        if old_key not in _deprecation_warned:
            _deprecation_warned.add(old_key)
            warnings.warn(
                f"{old_key} is deprecated — rename to {new_key}. "
                "Support for HEADROOM_* prefixes will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            _env_logger.warning("Deprecated env var %s — please rename to %s", old_key, new_key)
        return val

    return default
