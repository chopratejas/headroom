"""rtk (Rust Token Killer) integration for Headroom.

rtk compresses CLI output (test results, git diffs, log dumps) before it
enters the LLM context window. Headroom downloads and manages the rtk binary.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

RTK_VERSION = "v0.28.2"
RTK_BIN_DIR = Path.home() / ".headroom" / "bin"
_RTK_NAME = "rtk.exe" if platform.system() == "Windows" else "rtk"
RTK_BIN_PATH = RTK_BIN_DIR / _RTK_NAME


def get_rtk_path() -> Path | None:
    """Get path to rtk binary — check PATH first, then ~/.headroom/bin/."""
    # Check if rtk is already in PATH (e.g., installed via brew)
    system_rtk = shutil.which("rtk")
    if system_rtk:
        return Path(system_rtk)

    # Check Headroom-managed install
    if RTK_BIN_PATH.exists() and RTK_BIN_PATH.is_file():
        return RTK_BIN_PATH

    return None


def is_rtk_installed() -> bool:
    """Check if rtk is available."""
    return get_rtk_path() is not None
