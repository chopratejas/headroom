#!/usr/bin/env python3
"""Verify all package versions are in sync before publishing."""

import json
from pathlib import Path

import tomllib

ROOT = Path(__file__).parent.parent


def main():
    with open(ROOT / "pyproject.toml", "rb") as f:
        py_ver = tomllib.load(f)["project"]["version"]

    with open(ROOT / "plugins/openclaw/package.json") as f:
        npm_openclaw_ver = json.load(f)["version"]

    with open(ROOT / "sdk/typescript/package.json") as f:
        npm_sdk_ver = json.load(f)["version"]

    headroom_ver = (
        (ROOT / "headroom" / "_version.py").read_text().split('__version__ = "')[1].split('"')[0]
    )

    versions = {
        "pyproject.toml": py_ver,
        "openclaw/package.json": npm_openclaw_ver,
        "typescript/package.json": npm_sdk_ver,
        "headroom/_version.py": headroom_ver,
    }

    if not all(v == py_ver for v in versions.values()):
        print("Version mismatch detected:")
        for file, ver in versions.items():
            print(f"  {file}: {ver}")
        print(f"Expected all to be: {py_ver}")
        raise SystemExit(1)

    print(f"All versions aligned at {py_ver}")
    print("Packages:", ", ".join(versions.keys()))


if __name__ == "__main__":
    main()
