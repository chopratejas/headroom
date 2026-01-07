"""Storage modules for Headroom SDK."""

from .base import Storage
from .jsonl import JSONLStorage
from .sqlite import SQLiteStorage

__all__ = [
    "Storage",
    "SQLiteStorage",
    "JSONLStorage",
]


def create_storage(store_url: str) -> Storage:
    """
    Create a storage instance from URL.

    Supported URLs:
    - sqlite:///path/to/file.db
    - jsonl:///path/to/file.jsonl

    Args:
        store_url: Storage URL.

    Returns:
        Storage instance.
    """
    if store_url.startswith("sqlite://"):
        path = store_url.replace("sqlite://", "")
        # Handle sqlite:/// (3 slashes for absolute path)
        if path.startswith("/"):
            path = path  # Already absolute
        return SQLiteStorage(path)
    elif store_url.startswith("jsonl://"):
        path = store_url.replace("jsonl://", "")
        if path.startswith("/"):
            path = path
        return JSONLStorage(path)
    else:
        # Default to SQLite
        return SQLiteStorage(store_url)
