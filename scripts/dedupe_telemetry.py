#!/usr/bin/env python3
"""Deduplicate proxy_telemetry_v2 rows by session_id.

Keeps the latest entry (by created_at) per session and deletes older duplicates.
Run manually or via cron:

    # One-off
    python scripts/dedupe_telemetry.py

    # Hourly via crontab -e
    0 * * * * cd /Users/tchopra/claude-projects/headroom && python scripts/dedupe_telemetry.py >> /tmp/dedupe_telemetry.log 2>&1

    # Dry run (show what would be deleted)
    python scripts/dedupe_telemetry.py --dry-run
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

import requests

# Supabase config (same anon key as beacon — INSERT/SELECT/DELETE via RLS)
_SUPABASE_URL = "https://dtlllcsudcoasebbamcq.supabase.co"
_SUPABASE_KEY = ".".join(
    [
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR0bGxsY3N1ZGNvYXNlYmJhbWNxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM3MDc4NDUsImV4cCI6MjA4OTI4Mzg0NX0",
        "h_C6dLQKa8BVc3upgEvulR4E0K4eiEViyddRMIylKjU",
    ]
)
_TABLE = "proxy_telemetry_v2"
_ENDPOINT = f"{_SUPABASE_URL}/rest/v1/{_TABLE}"
_HEADERS = {
    "apikey": _SUPABASE_KEY,
    "Authorization": f"Bearer {_SUPABASE_KEY}",
}


def fetch_all_rows() -> list[dict]:
    """Fetch id, session_id, created_at for all rows."""
    rows = []
    offset = 0
    limit = 1000
    while True:
        resp = requests.get(
            _ENDPOINT,
            params={
                "select": "id,session_id,created_at",
                "order": "created_at.desc",
                "offset": offset,
                "limit": limit,
            },
            headers=_HEADERS,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return rows


def find_duplicates(rows: list[dict]) -> list[str]:
    """Find row IDs to delete (all but the latest per session_id)."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)

    ids_to_delete = []
    for _session_id, entries in by_session.items():
        if len(entries) <= 1:
            continue
        # Sort by created_at descending — keep first (latest), delete rest
        entries.sort(key=lambda r: r["created_at"], reverse=True)
        for entry in entries[1:]:
            ids_to_delete.append(entry["id"])

    return ids_to_delete


def delete_rows(ids: list[str], dry_run: bool = False) -> int:
    """Delete rows by ID. Returns count deleted."""
    if dry_run or not ids:
        return 0

    deleted = 0
    # PostgREST supports filtering by id in a list
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        id_filter = ",".join(batch)
        resp = requests.delete(
            _ENDPOINT,
            params={"id": f"in.({id_filter})"},
            headers={**_HEADERS, "Prefer": "return=minimal"},
        )
        resp.raise_for_status()
        deleted += len(batch)

    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate proxy_telemetry_v2 by session_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Fetching telemetry rows...")

    rows = fetch_all_rows()
    print(f"  Total rows: {len(rows)}")

    # Group stats
    by_session: dict[str, int] = defaultdict(int)
    for row in rows:
        by_session[row["session_id"]] += 1
    dup_sessions = {k: v for k, v in by_session.items() if v > 1}

    if not dup_sessions:
        print("  No duplicates found. Nothing to do.")
        return

    print(f"  Sessions with duplicates: {len(dup_sessions)}")
    total_dups = sum(v - 1 for v in dup_sessions.values())
    print(f"  Rows to delete: {total_dups}")

    ids_to_delete = find_duplicates(rows)

    if args.dry_run:
        print("\n  [DRY RUN] Would delete these rows:")
        for row in rows:
            if row["id"] in ids_to_delete:
                print(
                    f"    {row['id']}  session={row['session_id'][:12]}...  created={row['created_at']}"
                )
        print(f"\n  [DRY RUN] Would keep {len(rows) - len(ids_to_delete)} rows")
        return

    deleted = delete_rows(ids_to_delete)
    print(f"  Deleted: {deleted} duplicate rows")
    print(f"  Remaining: {len(rows) - deleted} rows (1 per session)")


if __name__ == "__main__":
    main()
