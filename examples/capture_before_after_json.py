#!/usr/bin/env python3
"""Capture the actual before/after JSON for the README."""

import json

from headroom.config import SmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher

# The critical error (needle)
CRITICAL_ERROR = {
    "timestamp": "2024-12-15T03:47:23Z",
    "level": "FATAL",
    "service": "payment-gateway",
    "message": "Connection pool exhausted - max_connections=100 exceeded",
    "error_code": "PG-5523",
    "resolution": "Increase max_connections to 500 in config/database.yml",
    "affected_transactions": 1847,
}


def create_log_entries(n: int = 100, needle_position: int = 67) -> list:
    """Create n log entries with the critical error at needle_position."""
    logs = []
    for i in range(n):
        if i == needle_position:
            logs.append(CRITICAL_ERROR)
        else:
            logs.append(
                {
                    "timestamp": f"2024-12-15T{(i % 24):02d}:{(i % 60):02d}:00Z",
                    "level": "INFO",
                    "service": ["api-gateway", "user-service", "inventory", "auth"][i % 4],
                    "message": f"Request processed successfully - latency={50 + (i % 100)}ms",
                    "request_id": f"req-{i:06d}",
                    "status_code": 200,
                }
            )
    return logs


def main():
    logs = create_log_entries(100, needle_position=67)
    original_json = json.dumps(logs)

    # Compress with SmartCrusher
    config = SmartCrusherConfig()
    crusher = SmartCrusher(config)
    result = crusher.crush(original_json, query="error outage fatal")

    compressed_data = json.loads(result.compressed)

    print("=" * 70)
    print("BEFORE: First 3 of 100 log entries")
    print("=" * 70)
    print(json.dumps(logs[:3], indent=2))
    print(f"\n... plus 97 more entries (100 total, {len(original_json):,} chars)")

    print("\n" + "=" * 70)
    print(f"AFTER: Headroom keeps {len(compressed_data)} entries")
    print("=" * 70)
    print(json.dumps(compressed_data, indent=2))

    # Check if the needle was preserved
    print("\n" + "=" * 70)
    print("NEEDLE PRESERVED?")
    print("=" * 70)
    needle_found = any(item.get("error_code") == "PG-5523" for item in compressed_data)
    print(f"Critical error (PG-5523) in compressed output: {needle_found}")

    # Stats
    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)
    print(f"Items: {len(logs)} → {len(compressed_data)}")
    print(f"Chars: {len(original_json):,} → {len(result.compressed):,}")
    print(f"Reduction: {(1 - len(result.compressed) / len(original_json)) * 100:.1f}%")


if __name__ == "__main__":
    main()
