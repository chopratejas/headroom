#!/usr/bin/env python3
"""
Show the actual before/after JSON - what gets sent to the LLM.
"""

import json

from headroom.config import SmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher


# Same realistic data as the API test
def create_search_results(n: int = 100) -> list:
    return [
        {
            "file": f"src/components/{['Button', 'Modal', 'Form', 'Table', 'Card'][i % 5]}.tsx",
            "line": 10 + (i * 3),
            "content": f"export function {['Button', 'Modal', 'Form', 'Table', 'Card'][i % 5]}Component{i}(props: Props) {{",
            "language": "typescript",
            "repository": "frontend-app",
            "branch": "main",
            "last_modified": "2024-12-15T10:00:00Z",
            "author": f"dev{i % 10}@company.com",
            "match_score": 0.95 - (i * 0.005),
            "context": {
                "before": ["import React from 'react';", "import { useCallback } from 'react';"],
                "after": ["  return <div>...</div>;", "}"],
            },
            "metadata": {
                "size_bytes": 1500 + (i * 10),
                "encoding": "utf-8",
                "mime_type": "text/typescript",
            },
        }
        for i in range(n)
    ]


def main():
    # Create the crusher
    config = SmartCrusherConfig()
    crusher = SmartCrusher(config)

    # Original data
    original_data = create_search_results(100)
    original_json = json.dumps(original_data, indent=2)

    print("=" * 70)
    print("BEFORE: Original Tool Output (first 2 items shown)")
    print("=" * 70)
    print(json.dumps(original_data[:2], indent=2))
    print(f"\n... plus {len(original_data) - 2} more items (100 total)")
    print(f"\nTotal characters: {len(original_json):,}")

    # Compress it using the crush method
    result = crusher.crush(
        content=json.dumps(original_data), query="Find all React form components"
    )

    compressed_content = result.compressed
    compressed_data = json.loads(compressed_content)

    print("\n" + "=" * 70)
    print("AFTER: Compressed Tool Output (all items shown)")
    print("=" * 70)
    print(json.dumps(compressed_data, indent=2))

    print("\n" + "=" * 70)
    print("COMPRESSION STATS")
    print("=" * 70)
    print(f"Items before: {len(original_data)}")
    print(f"Items after:  {len(compressed_data)}")
    print(f"Characters before: {len(original_json):,}")
    print(f"Characters after:  {len(compressed_content):,}")
    print(f"Reduction: {(1 - len(compressed_content) / len(original_json)) * 100:.1f}%")

    # Show what was kept and why
    print("\n" + "=" * 70)
    print("WHAT WAS KEPT AND WHY")
    print("=" * 70)

    for i, item in enumerate(compressed_data):
        file_name = item.get("file", "unknown")
        score = item.get("match_score", 0)
        reason = (
            "first" if i < 3 else ("last" if i >= len(compressed_data) - 2 else "high relevance")
        )
        print(f"  {i + 1}. {file_name} (score: {score:.2f}) - {reason}")


if __name__ == "__main__":
    main()
