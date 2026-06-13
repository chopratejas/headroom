"""PR Governance validator for pull_request_target CI.

Reads the GitHub event JSON, validates the PR body against the project
template, and writes a JSON report consumed by the pr-health.yml workflow.

Exit codes:
    0 -- report written (valid or not; failure signalled via 'valid' output)
    1 -- unexpected runtime error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

COMMENT_MARKER = "<!-- pr-governance-bot -->"

PLACEHOLDER_PATTERNS = [
    r"Brief description of changes and motivation\.",
    r"Change 1\b",
    r"Paste relevant test output here",
    r"Add screenshots to help explain your changes\.",
    r"Any additional information that reviewers should know\.",
    r"Fixes #\(issue number\)",
]


def _set_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    else:
        print(f"::set-output name={name}::{value}", file=sys.stderr)


def _is_bot(author_login: str, author_type: str) -> bool:
    return author_type == "Bot" or author_login.endswith("[bot]")


def _count_checkboxes(body: str) -> tuple[int, int]:
    """Return (checked, total) checkbox counts."""
    checked = len(re.findall(r"- \[x\]", body, re.IGNORECASE))
    unchecked = len(re.findall(r"- \[ \]", body))
    return checked, checked + unchecked


def _has_placeholder(body: str) -> str | None:
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, body):
            return pattern
    return None


def _has_headings(body: str) -> bool:
    return bool(re.search(r"^##\s+\S", body, re.MULTILINE))


def validate(pr: dict) -> dict:  # type: ignore[type-arg]
    """Return a validation result dict."""
    body: str = pr.get("body") or ""
    author_login: str = pr.get("user", {}).get("login", "")
    author_type: str = pr.get("user", {}).get("type", "User")
    pr_number: int = pr.get("number", 0)
    pr_title: str = pr.get("title", "")

    is_bot = _is_bot(author_login, author_type)
    placeholder = _has_placeholder(body)
    checked, total = _count_checkboxes(body)

    body_too_short = len(body.strip()) < 50
    all_boxes_unchecked = total > 0 and checked == 0
    no_structure = not _has_headings(body)

    problems: list[str] = []
    if body_too_short:
        problems.append("PR body is empty or too short.")
    if placeholder:
        problems.append(f"Placeholder text still present: {placeholder!r}")
    if no_structure and not is_bot:
        problems.append("PR body has no section headings -- please use the template.")
    if all_boxes_unchecked and not is_bot:
        problems.append("All checklist checkboxes are unchecked -- please review the checklist.")

    valid = len(problems) == 0

    labels_to_add: list[str] = []
    labels_to_remove: list[str] = []
    if valid:
        labels_to_add.append("status: ready for review")
        labels_to_remove.append("status: needs author action")
    else:
        labels_to_add.append("status: needs author action")
        labels_to_remove.append("status: ready for review")

    status_word = "PASS" if valid else "FAIL"
    status_line = "PR template looks good." if valid else "PR body needs attention before review."

    problem_md = ""
    if problems:
        items = "\n".join(f"- {p}" for p in problems)
        problem_md = f"\n\n**Issues found:**\n{items}"

    checklist_md = f"\n\n**Checklist:** {checked}/{total} items checked." if total > 0 else ""

    comment_markdown = (
        f"## PR Governance: {status_word}\n\n"
        f"**PR #{pr_number} -- {pr_title}**\n\n"
        f"{status_line}{problem_md}{checklist_md}\n\n"
        f"<sub>Automated check - update the PR body to re-trigger</sub>"
    )

    summary_markdown = f"### PR Governance -- {status_word}\n\n{status_line}{problem_md}"

    return {
        "valid": valid,
        "is_bot_pr": is_bot,
        "problems": problems,
        "labels_to_add": labels_to_add,
        "labels_to_remove": labels_to_remove,
        "comment_marker": COMMENT_MARKER,
        "comment_markdown": comment_markdown,
        "summary_markdown": summary_markdown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PR template compliance.")
    parser.add_argument("--event", required=True, help="Path to GitHub event JSON")
    parser.add_argument("--report", required=True, help="Output path for JSON report")
    args = parser.parse_args(argv)

    try:
        event = json.loads(Path(args.event).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading event file: {exc}", file=sys.stderr)
        return 1

    pr = event.get("pull_request", {})
    if not pr:
        result: dict = {  # type: ignore[type-arg]
            "valid": True,
            "is_bot_pr": False,
            "problems": [],
            "labels_to_add": [],
            "labels_to_remove": [],
            "comment_marker": COMMENT_MARKER,
            "comment_markdown": "",
            "summary_markdown": ("### PR Governance\n\nNo pull_request in event -- skipped."),
        }
    else:
        result = validate(pr)

    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")

    _set_output("valid", "true" if result["valid"] else "false")
    _set_output("is_bot_pr", "true" if result["is_bot_pr"] else "false")

    for problem in result["problems"]:
        print(f"  FAIL: {problem}")
    if result["problems"]:
        print()

    print(f"PR Governance: {'PASS' if result['valid'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
