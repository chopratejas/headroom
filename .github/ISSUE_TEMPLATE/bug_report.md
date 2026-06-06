---
name: Bug Report
about: Report a bug to help us improve Headroom
title: '[BUG] '
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## Environment

- **Headroom version**: (run `python -c "import headroom; print(headroom.__version__)"`)
- **Python version**: (run `python --version`)
- **OS**: (e.g., macOS 14.0, Ubuntu 22.04, Windows 11)
- **Install mode**: (pip, pipx, Docker, native wrapper, editable checkout)
- **Client / wrapper**: (e.g., Claude Code, Codex, Cursor, Copilot CLI, direct SDK)
- **LLM Provider**: (e.g., OpenAI, Anthropic, Bedrock, Copilot subscription)

## Reproduction

Steps to reproduce the behavior:

1. Install headroom with '...'
2. Run this code '...'
3. See error

Please include the smallest command, script, request body, config, or log snippet that still reproduces the issue.

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Code Sample

```python
# Minimal code to reproduce the issue
from headroom import HeadroomClient

# Your code here
```

## Error Output

```
Paste any error messages or stack traces here
```

## Compatibility Notes

- Does this happen only on a specific OS, shell, provider, model, install mode, wrapper, Docker/native mode, or Python version?
- Did this work in an older Headroom version?

## Regression Test Idea

If you know where this could be covered, mention the narrowest useful test layer: unit, integration, e2e, docs example, or manual reproduction.

## Additional Context

Add any other context about the problem here (logs, screenshots, etc.)
