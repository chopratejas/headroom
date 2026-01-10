# Contributing to Headroom

Thank you for your interest in contributing to Headroom! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Environment details** (Python version, OS, Headroom version)
- **Code samples** or minimal reproduction if possible

### Suggesting Features

Feature requests are welcome! Please:

- Check existing issues/discussions first
- Clearly describe the use case and motivation
- Explain how it fits with Headroom's goals (context optimization, safety, determinism)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   pytest
   ```
6. **Run linting**:
   ```bash
   ruff check .
   ruff format .
   ```
7. **Update documentation** if needed
8. **Submit your PR** with a clear description

## Development Setup

```bash
# Clone the repository
git clone https://github.com/chopratejas/headroom.git
cd headroom

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode with all dependencies
pip install -e ".[dev,relevance,proxy]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=headroom --cov-report=html
```

## Coding Standards

### Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 100 characters
- Use type hints for all public functions
- Follow PEP 8 naming conventions

### Code Organization

```
headroom/
├── __init__.py          # Public API exports
├── client.py            # HeadroomClient wrapper
├── config.py            # Configuration dataclasses
├── transforms/          # Context transforms
│   ├── smart_crusher.py # Statistical compression
│   ├── cache_aligner.py # Cache optimization
│   └── rolling_window.py# Context windowing
├── relevance/           # Relevance scoring
├── providers/           # LLM provider adapters
├── proxy/               # Proxy server
└── storage/             # Metrics storage
```

### Testing

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Aim for >80% coverage on new code

Example test structure:
```python
class TestSmartCrusher:
    """Tests for SmartCrusher transform."""

    def test_compresses_large_arrays(self):
        """Should compress arrays above token threshold."""
        ...

    def test_preserves_errors(self):
        """Should never drop items containing errors."""
        ...
```

### Documentation

- Add docstrings to all public classes and functions
- Use Google-style docstrings
- Update README.md for user-facing changes
- Add examples for new features

```python
def compress_tool_output(
    content: str,
    max_items: int = 50,
) -> str:
    """Compress tool output while preserving important items.

    Args:
        content: The tool output content (usually JSON).
        max_items: Maximum items to keep in arrays.

    Returns:
        Compressed content string.

    Raises:
        ValueError: If content is not valid JSON.

    Example:
        >>> compress_tool_output('[{"id": 1}, {"id": 2}]', max_items=1)
        '[{"id": 1}]'
    """
```

## Pull Request Guidelines

### PR Title Format

Use conventional commit style:
- `feat: Add semantic caching to proxy`
- `fix: Handle empty tool outputs correctly`
- `docs: Update proxy documentation`
- `test: Add tests for CacheAligner`
- `refactor: Simplify rolling window logic`

### PR Description

Include:
- **What** changes were made
- **Why** the changes were needed
- **How** to test the changes
- **Breaking changes** if any

### Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting, type checking)
3. Maintain or improve test coverage
4. Update CHANGELOG.md for notable changes

## Architecture Decisions

### Safety First

Headroom's core principle is **safety**. When in doubt:
- Never drop user/assistant content
- Never break tool call/response pairing
- Malformed content passes through unchanged
- Prefer false negatives over false positives

### Performance

- Transforms should add <50ms latency at P99
- Use lazy loading for optional dependencies
- Profile before optimizing

### Compatibility

- Support Python 3.10+
- Core functionality has minimal dependencies
- Optional features use extras (e.g., `pip install headroom[relevance]`)


## Recognition

Contributors are recognized in:
- The CHANGELOG for their contributions
- The GitHub contributors page
- Release notes for significant features

Thank you for contributing to Headroom!
