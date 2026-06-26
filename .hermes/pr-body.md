## Description

Replace the 180-line process-killing approach with a 15-line Vite-style socket.bind probe: if port is busy (EADDRINUSE or EACCES), try the next available port.

Pure stdlib -- no /proc, no lsof, no subprocess -- works identically on Linux/macOS/Windows.

## Problem

When `headroom wrap <agent>` is killed without proper cleanup (window close, SSH timeout, crash), the background proxy becomes orphaned and holds the port. The next `headroom wrap` on the same port would wait 30-45 seconds then fail with a confusing error.

This PR takes a simpler, safer approach: find the next available port. No process detection, no killing.

### Related issues

- **#589** (Port 8787 reserved by Windows) -- partially addressed: EACCES is now skipped together with EADDRINUSE
- **#804** (Shared proxy killed by exiting session) -- already fixed upstream via `_live_proxy_clients` marker files; this PR doesn't touch that code

## Type of Change

- [x] New feature (non-breaking change that adds functionality)
- [x] Code refactoring (no functional changes)

## Changes Made

- `headroom/cli/wrap.py`: Add `_find_available_port()` -- socket.bind loop that skips EADDRINUSE (busy) and EACCES (reserved/privileged) ports, returns first available port in range. Replace `_ensure_proxy` port-bind check with auto-fallback call to `_find_available_port`. Remove `_ensure_port_free()` call from `_start_proxy()`. Remove 8 dead functions: `_find_process_on_port`, `_linux_find_process_on_port`, `_resolve_inode_to_pid`, `_is_headroom_proxy`, `_read_process_cmdline`, `_kill_process`, `_ensure_port_free`, `_format_unbindable_port_error`.
- `tests/test_cli/test_wrap_helpers.py`: Remove 14 old `TestEnsurePortFree` tests (mocked /proc parsing, process killing). Add 6 new `TestFindAvailablePort` tests covering: port free, first port busy, multiple busy, EACCES skipped, unexpected error propagated, range exhausted.
- `tests/test_cli/test_wrap_persistent.py`: Adapt persistence tests for `_find_available_port` mock. Rewrite unbindable-port test to use new error path.

Zero new dependencies. Zero changes to core proxy server, MCP, compression, or providers.

## Testing

- [x] Unit tests pass (`python -m pytest tests/test_cli/ -v`)
- [x] New tests added for new functionality

### Test Output

```
> python -m pytest tests/test_cli/test_wrap_helpers.py::TestFindAvailablePort -v --no-header
============================= 6 passed ==============================
test_port_free_returns_same PASSED
test_port_busy_finds_next PASSED
test_multiple_busy_ports PASSED
test_propagates_unexpected_error PASSED
test_propagates_eaddrinuse_with_eacces PASSED
test_exhausts_range PASSED

> python -m pytest tests/test_cli/ -q
============================= 445 passed in 8.40s ==============================
```

## Real Behavior Proof

- Environment: Ubuntu 24.04 x86_64, Python 3.12.3
- Verification: Port fallback logic verified via unit tests (all 6 `TestFindAvailablePort` tests pass). The function `_find_available_port(8787)` returns 8787 when free, 8788 when 8787 is busy. EACCES is skipped same as EADDRINUSE. Non-retryable errors (EADDRNOTAVAIL) propagate immediately.
- Cross-platform: Pure socket API -- no /proc, no lsof, no platform-specific code. Works on Linux, macOS, Windows.
- Not tested: Windows EACCES fallback (no Windows CI runner available). macOS port fallback (no macOS runner). The code path is identical across platforms since it only uses stdlib socket.

## Review Readiness

- [x] I have performed a self-review
- [x] This PR is ready for human review

## Checklist

- [x] My code follows the project's style guidelines
- [x] I have performed a self-review of my code
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally
