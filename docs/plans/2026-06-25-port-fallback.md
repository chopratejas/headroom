# Port Fallback: Auto-increment When Port Is Busy

**代替 PR #1385 的杀进程方案。** 改为 Vite 风格的纯 socket 探测 + 端口 +1 策略。

## 设计原则

- **在 `_ensure_proxy` 中 resolve port，`_start_proxy` 不操心端口选择**
- `_start_proxy` 保持 `-> subprocess.Popen`，不改返回类型
- `_ensure_proxy` 保持 `-> subprocess.Popen | None`，不改返回类型
- 调用方（3 个 caller）不需任何改动
- 纯 socket.bind，Linux/macOS/Windows 同一套代码

## Task 1: 新增 `_find_available_port`

文件：`headroom/cli/wrap.py`

在文件顶部的 import 块中添加 `import errno`（Python 标准库，放在 `import io` 后面，和其他 stdlib import 一起）。

`_find_available_port` 函数放在 `_port_bind_error` 函数后面（~line 336）：

```python
def _find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find first available port >= start_port via socket.bind probe.
    
    Only catches EADDRINUSE; other OS errors (EACCES on reserved ports,
    EADDRNOTAVAIL) propagate immediately.
    Raises RuntimeError when no port is found in range.
    """
    for port in range(start_port, start_port + max_attempts):
        error = _port_bind_error(port)
        if error is None:
            return port
        if error.errno != errno.EADDRINUSE:
            raise error  # Permission denied, propagate the actual OSError
    raise RuntimeError(
        f"No available port found in range "
        f"{start_port}-{start_port + max_attempts - 1}"
    )
```

**关键：第 9 行用 `raise error` 而不是 bare `raise`**（`_port_bind_error` 返回异常对象，不是 `except` 块内的异常）。

**验收标准：**
- [ ] 端口空闲时 `_find_available_port(8787)` 返回 8787
- [ ] 端口被占时返回第一个空闲端口
- [ ] 非 EADDRINUSE 错误用 `raise error` 传播
- [ ] 范围耗尽时抛 RuntimeError

## Task 2: 修改 `_ensure_proxy` — 用 `_find_available_port` 替换 `_port_bind_error` 检查

文件：`headroom/cli/wrap.py`，函数末尾 (~line 2754-2784)

将这段代码：

```python
        # Start (or restart) the proxy with the requested flags
        bind_error = helpers._port_bind_error(port)
        if bind_error is not None:
            raise click.ClickException(
                helpers._format_unbindable_port_error(port, bind_error, agent_type)
            )

        click.echo(f"  Starting Headroom proxy on port {port}...")
        try:
            proc = cast(
                subprocess.Popen[Any],
                _live_wrap_module()._start_proxy(
                    port,
                    learn=learn,
                    memory=memory,
                    agent_type=agent_type,
                    code_graph=code_graph,
                    backend=backend,
                    anyllm_provider=anyllm_provider,
                    region=region,
                    openai_api_url=openai_api_url,
                    anthropic_api_url=anthropic_api_url,
                    copilot_api_token=copilot_api_token,
                ),
            )
            click.echo(f"  Proxy ready on http://127.0.0.1:{port}")
            click.echo(f"  Dashboard:    http://127.0.0.1:{port}/dashboard")
            return proc
        except RuntimeError as e:
            click.echo(f"  Error: {e}")
            raise SystemExit(1) from e
```

替换为：

```python
        # Find available port (may differ from original if occupied)
        try:
            actual_port = helpers._find_available_port(port)
        except OSError as e:
            raise click.ClickException(
                f"Port {port} is unavailable: {e}"
            )
        except RuntimeError as e:
            raise click.ClickException(str(e))

        if actual_port != port:
            click.echo(
                f"  Port {port} is in use, using port {actual_port} instead."
            )

        click.echo(f"  Starting Headroom proxy on port {actual_port}...")
        try:
            proc = cast(
                subprocess.Popen[Any],
                _live_wrap_module()._start_proxy(
                    actual_port,
                    learn=learn,
                    memory=memory,
                    agent_type=agent_type,
                    code_graph=code_graph,
                    backend=backend,
                    anyllm_provider=anyllm_provider,
                    region=region,
                    openai_api_url=openai_api_url,
                    anthropic_api_url=anthropic_api_url,
                    copilot_api_token=copilot_api_token,
                ),
            )
            click.echo(f"  Proxy ready on http://127.0.0.1:{actual_port}")
            click.echo(f"  Dashboard:    http://127.0.0.1:{actual_port}/dashboard")
            return proc
        except RuntimeError as e:
            click.echo(f"  Error: {e}")
            raise SystemExit(1) from e
```

变更点：
1. 移除 `_port_bind_error` + `_format_unbindable_port_error` 调用
2. 插入 `_find_available_port` resolve 端口
3. `_start_proxy(actual_port, ...)` 传入 resolve 后的端口
4. 打印信息使用 `actual_port`
5. 保留原有的 `try/except RuntimeError` 错误处理

**验收标准：**
- [ ] `_ensure_proxy` 返回类型不变（`Popen | None`）
- [ ] 端口被占时 resolve 到可用端口再启动
- [ ] 端口切换时打印提示
- [ ] 非 EADDRINUSE 错误包装为 click.ClickException

## Task 3: 清理 `_start_proxy` — 移除 `_ensure_port_free` 调用

文件：`headroom/cli/wrap.py`

在 `_start_proxy` 函数开头（~line 562），移除：

```python
    if not _ensure_port_free(port):
        raise click.ClickException(
            f"Port {port} is in use by a non-headroom process. "
            f"Please free it manually or use --port to specify a different port."
        )
```

函数签名和返回类型不变（`-> subprocess.Popen`）。

**验收标准：**
- [ ] 不再调用 `_ensure_port_free` 或任何已删除函数
- [ ] 返回类型保持 `subprocess.Popen`
- [ ] lint 通过

## Task 4: 移除已废弃函数 + 清理 import

从 `headroom/cli/wrap.py` 中移除 8 个函数：

- `_find_process_on_port` (line ~360)
- `_linux_find_process_on_port` (line ~388)
- `_resolve_inode_to_pid` (line ~424)
- `_is_headroom_proxy` (line ~449)
- `_read_process_cmdline` (line ~464)
- `_kill_process` (line ~492)
- `_ensure_port_free` (line ~513)
- `_format_unbindable_port_error` (line ~333) — 唯一调用点已移除

import 检查：
- `signal` — 仍被 `_make_cleanup`、`_kill_proxy_by_pid` 等处使用，**保留**
- `time` — 仍被多处使用，**保留**
- `os` — 仍被多处使用，**保留**
- `Path` — 仍被多处使用，**保留**
- `subprocess` — 仍被多处使用，**保留**

不需要新增 import（`errno` 已在 Task 1 中添加）。

**验收标准：**
- [ ] 所有 8 个函数被移除
- [ ] 没有死代码残留
- [ ] lint 通过

## Task 5: 重写测试

### 5a: 移除旧测试 + 添加新测试

文件：`tests/test_cli/test_wrap_helpers.py`

移除 `TestEnsurePortFree`（lines 794-1032，14 个测试）。

在文件末尾添加 `TestFindAvailablePort`（5 个测试）：

```python
import errno  # 文件顶部添加（如果还没有）

class TestFindAvailablePort:
    """Tests for _find_available_port (Vite-style port fallback)."""

    def test_port_free_returns_same(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When port is free, returns the same port."""
        monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda port: None)
        assert wrap_mod._find_available_port(8787) == 8787

    def test_port_busy_finds_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When port is busy, returns the next free port."""
        def mock_bind(port: int) -> OSError | None:
            if port == 8787:
                return OSError(errno.EADDRINUSE, "Address in use")
            return None
        monkeypatch.setattr(wrap_mod, "_port_bind_error", mock_bind)
        assert wrap_mod._find_available_port(8787) == 8788

    def test_multiple_busy_ports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When multiple consecutive ports are busy, skips all of them."""
        def mock_bind(port: int) -> OSError | None:
            if port in (8787, 8788, 8789):
                return OSError(errno.EADDRINUSE, "Address in use")
            return None
        monkeypatch.setattr(wrap_mod, "_port_bind_error", mock_bind)
        assert wrap_mod._find_available_port(8787) == 8790

    def test_propagates_non_eaddrinuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-EADDRINUSE errors (EACCES) propagate immediately."""
        monkeypatch.setattr(
            wrap_mod, "_port_bind_error",
            lambda port: OSError(errno.EACCES, "Permission denied"),
        )
        with pytest.raises(OSError, match="Permission denied"):
            wrap_mod._find_available_port(8787)

    def test_exhausts_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all ports in range are busy, raises RuntimeError."""
        monkeypatch.setattr(
            wrap_mod, "_port_bind_error",
            lambda port: OSError(errno.EADDRINUSE, "Address in use"),
        )
        with pytest.raises(RuntimeError, match="No available port found"):
            wrap_mod._find_available_port(8787, max_attempts=3)
```

### 5b: 更新 test_wrap_persistent.py

文件：`tests/test_cli/test_wrap_persistent.py`

**`test_ensure_proxy_falls_back_when_persistent_manifest_is_stale`（line 96）：**
`_start_proxy` 签名不变（`-> Popen`），mock 不变。但需要 mock `_find_available_port`：
```python
monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)  # 已有
monkeypatch.setattr(wrap_cli, "_find_available_port", lambda port, **kw: port)  # 新增
```

**`test_ensure_proxy_reports_unbindable_port_before_starting_subprocess`（lines 104-126）：**
重写这个测试。`_port_bind_error` mock → `_find_available_port` mock 抛 EACCES：
```python
def test_ensure_proxy_reports_unbindable_port_before_starting_proxy(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(
        wrap_cli,
        "_find_available_port",
        lambda port, **kw: (_ for _ in ()).throw(
            OSError(errno.EACCES, "access denied by OS port reservation")
        ),
    )
    monkeypatch.setattr(wrap_cli, "_start_proxy", lambda *args, **kwargs: calls.append("start"))

    try:
        wrap_cli._ensure_proxy(8787, False, agent_type="cursor")
    except click.ClickException as exc:
        message = str(exc)
    else:
        raise AssertionError("expected unbindable port to raise before starting proxy")

    assert "Port 8787 is unavailable" in message
    assert calls == []
```

需要添加 `import errno` 到文件顶部。

**`test_ensure_proxy_restarts_idle_stale_ephemeral_proxy`（line 270）：**
```python
# _ensure_proxy 现在需要 _find_available_port
monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)  # 已有
monkeypatch.setattr(wrap_cli, "_find_available_port", lambda port, **kw: port)  # 新增
```

检查其他测试（lines 306, 344, 374, 420, 451, 486, 520, 553）：
这些测试在 mock `_start_proxy` 的同时也 mock 了 `_port_bind_error`，需要同步 mock `_find_available_port`。

要修改的测试：
- `test_ensure_proxy_restarts_ephemeral_proxy_for_openai_api_url_mismatch` (line 283)
- `test_ensure_proxy_restarts_ephemeral_proxy_for_foundry_url_mismatch` (line 320)
- `test_ensure_proxy_restarts_ephemeral_proxy_for_anthropic_url_mismatch` (line 356)
- `test_ensure_proxy_restarts_ephemeral_proxy_for_version_mismatch` (line 385)
- `test_ensure_proxy_restarts_ephemeral_proxy_for_missing_flags_solo` (line 435)
- `test_ensure_proxy_reports_non_restartable_non_pid_proxy` (line 465)
- `test_ensure_proxy_skips_restart_for_attached_wrappers` (line 500)
- `test_ensure_proxy_skips_restart_for_mixed_attached_and_missing` (line 535)
- `test_ensure_proxy_parses_config_status_ok` (line 558) — 可能不需改动

### 5c: test_wrap_codex.py 不改动

`_start_proxy` 签名不变（`-> Popen`），测试不需改动。

**验收标准：**
- [ ] 所有 14 个旧测试被移除
- [ ] 5 个新测试覆盖正常/被占/多端口/非 EADDRINUSE/耗尽
- [ ] test_wrap_persistent.py 全部测试通过
- [ ] test_wrap_codex.py 全部测试通过

## Task 6: 验证

```bash
cd /home/ubuntu/headroom

# 运行相关测试
python -m pytest tests/test_cli/test_wrap_helpers.py -v --no-header -x
python -m pytest tests/test_cli/test_wrap_persistent.py -v --no-header -x
python -m pytest tests/test_cli/test_wrap_codex.py -v --no-header -x

# Lint
ruff check headroom/cli/wrap.py

# 全量测试
python -m pytest tests/ -x --timeout=120
```
