import sys

from headroom.onnx_runtime import (
    ONNX_CPU_ARENA_ENV,
    cpu_arena_enabled,
    create_cpu_session_options,
)


class _FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True


class _FakeOrt:
    SessionOptions = _FakeSessionOptions


class _FakeSessionOptionsWithoutToggles:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None


class _FakeOrtWithoutToggles:
    SessionOptions = _FakeSessionOptionsWithoutToggles


def test_create_cpu_session_options_disables_retention_features(monkeypatch):
    """Non-Windows keeps the legacy low-RSS behavior: arena + mem pattern off."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(
        _FakeOrt,
        intra_op_num_threads=1,
        inter_op_num_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_darwin_unchanged(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    options = create_cpu_session_options(_FakeOrt)

    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_keeps_arena_on_windows(monkeypatch):
    """Disabling the arena on Windows degrades inference by orders of
    magnitude (onnxruntime#11627) — ORT defaults must stay untouched there."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    options = create_cpu_session_options(_FakeOrt, intra_op_num_threads=3)

    assert options.enable_cpu_mem_arena is True
    assert options.enable_mem_pattern is True
    assert options.intra_op_num_threads == 3


def test_arena_env_override_forces_on(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "1")

    assert cpu_arena_enabled() is True
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is True


def test_arena_env_override_forces_off(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "0")

    assert cpu_arena_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is False


def test_arena_env_invalid_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "bananas")

    monkeypatch.setattr(sys, "platform", "win32")
    assert cpu_arena_enabled() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert cpu_arena_enabled() is False


def test_create_cpu_session_options_handles_older_session_options(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(_FakeOrtWithoutToggles)

    assert options.intra_op_num_threads is None
    assert options.inter_op_num_threads is None
