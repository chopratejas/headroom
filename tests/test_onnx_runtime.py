import pytest

from headroom.onnx_runtime import create_cpu_session_options

_THREAD_ENV_VARS = (
    "HEADROOM_ONNX_INTRA_OP_THREADS",
    "HEADROOM_ONNX_INTER_OP_THREADS",
)


@pytest.fixture(autouse=True)
def _clear_thread_env(monkeypatch):
    """Keep thread-count tests independent of the ambient environment."""
    for name in _THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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


def test_create_cpu_session_options_disables_retention_features():
    options = create_cpu_session_options(
        _FakeOrt,
        intra_op_num_threads=1,
        inter_op_num_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_handles_older_session_options():
    options = create_cpu_session_options(_FakeOrtWithoutToggles)

    # Older SessionOptions without the toggle attributes must not raise.
    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0
    assert options.inter_op_num_threads == 1


def test_create_cpu_session_options_defaults_to_positive_cpuset():
    """With no explicit values and no env override, intra is sized to the cpuset."""
    options = create_cpu_session_options(_FakeOrt)

    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0
    assert options.inter_op_num_threads == 1


def test_create_cpu_session_options_honors_thread_env_overrides(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "3")
    monkeypatch.setenv("HEADROOM_ONNX_INTER_OP_THREADS", "2")

    options = create_cpu_session_options(_FakeOrt)

    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 2


def test_create_cpu_session_options_ignores_invalid_thread_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "not-a-number")

    options = create_cpu_session_options(_FakeOrt)

    # Invalid override falls back to the cpuset-sized default.
    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0


def test_create_cpu_session_options_explicit_values_ignore_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "9")

    options = create_cpu_session_options(_FakeOrt, intra_op_num_threads=1)

    # Explicit caller value wins over the env override.
    assert options.intra_op_num_threads == 1
