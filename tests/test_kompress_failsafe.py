"""Fail-safe behavior for Kompress on degraded machines.

Reproduces the Windows incident where one pathologically slow ONNX inference
held the execution semaphore forever: every later compression blocked on an
unbounded acquire, every request hit the proxy's 30s stage timeout, and the
proxy delivered 0% savings plus +30s latency until restart. These tests pin
the three layers of defense (bounded acquire, wall-clock budget, preload
canary) and that the normal fast path is untouched.

No ML dependencies — the model/tokenizer are fakes injected via
``_load_kompress``.
"""

import time

import pytest

import headroom.transforms.kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    KOMPRESS_ACQUIRE_TIMEOUT_ENV,
    KOMPRESS_CANARY_THRESHOLD_ENV,
    KOMPRESS_TIME_BUDGET_ENV,
    KompressCompressor,
    KompressConfig,
)


class FakeEncoding:
    """Mimics a transformers BatchEncoding for is_split_into_words inputs.

    One token per word, no special tokens — word_ids(i) is identity.
    """

    def __init__(self, rows: list[list[str]]):
        self._rows = rows

    def __getitem__(self, key: str):
        if key == "input_ids":
            return [[0] * len(r) for r in self._rows]
        if key == "attention_mask":
            return [[1] * len(r) for r in self._rows]
        raise KeyError(key)

    def word_ids(self, batch_index: int = 0):
        return list(range(len(self._rows[batch_index])))


class FakeTokenizer:
    def __call__(self, words, **kwargs):
        # is_split_into_words inputs: either one word list or a batch of them.
        rows = words if words and isinstance(words[0], list) else [words]
        return FakeEncoding(rows)


class FakeModel:
    """Keeps every other word; optional per-call delay to simulate slowness."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def _tick(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)

    def get_keep_mask(self, input_ids, attention_mask):
        self._tick()
        return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        self._tick()
        return [[1.0 if i % 2 == 0 else 0.0 for i in range(len(row))] for row in input_ids]


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    kc._execution_semaphores.clear()
    monkeypatch.setattr(kc, "_giveup_warned", False)
    for env in (
        KOMPRESS_ACQUIRE_TIMEOUT_ENV,
        KOMPRESS_TIME_BUDGET_ENV,
        KOMPRESS_CANARY_THRESHOLD_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    yield
    kc._execution_semaphores.clear()


def _make_compressor(monkeypatch, model: FakeModel, **config_kwargs) -> KompressCompressor:
    config_kwargs.setdefault("enable_ccr", False)
    compressor = KompressCompressor(config=KompressConfig(**config_kwargs))
    monkeypatch.setattr(
        kc,
        "_load_kompress",
        lambda model_id, device="auto", **kwargs: (model, FakeTokenizer(), "onnx"),
    )
    return compressor


CONTENT_40_WORDS = " ".join(f"word{i}" for i in range(40))


# ── Normal path: behavior and performance must be unchanged ───────────


def test_fast_model_compresses_normally(monkeypatch):
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    result = compressor.compress(CONTENT_40_WORDS)

    assert result.compressed != CONTENT_40_WORDS
    assert result.compressed_tokens == 20  # every other word kept
    assert result.compression_ratio == 0.5
    assert model.calls == 1


def test_fast_model_releases_semaphore(monkeypatch):
    compressor = _make_compressor(monkeypatch, FakeModel())
    compressor.compress(CONTENT_40_WORDS)

    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    semaphore.release()


def test_semaphore_released_when_inference_raises(monkeypatch):
    class ExplodingModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            raise RuntimeError("boom")

    compressor = _make_compressor(monkeypatch, ExplodingModel())
    result = compressor.compress(CONTENT_40_WORDS)

    assert result.compressed == CONTENT_40_WORDS  # passthrough, not an exception
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    semaphore.release()


# ── Bounded acquire: a stuck inference must not wedge other requests ──


def test_stuck_semaphore_passes_through_instead_of_blocking(monkeypatch):
    monkeypatch.setenv(KOMPRESS_ACQUIRE_TIMEOUT_ENV, "0.1")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    # Simulate the Windows incident: another thread holds the semaphore
    # indefinitely (abandoned by its asyncio timeout but still running).
    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    try:
        started = time.monotonic()
        result = compressor.compress(CONTENT_40_WORDS)
        elapsed = time.monotonic() - started
    finally:
        stuck.release()

    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 0
    assert elapsed < 3.0  # used to block forever


def test_stuck_semaphore_batch_passes_through(monkeypatch):
    monkeypatch.setenv(KOMPRESS_ACQUIRE_TIMEOUT_ENV, "0.1")
    compressor = _make_compressor(monkeypatch, FakeModel())
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)

    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    try:
        contents = [CONTENT_40_WORDS, " ".join(f"x{i}" for i in range(30))]
        results = compressor.compress_batch(contents)
    finally:
        stuck.release()

    assert [r.compressed for r in results] == contents  # all passthrough, no data loss


def test_acquire_bounded_unbounded_when_both_disabled():
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert kc._acquire_bounded(semaphore, None, None) is True
    semaphore.release()


def test_acquire_bounded_negative_remaining_does_not_raise():
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    try:
        assert kc._acquire_bounded(semaphore, 5.0, -1.0) is False
    finally:
        semaphore.release()


# ── Wall-clock budget: give up before the proxy's stage timeout ───────


def test_time_budget_bails_to_passthrough(monkeypatch):
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0.2")
    model = FakeModel(delay=0.15)
    compressor = _make_compressor(monkeypatch, model, chunk_words=10)

    result = compressor.compress(CONTENT_40_WORDS)  # 4 chunks at ~0.15s each

    assert result.compressed == CONTENT_40_WORDS
    assert model.calls < 4  # bailed before processing every chunk


def test_time_budget_disabled_processes_all_chunks(monkeypatch):
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0")
    model = FakeModel(delay=0.01)
    compressor = _make_compressor(monkeypatch, model, chunk_words=10)

    result = compressor.compress(CONTENT_40_WORDS)

    assert model.calls == 4
    assert result.compression_ratio == 0.5


def test_time_budget_batch_keeps_completed_texts(monkeypatch):
    """Mid-queue bail: fully processed texts stay compressed; any text with
    an unprocessed chunk passes through whole (never partially dropped)."""
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0.2")
    model = FakeModel(delay=0.25)  # one batch alone exhausts the budget
    compressor = _make_compressor(monkeypatch, model)
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)

    contents = [
        " ".join(f"a{i}" for i in range(20)),
        " ".join(f"b{i}" for i in range(20)),
        " ".join(f"c{i}" for i in range(20)),
    ]
    results = compressor.compress_batch(contents, batch_size=1)

    assert len(results) == 3
    # First batch ran; later ones bailed to passthrough.
    assert results[0].compression_ratio == 0.5
    assert results[1].compressed == contents[1]
    assert results[2].compressed == contents[2]
    # Every result preserves all information (compressed or original).
    for r in results:
        assert r.compressed


# ── Preload canary: detect degraded runtimes before live traffic ──────


def _join_canary(compressor: KompressCompressor) -> None:
    assert compressor._canary_thread is not None
    compressor._canary_thread.join(timeout=10)
    assert not compressor._canary_thread.is_alive()


def test_canary_disables_kompress_on_slow_inference(monkeypatch, caplog):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.05")
    model = FakeModel(delay=0.15)
    compressor = _make_compressor(monkeypatch, model)

    with caplog.at_level("WARNING"):
        backend = compressor.preload()
        _join_canary(compressor)

    assert backend == "onnx"
    assert compressor._degraded_reason is not None
    assert model.calls == 2  # probe + one retry
    assert "DISABLED" in caplog.text

    result = compressor.compress(CONTENT_40_WORDS)
    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 2  # model never touched again

    batch = compressor.compress_batch([CONTENT_40_WORDS])
    assert batch[0].compressed == CONTENT_40_WORDS
    assert model.calls == 2


def test_canary_fast_inference_stays_enabled(monkeypatch):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "5")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    compressor.preload()
    _join_canary(compressor)

    assert compressor._degraded_reason is None
    result = compressor.compress(CONTENT_40_WORDS)
    assert result.compression_ratio == 0.5


def test_canary_retry_forgives_oneoff_warmup_slowness(monkeypatch):
    """First inference pays one-off warmup costs; only a slow retry condemns."""
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.1")

    class WarmupModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.2)  # cold first run
            return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    model = WarmupModel()
    compressor = _make_compressor(monkeypatch, model)
    compressor.preload()
    _join_canary(compressor)

    assert compressor._degraded_reason is None
    assert model.calls == 2


def test_canary_disabled_via_env(monkeypatch):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0")
    model = FakeModel(delay=0.2)
    compressor = _make_compressor(monkeypatch, model)

    compressor.preload()

    assert compressor._canary_thread is None  # probe never scheduled
    assert model.calls == 0
    assert compressor._degraded_reason is None


def test_preload_does_not_block_on_slow_canary(monkeypatch):
    """The probe runs off the startup path: preload blocks proxy boot (the
    HTTP server binds after it), and a slow probe once pushed the wrap-e2e
    container past its 30s health-check timeout."""
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.05")
    model = FakeModel(delay=1.0)
    compressor = _make_compressor(monkeypatch, model)

    started = time.monotonic()
    compressor.preload()
    preload_elapsed = time.monotonic() - started

    assert preload_elapsed < 0.5  # returns before the ~2s of probe inference
    _join_canary(compressor)
    assert compressor._degraded_reason is not None


def test_canary_probe_error_never_breaks_preload(monkeypatch):
    class ExplodingModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            raise RuntimeError("probe boom")

    compressor = _make_compressor(monkeypatch, ExplodingModel())

    assert compressor.preload() == "onnx"
    _join_canary(compressor)
    assert compressor._degraded_reason is None
