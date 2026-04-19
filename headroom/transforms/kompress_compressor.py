"""Kompress: ModernBERT token compressor for structured tool outputs.

Auto-downloads the model from HuggingFace (chopratejas/kompress-base)
on first use.

Requires the [ml] extra: pip install headroom-ai[ml]

Usage:
    >>> from headroom.transforms.kompress_compressor import KompressCompressor
    >>> compressor = KompressCompressor()
    >>> result = compressor.compress(long_tool_output)
    >>> print(result.compressed)
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from dataclasses import dataclass
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform

logger = logging.getLogger(__name__)

# Default HuggingFace model ID
HF_MODEL_ID = "chopratejas/kompress-base"

# Model cache: model_id -> (model, tokenizer, backend)
# Supports multiple models loaded simultaneously.
_kompress_cache: dict[str, tuple[Any, Any, str]] = {}
_kompress_lock = threading.Lock()


def _is_onnx_available() -> bool:
    """Check if ONNX Runtime is available (lightweight, no torch needed)."""
    try:
        import onnxruntime  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _is_pytorch_available() -> bool:
    """Check if full PyTorch stack is available (requires [ml] extra)."""
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def is_kompress_available() -> bool:
    """Check if Kompress can run — ONNX (lightweight) or PyTorch (full)."""
    return _is_onnx_available() or _is_pytorch_available()


# ── Model Architecture (must match training) ──────────────────────────
# torch/transformers are imported lazily — only when actually needed.
# This allows `from kompress_compressor import is_kompress_available`
# to work without torch installed.


def _get_model_class() -> type:
    """Return the HeadroomCompressorModel class, importing torch on demand."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class HeadroomCompressorModel(nn.Module):
        """Dual-head ModernBERT: token classification + span importance CNN."""

        def __init__(self, model_name: str = "answerdotai/ModernBERT-base"):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
            hidden_size = self.encoder.config.hidden_size  # 768

            # Head 1: Token keep/discard
            self.token_dropout = nn.Dropout(0.1)
            self.token_head = nn.Linear(hidden_size, 2)

            # Head 2: Span importance (1D CNN)
            self.span_conv = nn.Sequential(
                nn.Conv1d(hidden_size, 256, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(256, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def get_keep_mask(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            """Get per-token keep/discard decision. True = keep."""
            with torch.no_grad():
                hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

                # Token head: binary classifier — argmax decides keep/discard
                token_logits = self.token_head(hidden)  # [B, L, 2]
                token_keep = (
                    token_logits[:, :, 1] > token_logits[:, :, 0]
                )  # True if class 1 > class 0

                # Span head: boost tokens in important spans
                # If a token is borderline but its span is important, keep it
                span_scores = self.span_conv(hidden.transpose(1, 2)).squeeze(1)
                span_boost = span_scores > 0.5  # span says this region matters

                # Keep if: token head says keep, OR token is borderline and span says keep
                token_probs = torch.softmax(token_logits, dim=-1)[:, :, 1]
                borderline = (token_probs > 0.3) & (token_probs <= 0.5)
                keep = token_keep | (borderline & span_boost)

                return keep  # type: ignore[no-any-return]

        def get_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Get per-token importance scores (for ranking when target_ratio is set)."""
            with torch.no_grad():
                hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
                token_probs = torch.softmax(self.token_head(hidden), dim=-1)[:, :, 1]
                span_scores = self.span_conv(hidden.transpose(1, 2)).squeeze(1)
                return token_probs * (0.5 + 0.5 * span_scores)  # type: ignore[no-any-return]

    return HeadroomCompressorModel


# ── Model Loading ─────────────────────────────────────────────────────


class _OnnxModel:
    """Thin wrapper so ONNX session has the same interface as PyTorch model."""

    def __init__(self, session: Any):
        self._session = session

    def get_scores(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return [batch, seq] scores via ONNX Runtime."""
        import numpy as np

        scores = self._session.run(
            ["final_scores"],
            {
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            },
        )
        return scores[0]  # [batch, seq] numpy array

    def get_keep_mask(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return [batch, seq] boolean mask (score > 0.5)."""
        import numpy as np

        scores = self.get_scores(input_ids, attention_mask)
        return (np.array(scores) > 0.5).tolist()


def _load_kompress_onnx(model_id: str) -> tuple[Any, Any, str]:
    """Download ONNX INT8 model from HuggingFace and load with onnxruntime."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    with _kompress_lock:
        if model_id in _kompress_cache:
            return _kompress_cache[model_id]

        from huggingface_hub import hf_hub_download

        logger.info("Downloading Kompress ONNX model from %s ...", model_id)
        onnx_path = hf_hub_download(model_id, "onnx/kompress-int8.onnx")

        session = ort.InferenceSession(onnx_path)
        model = _OnnxModel(session)
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        _kompress_cache[model_id] = (model, tokenizer, "onnx")
        logger.info("Kompress ONNX INT8 loaded: %s", model_id)
        return model, tokenizer, "onnx"


def _load_kompress_pytorch(model_id: str, device: str = "auto") -> tuple[Any, Any, str]:
    """Download PyTorch model from HuggingFace and load with torch."""
    import torch
    from transformers import AutoTokenizer

    with _kompress_lock:
        if model_id in _kompress_cache:
            return _kompress_cache[model_id]

        from huggingface_hub import hf_hub_download

        logger.info("Downloading Kompress PyTorch model from %s ...", model_id)
        weights_path = hf_hub_download(model_id, "model.safetensors")

        HeadroomCompressorModel = _get_model_class()
        model = HeadroomCompressorModel()

        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        _kompress_cache[model_id] = (model, tokenizer, "pytorch")
        logger.info("Kompress PyTorch loaded on %s (%s)", device, model_id)
        return model, tokenizer, "pytorch"


def _load_kompress(model_id: str = HF_MODEL_ID, device: str = "auto") -> tuple[Any, Any, str]:
    """Load Kompress model, returns (model, tokenizer, backend).

    Selection order:

    1. ``HEADROOM_KOMPRESS_BACKEND=onnx|pytorch`` — explicit override.
    2. ``auto`` (default): prefer PyTorch when CUDA or Apple-Silicon MPS is
       available, else ONNX.
    3. Falls back to whichever of ONNX / PyTorch imports successfully.

    Models are cached by ``model_id`` — multiple models can coexist.
    """
    if model_id in _kompress_cache:
        return _kompress_cache[model_id]

    backend = os.environ.get("HEADROOM_KOMPRESS_BACKEND", "auto").lower()

    # Explicit override — no fallback, let the error surface.
    if backend == "onnx":
        return _load_kompress_onnx(model_id)
    if backend == "pytorch":
        return _load_kompress_pytorch(model_id, device)

    if backend != "auto":
        logger.warning(
            "Unknown HEADROOM_KOMPRESS_BACKEND=%r; falling back to auto-detect. "
            "Valid values: auto, onnx, pytorch.",
            backend,
        )

    # Auto: prefer PyTorch only when an accelerator is actually usable.
    prefer_pytorch = False
    if _is_pytorch_available():
        try:
            import torch

            is_apple_silicon = (
                platform.machine() == "arm64" and platform.system() == "Darwin"
            )
            prefer_pytorch = bool(
                torch.cuda.is_available()
                or (is_apple_silicon and torch.backends.mps.is_available())
            )
        except Exception:  # noqa: BLE001 — torch probe must never break loading
            prefer_pytorch = False

    if prefer_pytorch:
        try:
            return _load_kompress_pytorch(model_id, device)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Kompress PyTorch load failed, falling back to ONNX: %s", e
            )

    # Default / fallback: ONNX first (lightweight), then PyTorch.
    if _is_onnx_available():
        try:
            return _load_kompress_onnx(model_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Kompress ONNX load failed for %s, trying PyTorch: %s", model_id, e)

    if _is_pytorch_available():
        return _load_kompress_pytorch(model_id, device)

    raise ImportError(
        "Kompress requires onnxruntime or torch. Install with: pip install headroom-ai[proxy]"
    )


def unload_kompress_model(model_id: str | None = None) -> bool:
    """Unload Kompress model(s) to free memory.

    Args:
        model_id: Specific model to unload. If None, unloads all cached models.
    """
    with _kompress_lock:
        if model_id is not None:
            if model_id in _kompress_cache:
                del _kompress_cache[model_id]
            else:
                return False
        elif _kompress_cache:
            _kompress_cache.clear()
        else:
            return False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return True


# ── Compressor ────────────────────────────────────────────────────────


@dataclass
class KompressConfig:
    """Configuration for Kompress compression.

    The model_id, chunk_words, and score_threshold are coupled: a model
    trained on 50-word chunks needs chunk_words=50 at inference. The
    defaults match kompress-base. For domain-specific models, set all three.

    Example — financial documents::

        KompressConfig(
            model_id="chopratejas/kompress-finance",
            chunk_words=50,
            score_threshold=0.5,
        )
    """

    device: str = "auto"
    enable_ccr: bool = True
    model_id: str = HF_MODEL_ID
    chunk_words: int = 350
    score_threshold: float = 0.5


@dataclass
class KompressResult:
    """Result of Kompress compression."""

    compressed: str
    original: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cache_key: str | None = None
    model_used: str = HF_MODEL_ID

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class KompressCompressor(Transform):
    """Kompress: ModernBERT token compressor.

    Auto-downloads the model from HuggingFace on first use.
    Configure via KompressConfig to select model, chunk size, and threshold.
    """

    name: str = "kompress_compressor"

    def __init__(self, config: KompressConfig | None = None):
        self.config = config or KompressConfig()

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | None = None,
    ) -> KompressResult:
        """Compress content using Kompress model.

        Args:
            content: Text to compress.
            context: Optional surrounding context (unused by model).
            content_type: Ignored — model decides importance per content type.
            question: Ignored — reserved for future QA-aware compression.
            target_ratio: If None (default), model decides how much to keep using
                score threshold. If set (e.g. 0.3), forces that keep ratio.
                The proxy never sets this — only user-facing API does.

        Returns:
            KompressResult with compressed text.
        """
        words = content.split()
        n_words = len(words)

        if n_words < 10:
            return self._passthrough(content, n_words)

        try:
            model, tokenizer, backend = _load_kompress(self.config.model_id, self.config.device)
            is_onnx = backend == "onnx"

            max_chunk_words = self.config.chunk_words
            kept_ids: set[int] = set()

            for chunk_start in range(0, n_words, max_chunk_words):
                chunk_words = words[chunk_start : chunk_start + max_chunk_words]

                # ONNX uses numpy tensors, PyTorch uses torch tensors
                return_tensors = "np" if is_onnx else "pt"
                encoding = tokenizer(
                    chunk_words,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors=return_tensors,
                )

                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                word_ids = encoding.word_ids(batch_index=0)

                if not is_onnx:
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                if target_ratio is not None:
                    scores = model.get_scores(input_ids, attention_mask)
                    if is_onnx:
                        score_list = scores[0]  # numpy: [seq_len]
                    else:
                        score_list = scores[0].cpu()
                    word_scores: dict[int, float] = {}
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        s = float(score_list[idx])
                        if wid not in word_scores or s > word_scores[wid]:
                            word_scores[wid] = s
                    if word_scores:
                        sorted_wids = sorted(
                            word_scores, key=lambda w: word_scores[w], reverse=True
                        )
                        num_keep = max(1, int(len(sorted_wids) * target_ratio))
                        for wid in sorted_wids[:num_keep]:
                            kept_ids.add(wid + chunk_start)
                else:
                    keep_mask = model.get_keep_mask(input_ids, attention_mask)
                    if is_onnx:
                        mask_list = keep_mask[0]  # list of bools
                    else:
                        mask_list = keep_mask[0].cpu()
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        if bool(mask_list[idx]):
                            kept_ids.add(wid + chunk_start)

            if not kept_ids:
                return self._passthrough(content, n_words)

            compressed_words = [words[w] for w in sorted(kept_ids) if w < n_words]
            compressed = " ".join(compressed_words)
            compressed_count = len(compressed_words)
            ratio = compressed_count / n_words if n_words else 1.0

            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=n_words,
                compressed_tokens=compressed_count,
                compression_ratio=ratio,
                model_used=self.config.model_id,
            )

            # CCR marker
            if self.config.enable_ccr and ratio < 0.8:
                cache_key = self._store_in_ccr(content, compressed, n_words)
                if cache_key:
                    result.cache_key = cache_key
                    result.compressed += (
                        f"\n[{n_words} items compressed to {compressed_count}."
                        f" Retrieve more: hash={cache_key}]"
                    )

            return result

        except Exception as e:
            logger.warning("Kompress compression failed: %s", e)
            return self._passthrough(content, n_words)

    def compress_batch(
        self,
        contents: list[str],
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | list[float | None] | None = None,
        batch_size: int = 32,
    ) -> list[KompressResult]:
        """Compress multiple texts. Uses batched inference on GPU, sequential on CPU.

        On GPU (PyTorch + CUDA / MPS), runs a single batched forward pass per
        chunk batch, amortizing model inference across N texts. On CPU (ONNX
        or PyTorch), falls back to sequential ``compress()`` calls because
        ONNX Runtime's CPU provider does not parallelize across the batch
        dimension for this model (empirically 0.7-0.9x vs sequential).

        The fallback is transparent: callers get the best available
        performance per device without needing to detect the backend
        themselves.

        Measured performance (RTX 3080 Ti, ~350-word inputs):

            GPU batched vs sequential:
                N=3:  1.76x speedup
                N=5:  2.08x speedup
                N=12: 2.18x speedup
                N=24: 2.34x speedup

            CPU (ONNX, 16 logical threads): falls back to sequential;
                net effect is parity with direct ``compress()`` in a loop.

        Args:
            contents: List of texts to compress. May contain short texts or
                empty strings — those pass through without a model call.
            context: Unused (parity with ``compress``).
            content_type: Unused (parity with ``compress``).
            question: Unused (parity with ``compress``).
            target_ratio: Compression target, one of:

                * ``None`` — model decides per text (same as :meth:`compress`).
                * ``float`` — applied uniformly to every text in the batch.
                * ``list`` of ``float | None`` — per-text ratio; must match
                  ``len(contents)``. ``None`` entries let the model decide for
                  that text.

            batch_size: Maximum number of chunks per forward pass on the
                batched path (GPU only — ignored on CPU fallback). Default
                ``32`` is a reasonable balance for ModernBERT on GPU.

        Returns:
            List of :class:`KompressResult`, one per input text, in input order.
            Empty input returns empty list. Failed texts fall back to
            passthrough rather than raising.

        Notes:
            On the batched GPU path, scoring uses ``get_scores`` uniformly
            (threshold at 0.5 when ``target_ratio`` is ``None``). This
            matches the ONNX non-batched behavior exactly. The PyTorch
            non-batched path applies an additional borderline + span-boost
            rule, so results may differ by a small fraction of tokens on
            ``target_ratio=None`` calls via the batched path vs direct
            :meth:`compress` on PyTorch. Call :meth:`compress` directly if
            the exact PyTorch borderline behavior is required.
        """
        n = len(contents)
        if n == 0:
            return []

        # Normalize target_ratio to a per-text list
        if isinstance(target_ratio, list):
            if len(target_ratio) != n:
                raise ValueError(
                    f"target_ratio list length {len(target_ratio)} does not match "
                    f"contents length {n}"
                )
            ratios: list[float | None] = list(target_ratio)
        else:
            ratios = [target_ratio] * n

        # Fast path: on backends where batch-dim parallelism does NOT help
        # (ONNX CPU, PyTorch CPU), fall back to sequential `compress()`
        # internally. This keeps the public API consistent while avoiding the
        # per-item slowdown measured on ONNX CPU (~0.7-0.9x vs sequential).
        # GPU users still benefit from the batched forward pass below.
        if self._should_use_sequential_fallback():
            return [
                self.compress(
                    content,
                    context=context,
                    content_type=content_type,
                    question=question,
                    target_ratio=r,
                )
                for content, r in zip(contents, ratios, strict=True)
            ]

        results: list[KompressResult | None] = [None] * n
        word_lists: list[list[str]] = [c.split() for c in contents]

        # Short texts short-circuit to passthrough — no model call needed.
        max_chunk_words = self.config.chunk_words
        chunk_queue: list[tuple[int, int, list[str], float | None]] = []
        for i, (words, ratio) in enumerate(zip(word_lists, ratios, strict=True)):
            if len(words) < 10:
                results[i] = self._passthrough(contents[i], len(words))
                continue
            for chunk_start in range(0, len(words), max_chunk_words):
                chunk_words = words[chunk_start : chunk_start + max_chunk_words]
                chunk_queue.append((i, chunk_start, chunk_words, ratio))

        if not chunk_queue:
            # Every input was short — all passthrough, no model needed.
            return [r for r in results if r is not None]

        # Load model once for the whole batch.
        try:
            model, tokenizer, backend = _load_kompress(self.config.model_id, self.config.device)
        except Exception as e:
            logger.warning("Kompress load failed for batch: %s — passthrough all", e)
            for i in range(n):
                if results[i] is None:
                    results[i] = self._passthrough(contents[i], len(word_lists[i]))
            return [r for r in results if r is not None]

        is_onnx = backend == "onnx"
        kept_ids_per_text: dict[int, set[int]] = {i: set() for i in range(n) if results[i] is None}

        for batch_start in range(0, len(chunk_queue), batch_size):
            batch = chunk_queue[batch_start : batch_start + batch_size]
            batch_word_lists = [c[2] for c in batch]

            try:
                return_tensors = "np" if is_onnx else "pt"
                encoding = tokenizer(
                    batch_word_lists,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors=return_tensors,
                )

                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]

                if not is_onnx:
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                # Single forward pass for all chunks in this batch.
                scores = model.get_scores(input_ids, attention_mask)

                for batch_idx, (text_idx, chunk_start, _chunk_words, ratio) in enumerate(batch):
                    word_ids = encoding.word_ids(batch_index=batch_idx)
                    score_list = scores[batch_idx] if is_onnx else scores[batch_idx].cpu()

                    # Token -> word reduction (max score per word).
                    word_scores: dict[int, float] = {}
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        s = float(score_list[idx])
                        if wid not in word_scores or s > word_scores[wid]:
                            word_scores[wid] = s

                    if not word_scores:
                        continue

                    if ratio is not None:
                        # Top-k by score.
                        sorted_wids = sorted(
                            word_scores, key=lambda w: word_scores[w], reverse=True
                        )
                        num_keep = max(1, int(len(sorted_wids) * ratio))
                        for wid in sorted_wids[:num_keep]:
                            kept_ids_per_text[text_idx].add(wid + chunk_start)
                    else:
                        # Threshold from config (default 0.5, matches ONNX get_keep_mask).
                        for wid, score in word_scores.items():
                            if score > self.config.score_threshold:
                                kept_ids_per_text[text_idx].add(wid + chunk_start)

            except Exception as e:
                logger.warning(
                    "Kompress batch forward pass failed: %s — passthrough affected texts", e
                )
                for text_idx, _, _, _ in batch:
                    if results[text_idx] is None:
                        results[text_idx] = self._passthrough(
                            contents[text_idx], len(word_lists[text_idx])
                        )
                        kept_ids_per_text.pop(text_idx, None)

        # Reconstruct compressed text for each non-passthrough result.
        for text_idx, kept_ids in kept_ids_per_text.items():
            if results[text_idx] is not None:
                continue
            content = contents[text_idx]
            words = word_lists[text_idx]
            n_words = len(words)

            if not kept_ids:
                results[text_idx] = self._passthrough(content, n_words)
                continue

            compressed_words = [words[w] for w in sorted(kept_ids) if w < n_words]
            compressed = " ".join(compressed_words)
            compressed_count = len(compressed_words)
            comp_ratio = compressed_count / n_words if n_words else 1.0

            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=n_words,
                compressed_tokens=compressed_count,
                compression_ratio=comp_ratio,
                model_used=self.config.model_id,
            )

            if self.config.enable_ccr and comp_ratio < 0.8:
                cache_key = self._store_in_ccr(content, compressed, n_words)
                if cache_key:
                    result.cache_key = cache_key
                    result.compressed += (
                        f"\n[{n_words} items compressed to {compressed_count}."
                        f" Retrieve more: hash={cache_key}]"
                    )

            results[text_idx] = result

        # Safety: every slot must be populated.
        final: list[KompressResult] = []
        for i, r in enumerate(results):
            if r is None:
                final.append(self._passthrough(contents[i], len(word_lists[i])))
            else:
                final.append(r)
        return final

    def _should_use_sequential_fallback(self) -> bool:
        """Return True if batched inference wouldn't speed up on this backend.

        Empirically measured:
          - ONNX CPU: no batch-dim parallelism; batched is 0.7-0.9x vs sequential.
          - PyTorch CPU: typically similar (conservative fallback).
          - PyTorch + CUDA: 2.0-2.3x speedup at N>=3 — use batched path.

        If the model isn't loaded yet, we trigger loading so the backend
        is known. This is a no-op if the model is already in cache.
        """
        model_id = self.config.model_id
        if model_id not in _kompress_cache:
            try:
                _load_kompress(model_id, self.config.device)
            except Exception:
                return True

        if model_id not in _kompress_cache:
            return True

        model, _tokenizer, backend = _kompress_cache[model_id]

        if backend == "onnx":
            return True  # ONNX CPU provider doesn't parallelize batch dim
        if backend == "pytorch":
            try:
                import torch

                if hasattr(model, "parameters"):
                    device = next(model.parameters()).device
                    if device.type in ("cuda", "mps"):
                        return False  # GPU/MPS benefits from batching
                _ = torch
            except ImportError:
                return True
        return True  # Conservative default: sequential

    def _passthrough(self, content: str, n_words: int) -> KompressResult:
        return KompressResult(
            compressed=content,
            original=content,
            original_tokens=n_words,
            compressed_tokens=n_words,
            compression_ratio=1.0,
        )

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply Kompress compression to messages (Transform interface)."""
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        transformed = []
        transforms_applied = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if not isinstance(content, str) or len(content.split()) < 10:
                transformed.append(message)
                continue

            # Compress tool outputs and long assistant messages
            # Model decides how much — no hardcoded ratios
            if role in ("tool", "assistant"):
                result = self.compress(content)
                if result.compression_ratio < 0.9:
                    transformed.append({**message, "content": result.compressed})
                    transforms_applied.append(f"kompress:{role}:{result.compression_ratio:.2f}")
                else:
                    transformed.append(message)
            else:
                transformed.append(message)

        tokens_after = sum(tokenizer.count_text(str(m.get("content", ""))) for m in transformed)

        return TransformResult(
            messages=transformed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied or ["kompress:noop"],
        )

    def _store_in_ccr(self, original: str, compressed: str, original_tokens: int) -> str | None:
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
            return store.store(
                original,
                compressed,
                original_tokens=original_tokens,
                compressed_tokens=len(compressed.split()),
                compression_strategy="kompress",
            )
        except Exception:
            return None
