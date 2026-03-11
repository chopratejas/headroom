"""Kompress: ModernBERT token compressor for structured tool outputs.

Drop-in replacement for LLMLingua-2. Auto-downloads the model from
HuggingFace (chopratejas/kompress-base) on first use.

No extra pip install needed — uses transformers + safetensors
which are already Headroom dependencies.

Usage:
    >>> from headroom.transforms.kompress_compressor import KompressCompressor
    >>> compressor = KompressCompressor()
    >>> result = compressor.compress(long_tool_output)
    >>> print(result.compressed)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform

logger = logging.getLogger(__name__)

# HuggingFace model ID
HF_MODEL_ID = "chopratejas/kompress-base"

# Lazy singleton
_kompress_model = None
_kompress_tokenizer = None
_kompress_lock = threading.Lock()


# ── Model Architecture (must match training) ──────────────────────────


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

    def get_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get per-token compression scores. Higher = more important."""
        with torch.no_grad():
            hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

            token_probs = torch.softmax(self.token_head(hidden), dim=-1)[:, :, 1]

            span_scores = self.span_conv(hidden.transpose(1, 2)).squeeze(1)

            return token_probs * (0.5 + 0.5 * span_scores)


# ── Model Loading ─────────────────────────────────────────────────────


def _load_kompress(device: str = "auto") -> tuple[HeadroomCompressorModel, Any]:
    """Download from HuggingFace and load the Kompress model."""
    global _kompress_model, _kompress_tokenizer

    with _kompress_lock:
        if _kompress_model is not None:
            return _kompress_model, _kompress_tokenizer

        from huggingface_hub import hf_hub_download

        logger.info("Downloading Kompress model from %s ...", HF_MODEL_ID)

        # Download model weights
        weights_path = hf_hub_download(HF_MODEL_ID, "model.safetensors")

        # Load architecture
        model = HeadroomCompressorModel()

        # Load trained weights
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        model.to(device)
        model.eval()
        logger.info("Kompress model loaded on %s (%s)", device, HF_MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

        _kompress_model = model
        _kompress_tokenizer = tokenizer
        return model, tokenizer


def is_kompress_available() -> bool:
    """Check if Kompress dependencies are available."""
    try:
        import huggingface_hub  # noqa: F401
        import safetensors  # noqa: F401

        return True
    except ImportError:
        return False


def unload_kompress_model() -> bool:
    """Unload the Kompress model to free memory."""
    global _kompress_model, _kompress_tokenizer
    with _kompress_lock:
        if _kompress_model is not None:
            _kompress_model = None
            _kompress_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
    return False


# ── Compressor ────────────────────────────────────────────────────────


@dataclass
class KompressConfig:
    """Minimal config. The model decides what's important — not us."""

    device: str = "auto"
    enable_ccr: bool = True


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
    """Kompress: ModernBERT token compressor for structured tool outputs.

    Auto-downloads chopratejas/kompress-base from HuggingFace on first use.
    Drop-in replacement for LLMLinguaCompressor with identical interface.
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
            context: Optional surrounding context (unused by model, kept for interface compat).
            content_type: Ignored — model decides importance per content type.
            question: Ignored — kept for LLMLingua interface compat.
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
            model, tokenizer = _load_kompress(self.config.device)

            # Tokenize
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=8192,
                padding=True,
                return_tensors="pt",
            )

            device = next(model.parameters()).device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Get per-token importance scores from dual-head model
            scores = model.get_scores(input_ids, attention_mask)[0].cpu()

            # Map subword scores to word-level (max pooling)
            word_ids = encoding.word_ids(batch_index=0)
            word_scores: dict[int, float] = {}
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                s = scores[idx].item()
                if wid not in word_scores or s > word_scores[wid]:
                    word_scores[wid] = s

            if not word_scores:
                return self._passthrough(content, n_words)

            # Token selection
            if target_ratio is not None:
                # User explicitly asked for a specific ratio — use top-k
                sorted_wids = sorted(word_scores, key=lambda w: word_scores[w], reverse=True)
                num_keep = max(1, int(len(sorted_wids) * target_ratio))
                kept_ids = set(sorted_wids[:num_keep])
            else:
                # Model decides. Trained with binary labels — score > 0.5 = keep.
                # Dense content → most tokens score high → keeps more.
                # Boilerplate → most score low → keeps less. That's correct.
                kept_ids = {wid for wid, score in word_scores.items() if score > 0.5}
                if not kept_ids:
                    # Edge case: nothing above threshold — keep the single highest
                    best = max(word_scores, key=lambda w: word_scores[w])
                    kept_ids = {best}

            # Reconstruct in original word order
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
