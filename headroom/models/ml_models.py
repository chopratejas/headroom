"""Centralized registry for ML model instances.

Provides shared access to ML models (sentence transformers, SIGLIP, spaCy, etc.)
to avoid loading the same model multiple times across different components.

This is different from registry.py which stores LLM metadata. This module
manages actual loaded model instances that consume memory.

Usage:
    from headroom.models.ml_models import MLModelRegistry

    # Get shared sentence transformer (loads on first access)
    model = MLModelRegistry.get_sentence_transformer()
    embeddings = model.encode(["hello", "world"])

    # Get SIGLIP for image embeddings
    siglip_model, processor = MLModelRegistry.get_siglip()

    # Check what's loaded
    print(MLModelRegistry.loaded_models())
    print(f"Memory: {MLModelRegistry.estimated_memory_mb():.1f} MB")
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Model size estimates in MB (approximate)
MODEL_SIZES_MB = {
    "sentence_transformer:all-MiniLM-L6-v2": 90,
    "sentence_transformer:all-mpnet-base-v2": 420,
    "siglip:google/siglip-base-patch16-224": 400,
    "siglip:google/siglip-large-patch16-384": 1200,
    "llmlingua:microsoft/llmlingua-2-xlm-roberta-large-meetingbank": 1000,
    "spacy:en_core_web_sm": 40,
    "spacy:en_core_web_md": 120,
    "technique_router:chopratejas/technique-router": 100,
}


class MLModelRegistry:
    """Singleton registry for shared ML model instances.

    Provides lazy-loaded, shared access to ML models across all components.
    This prevents the same model from being loaded multiple times.

    Thread-safe for concurrent access.
    """

    _instance: MLModelRegistry | None = None
    _lock = RLock()

    def __new__(cls) -> MLModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the registry."""
        self._models: dict[str, Any] = {}
        self._model_lock = RLock()

    @classmethod
    def get(cls) -> MLModelRegistry:
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._models.clear()
            cls._instance = None

    # =========================================================================
    # Sentence Transformers
    # =========================================================================

    @classmethod
    def get_sentence_transformer(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> Any:
        """Get a shared SentenceTransformer instance.

        Args:
            model_name: Model name (default: all-MiniLM-L6-v2).
            device: Device to use (cuda, mps, cpu). Auto-detected if None.

        Returns:
            SentenceTransformer model instance.
        """
        instance = cls.get()
        key = f"sentence_transformer:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading SentenceTransformer: {model_name}")
                from sentence_transformers import SentenceTransformer

                if device is None:
                    device = cls._detect_device()

                model = SentenceTransformer(model_name, device=device)
                instance._models[key] = model
                logger.info(f"Loaded SentenceTransformer: {model_name} on {device}")

            return instance._models[key]

    # =========================================================================
    # SIGLIP (Image Embeddings)
    # =========================================================================

    @classmethod
    def get_siglip(
        cls,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """Get shared SIGLIP model and processor.

        Args:
            model_name: Model name (default: google/siglip-base-patch16-224).
            device: Device to use. Auto-detected if None.

        Returns:
            Tuple of (model, processor).
        """
        instance = cls.get()
        key = f"siglip:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading SIGLIP: {model_name}")
                from transformers import AutoModel, AutoProcessor

                if device is None:
                    device = cls._detect_device()

                model = AutoModel.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)

                # Move to device and set eval mode
                if device != "cpu":
                    import torch

                    model = model.to(torch.device(device))
                model.eval()

                instance._models[key] = (model, processor)
                logger.info(f"Loaded SIGLIP: {model_name} on {device}")

            result: tuple[Any, Any] = instance._models[key]
            return result

    # =========================================================================
    # spaCy
    # =========================================================================

    @classmethod
    def get_spacy(cls, model_name: str = "en_core_web_sm") -> Any:
        """Get a shared spaCy model.

        Args:
            model_name: Model name (default: en_core_web_sm).

        Returns:
            spaCy Language model.
        """
        instance = cls.get()
        key = f"spacy:{model_name}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading spaCy: {model_name}")
                import spacy

                model = spacy.load(model_name)
                instance._models[key] = model
                logger.info(f"Loaded spaCy: {model_name}")

            return instance._models[key]

    # =========================================================================
    # Technique Router (Sequence Classification)
    # =========================================================================

    @classmethod
    def get_technique_router(
        cls,
        model_path: str | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """Get shared technique router model and tokenizer.

        Args:
            model_path: Path to model (default: chopratejas/technique-router).
            device: Device to use. Auto-detected if None.

        Returns:
            Tuple of (model, tokenizer).
        """
        from pathlib import Path

        instance = cls.get()

        # Default to HuggingFace model, check for local first
        if model_path is None:
            local_path = Path("headroom/models/technique-router-mini/final/")
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = "chopratejas/technique-router"

        key = f"technique_router:{model_path}"

        with instance._model_lock:
            if key not in instance._models:
                logger.info(f"Loading technique router: {model_path}")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                if device is None:
                    device = cls._detect_device()

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

                # Move to device and set eval mode
                if device != "cpu":
                    import torch

                    model = model.to(torch.device(device))
                model.eval()

                instance._models[key] = (model, tokenizer)
                logger.info(f"Loaded technique router: {model_path} on {device}")

            result: tuple[Any, Any] = instance._models[key]
            return result

    # =========================================================================
    # LLMLingua (uses existing singleton pattern)
    # =========================================================================

    @classmethod
    def get_llmlingua(cls, device: str | None = None, model_name: str | None = None) -> Any:
        """Get the LLMLingua compressor.

        Note: LLMLingua already has its own singleton in llmlingua_compressor.py.
        This method delegates to that implementation.

        Args:
            device: Device to use. Auto-detected if None.
            model_name: Model name (default: microsoft/llmlingua-2-xlm-roberta-large-meetingbank).

        Returns:
            PromptCompressor instance.
        """
        from headroom.transforms.llmlingua_compressor import _get_llmlingua_compressor

        if device is None:
            device = cls._detect_device()

        if model_name is None:
            model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

        return _get_llmlingua_compressor(model_name=model_name, device=device)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @classmethod
    def _detect_device(cls) -> str:
        """Auto-detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def loaded_models(cls) -> list[str]:
        """Get list of currently loaded model keys."""
        instance = cls.get()
        with instance._model_lock:
            return list(instance._models.keys())

    @classmethod
    def is_loaded(cls, key: str) -> bool:
        """Check if a model is loaded."""
        instance = cls.get()
        with instance._model_lock:
            return key in instance._models

    @classmethod
    def estimated_memory_mb(cls) -> float:
        """Estimate total memory used by loaded models."""
        instance = cls.get()
        total = 0.0
        with instance._model_lock:
            for key in instance._models:
                total += MODEL_SIZES_MB.get(key, 100)  # Default 100MB if unknown
        return total

    @classmethod
    def get_memory_stats(cls) -> dict[str, Any]:
        """Get memory statistics for all loaded models."""
        instance = cls.get()
        loaded_models: list[dict[str, Any]] = []
        total_estimated_mb: float = 0.0

        with instance._model_lock:
            for key in instance._models:
                size_mb = MODEL_SIZES_MB.get(key, 100)
                loaded_models.append({"key": key, "size_mb": size_mb})
                total_estimated_mb += size_mb

        return {
            "loaded_models": loaded_models,
            "total_estimated_mb": total_estimated_mb,
        }


# Convenience functions for direct access
def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
) -> Any:
    """Get a shared SentenceTransformer instance."""
    return MLModelRegistry.get_sentence_transformer(model_name, device)


def get_siglip(
    model_name: str = "google/siglip-base-patch16-224",
    device: str | None = None,
) -> tuple[Any, Any]:
    """Get shared SIGLIP model and processor."""
    return MLModelRegistry.get_siglip(model_name, device)


def get_spacy(model_name: str = "en_core_web_sm") -> Any:
    """Get a shared spaCy model."""
    return MLModelRegistry.get_spacy(model_name)
