"""Model registry and capabilities database.

Provides a centralized registry of LLM models with their capabilities,
context limits, pricing, and provider information.

Also provides MLModelRegistry for sharing ML model instances (sentence
transformers, SIGLIP, spaCy) to avoid loading the same model multiple times.

Usage:
    from headroom.models import ModelRegistry, get_model_info

    # Get info about a model
    info = get_model_info("gpt-4o")
    print(f"Context: {info.context_window}")
    print(f"Provider: {info.provider}")

    # List all models from a provider
    models = ModelRegistry.list_models(provider="openai")

    # Register a custom model
    ModelRegistry.register(
        "my-custom-model",
        provider="custom",
        context_window=32000,
    )

    # Get shared ML model instances
    from headroom.models import MLModelRegistry
    model = MLModelRegistry.get_sentence_transformer()
"""

from .ml_models import (
    MLModelRegistry,
    get_sentence_transformer,
    get_siglip,
    get_spacy,
)
from .registry import (
    ModelInfo,
    ModelRegistry,
    get_model_info,
    list_models,
    register_model,
)

__all__ = [
    # LLM Registry
    "ModelRegistry",
    "ModelInfo",
    "get_model_info",
    "list_models",
    "register_model",
    # ML Model Registry
    "MLModelRegistry",
    "get_sentence_transformer",
    "get_siglip",
    "get_spacy",
]
