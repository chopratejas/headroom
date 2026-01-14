"""Transform modules for Headroom SDK."""

from .base import Transform
from .cache_aligner import CacheAligner
from .content_detector import ContentType, DetectionResult, detect_content_type
from .log_compressor import LogCompressionResult, LogCompressor, LogCompressorConfig
from .pipeline import TransformPipeline
from .rolling_window import RollingWindow
from .search_compressor import (
    SearchCompressionResult,
    SearchCompressor,
    SearchCompressorConfig,
)
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .text_compressor import TextCompressionResult, TextCompressor, TextCompressorConfig
from .tool_crusher import ToolCrusher

# ML-based compression (optional dependency)
try:
    from .llmlingua_compressor import (  # noqa: F401
        LLMLinguaCompressor,
        LLMLinguaConfig,
        LLMLinguaResult,
        compress_with_llmlingua,
        is_llmlingua_model_loaded,
        unload_llmlingua_model,
    )

    _LLMLINGUA_AVAILABLE = True
except ImportError:
    _LLMLINGUA_AVAILABLE = False

# AST-based code compression (optional dependency)
from .code_compressor import (
    CodeAwareCompressor,
    CodeCompressionResult,
    CodeCompressorConfig,
    CodeLanguage,
    DocstringMode,
    detect_language,
    is_tree_sitter_available,
)

# Content routing (always available, lazy-loads compressors)
from .content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    generate_source_hint,
)

__all__ = [
    # Base
    "Transform",
    "TransformPipeline",
    # JSON compression
    "ToolCrusher",
    "SmartCrusher",
    "SmartCrusherConfig",
    # Text compression (coding tasks)
    "ContentType",
    "DetectionResult",
    "detect_content_type",
    "SearchCompressor",
    "SearchCompressorConfig",
    "SearchCompressionResult",
    "LogCompressor",
    "LogCompressorConfig",
    "LogCompressionResult",
    "TextCompressor",
    "TextCompressorConfig",
    "TextCompressionResult",
    # Code-aware compression (AST-based)
    "CodeAwareCompressor",
    "CodeCompressorConfig",
    "CodeCompressionResult",
    "CodeLanguage",
    "DocstringMode",
    "detect_language",
    "is_tree_sitter_available",
    # Content routing
    "ContentRouter",
    "ContentRouterConfig",
    "RouterCompressionResult",
    "CompressionStrategy",
    "generate_source_hint",
    # Other transforms
    "CacheAligner",
    "RollingWindow",
    # ML-based compression (optional)
    "_LLMLINGUA_AVAILABLE",
]

# Conditionally add LLMLingua exports
if _LLMLINGUA_AVAILABLE:
    __all__.extend(
        [
            "LLMLinguaCompressor",
            "LLMLinguaConfig",
            "LLMLinguaResult",
            "compress_with_llmlingua",
            "is_llmlingua_model_loaded",
            "unload_llmlingua_model",
        ]
    )
