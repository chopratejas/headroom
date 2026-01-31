"""Transform modules for Headroom SDK."""

from .anchor_selector import (
    AnchorSelector,
    AnchorStrategy,
    AnchorWeights,
    DataPattern,
    calculate_information_score,
    compute_item_hash,
)
from .base import Transform
from .cache_aligner import CacheAligner
from .content_detector import ContentType, DetectionResult, detect_content_type
from .diff_compressor import DiffCompressionResult, DiffCompressor, DiffCompressorConfig
from .intelligent_context import ContextStrategy, IntelligentContextManager
from .log_compressor import LogCompressionResult, LogCompressor, LogCompressorConfig
from .pipeline import TransformPipeline
from .rolling_window import RollingWindow
from .scoring import EmbeddingProvider, MessageScore, MessageScorer
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

# HTML content extraction (optional dependency - requires trafilatura)
try:
    from .html_extractor import (  # noqa: F401
        HTMLExtractionResult,
        HTMLExtractor,
        HTMLExtractorConfig,
        is_html_content,
    )

    _HTML_EXTRACTOR_AVAILABLE = True
except ImportError:
    _HTML_EXTRACTOR_AVAILABLE = False

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
)

__all__ = [
    # Base
    "Transform",
    "TransformPipeline",
    # Anchor selection
    "AnchorSelector",
    "AnchorStrategy",
    "AnchorWeights",
    "DataPattern",
    "calculate_information_score",
    "compute_item_hash",
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
    "DiffCompressor",
    "DiffCompressorConfig",
    "DiffCompressionResult",
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
    # Other transforms
    "CacheAligner",
    "RollingWindow",
    # Intelligent context management
    "IntelligentContextManager",
    "ContextStrategy",
    "MessageScorer",
    "MessageScore",
    "EmbeddingProvider",
    # ML-based compression (optional)
    "_LLMLINGUA_AVAILABLE",
    # HTML extraction (optional)
    "_HTML_EXTRACTOR_AVAILABLE",
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

# Conditionally add HTML extractor exports
if _HTML_EXTRACTOR_AVAILABLE:
    __all__.extend(
        [
            "HTMLExtractor",
            "HTMLExtractorConfig",
            "HTMLExtractionResult",
            "is_html_content",
        ]
    )
