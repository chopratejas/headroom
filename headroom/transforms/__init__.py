"""Transform modules for Headroom SDK."""

from .base import Transform
from .cache_aligner import CacheAligner
from .content_detector import ContentType, DetectionResult, detect_content_type
from .log_compressor import LogCompressor, LogCompressorConfig, LogCompressionResult
from .pipeline import TransformPipeline
from .rolling_window import RollingWindow
from .search_compressor import (
    SearchCompressor,
    SearchCompressorConfig,
    SearchCompressionResult,
)
from .smart_crusher import SmartCrusher, SmartCrusherConfig
from .text_compressor import TextCompressor, TextCompressorConfig, TextCompressionResult
from .tool_crusher import ToolCrusher

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
    # Other transforms
    "CacheAligner",
    "RollingWindow",
]
