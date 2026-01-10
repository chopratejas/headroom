"""Telemetry module for building the data flywheel.

This module collects PRIVACY-PRESERVING statistics about compression patterns
to enable cross-user learning and improve compression over time.

What we collect (anonymized):
- Tool output structure patterns (field types, not values)
- Compression decisions and ratios
- Retrieval patterns (rate, type, not content)
- Strategy effectiveness

What we DON'T collect:
- Actual data values
- User identifiers
- Queries or search terms
- File paths or tool names (unless opted in)

Usage:
    from headroom.telemetry import get_telemetry_collector

    collector = get_telemetry_collector()

    # Record a compression event
    collector.record_compression(
        tool_signature="search_api:v1",
        original_items=1000,
        compressed_items=20,
        strategy="top_n",
        field_stats={...},
    )

    # Export for aggregation
    stats = collector.export_stats()

TOIN (Tool Output Intelligence Network):
    from headroom.telemetry import get_toin

    toin = get_toin()

    # Get compression hints before compressing
    hint = toin.get_recommendation(tool_signature, query_context)

    # Record compression outcome
    toin.record_compression(tool_signature, ...)

    # Record retrieval (automatic via compression_store)
    toin.record_retrieval(sig_hash, retrieval_type, query, query_fields)
"""

from .collector import (
    TelemetryCollector,
    TelemetryConfig,
    get_telemetry_collector,
    reset_telemetry_collector,
)
from .models import (
    AnonymizedToolStats,
    CompressionEvent,
    FieldDistribution,
    RetrievalStats,
    ToolSignature,
)
from .toin import (
    CompressionHint,
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolPattern,
    get_toin,
    reset_toin,
)

__all__ = [
    # Collector
    "TelemetryCollector",
    "TelemetryConfig",
    "get_telemetry_collector",
    "reset_telemetry_collector",
    # Models
    "AnonymizedToolStats",
    "CompressionEvent",
    "FieldDistribution",
    "RetrievalStats",
    "ToolSignature",
    # TOIN
    "CompressionHint",
    "TOINConfig",
    "ToolIntelligenceNetwork",
    "ToolPattern",
    "get_toin",
    "reset_toin",
]
