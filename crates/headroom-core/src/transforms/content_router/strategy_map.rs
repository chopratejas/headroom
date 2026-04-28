//! Bidirectional [`ContentType`] ↔ [`CompressionStrategy`] mapping.
//!
//! Direct port of `_strategy_from_detection_type` and
//! `_content_type_from_strategy` from
//! `headroom/transforms/content_router.py`. The two helpers are only
//! quasi-inverses — the Python code maps both `Kompress` and
//! `Passthrough` strategies back to `PlainText`, so the round-trip is
//! lossy on those two strategies.

use super::types::CompressionStrategy;
use crate::transforms::ContentType;

/// Map a detected content type to the strategy that handles it.
/// Mirrors Python's `_strategy_from_detection_type`.
///
/// Python uses `mapping.get(content_type, self.config.fallback_strategy)`,
/// but the mapping is total over `ContentType` today — every variant has
/// an entry — so the fallback is dead-code-equivalent. We omit it here:
/// this function always returns a specialized strategy, and
/// `ContentRouter::_determine_strategy` (PR5) is where
/// `config.fallback_strategy` actually applies — for low-confidence
/// detections that bypass the mapping entirely.
pub fn strategy_for_content_type(content_type: ContentType) -> CompressionStrategy {
    match content_type {
        ContentType::SourceCode => CompressionStrategy::CodeAware,
        ContentType::JsonArray => CompressionStrategy::SmartCrusher,
        ContentType::SearchResults => CompressionStrategy::Search,
        ContentType::BuildOutput => CompressionStrategy::Log,
        ContentType::GitDiff => CompressionStrategy::Diff,
        ContentType::Html => CompressionStrategy::Html,
        ContentType::PlainText => CompressionStrategy::Text,
    }
}

/// Inverse of [`strategy_for_content_type`]. Mirrors Python's
/// `_content_type_from_strategy`. Returns `PlainText` for strategies
/// that don't have a unique content type (`Kompress`, `Passthrough`,
/// `Mixed`) — same as Python.
pub fn content_type_for_strategy(strategy: CompressionStrategy) -> ContentType {
    match strategy {
        CompressionStrategy::CodeAware => ContentType::SourceCode,
        CompressionStrategy::SmartCrusher => ContentType::JsonArray,
        CompressionStrategy::Search => ContentType::SearchResults,
        CompressionStrategy::Log => ContentType::BuildOutput,
        CompressionStrategy::Diff => ContentType::GitDiff,
        CompressionStrategy::Html => ContentType::Html,
        // Python's mapping returns PlainText for all of these:
        CompressionStrategy::Text
        | CompressionStrategy::Kompress
        | CompressionStrategy::Passthrough
        | CompressionStrategy::Mixed => ContentType::PlainText,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_for_each_content_type_matches_python() {
        let cases = [
            (ContentType::SourceCode, CompressionStrategy::CodeAware),
            (ContentType::JsonArray, CompressionStrategy::SmartCrusher),
            (ContentType::SearchResults, CompressionStrategy::Search),
            (ContentType::BuildOutput, CompressionStrategy::Log),
            (ContentType::GitDiff, CompressionStrategy::Diff),
            (ContentType::Html, CompressionStrategy::Html),
            (ContentType::PlainText, CompressionStrategy::Text),
        ];
        for (ct, expected) in cases {
            assert_eq!(
                strategy_for_content_type(ct),
                expected,
                "ContentType::{:?}",
                ct
            );
        }
    }

    #[test]
    fn content_type_for_each_strategy_matches_python() {
        let cases = [
            (CompressionStrategy::CodeAware, ContentType::SourceCode),
            (CompressionStrategy::SmartCrusher, ContentType::JsonArray),
            (CompressionStrategy::Search, ContentType::SearchResults),
            (CompressionStrategy::Log, ContentType::BuildOutput),
            (CompressionStrategy::Diff, ContentType::GitDiff),
            (CompressionStrategy::Html, ContentType::Html),
            (CompressionStrategy::Text, ContentType::PlainText),
            // Python collapses these three to PlainText:
            (CompressionStrategy::Kompress, ContentType::PlainText),
            (CompressionStrategy::Passthrough, ContentType::PlainText),
            (CompressionStrategy::Mixed, ContentType::PlainText),
        ];
        for (s, expected) in cases {
            assert_eq!(content_type_for_strategy(s), expected, "{:?}", s);
        }
    }

    #[test]
    fn round_trip_lossless_for_specialized_strategies() {
        // For every specialized strategy, ContentType → Strategy →
        // ContentType is the identity.
        let specialized = [
            CompressionStrategy::CodeAware,
            CompressionStrategy::SmartCrusher,
            CompressionStrategy::Search,
            CompressionStrategy::Log,
            CompressionStrategy::Diff,
            CompressionStrategy::Html,
            CompressionStrategy::Text,
        ];
        for s in specialized {
            let ct = content_type_for_strategy(s);
            let s2 = strategy_for_content_type(ct);
            assert_eq!(s2, s, "round-trip {:?}", s);
        }
    }
}
