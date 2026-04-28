//! `ContentRouter` data types.
//!
//! Direct port of the enums and dataclasses at the top of
//! `headroom/transforms/content_router.py`. String tags on
//! [`CompressionStrategy`] match the Python `Enum` values so PyO3
//! bridges and JSON fixtures cross the language boundary unchanged.

use std::str::FromStr;

use crate::transforms::ContentType;

/// Available compression strategies. String tag (`as_str`) matches
/// Python's `CompressionStrategy.<NAME>.value` 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionStrategy {
    CodeAware,
    SmartCrusher,
    Search,
    Log,
    Kompress,
    Text,
    Diff,
    Html,
    Mixed,
    Passthrough,
}

impl CompressionStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionStrategy::CodeAware => "code_aware",
            CompressionStrategy::SmartCrusher => "smart_crusher",
            CompressionStrategy::Search => "search",
            CompressionStrategy::Log => "log",
            CompressionStrategy::Kompress => "kompress",
            CompressionStrategy::Text => "text",
            CompressionStrategy::Diff => "diff",
            CompressionStrategy::Html => "html",
            CompressionStrategy::Mixed => "mixed",
            CompressionStrategy::Passthrough => "passthrough",
        }
    }
}

/// Inverse of [`CompressionStrategy::as_str`]. Implemented as the
/// stdlib `FromStr` trait so the parse path is discoverable; the error
/// type is `()` because the only failure mode is "unknown tag" and
/// callers (e.g. PyO3 bridge) build their own diagnostic from the
/// rejected string.
impl FromStr for CompressionStrategy {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "code_aware" => Self::CodeAware,
            "smart_crusher" => Self::SmartCrusher,
            "search" => Self::Search,
            "log" => Self::Log,
            "kompress" => Self::Kompress,
            "text" => Self::Text,
            "diff" => Self::Diff,
            "html" => Self::Html,
            "mixed" => Self::Mixed,
            "passthrough" => Self::Passthrough,
            _ => return Err(()),
        })
    }
}

/// Record of a single routing decision (one section, one strategy).
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub content_type: ContentType,
    pub strategy: CompressionStrategy,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub confidence: f64,
    pub section_index: usize,
}

impl RoutingDecision {
    /// `compressed / original` (1.0 when original_tokens is 0 — matches
    /// Python's safe-divide).
    pub fn compression_ratio(&self) -> f64 {
        if self.original_tokens == 0 {
            1.0
        } else {
            self.compressed_tokens as f64 / self.original_tokens as f64
        }
    }
}

/// A typed slice of a larger payload (used by the mixed-content
/// splitter — added in a later PR but the type lives here).
#[derive(Debug, Clone)]
pub struct ContentSection {
    pub content: String,
    pub content_type: ContentType,
    pub language: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub is_code_fence: bool,
}

/// Result of a `ContentRouter::compress` call. Carries the compressed
/// output alongside per-decision routing metadata for telemetry /
/// debug logging.
#[derive(Debug, Clone)]
pub struct RouterCompressionResult {
    pub compressed: String,
    pub original: String,
    pub strategy_used: CompressionStrategy,
    pub routing_log: Vec<RoutingDecision>,
    pub sections_processed: usize,
}

impl RouterCompressionResult {
    pub fn new(compressed: String, original: String, strategy_used: CompressionStrategy) -> Self {
        Self {
            compressed,
            original,
            strategy_used,
            routing_log: Vec::new(),
            sections_processed: 1,
        }
    }

    pub fn total_original_tokens(&self) -> usize {
        self.routing_log.iter().map(|r| r.original_tokens).sum()
    }

    pub fn total_compressed_tokens(&self) -> usize {
        self.routing_log.iter().map(|r| r.compressed_tokens).sum()
    }

    /// Total `compressed/original`. Returns 1.0 when no tokens were
    /// recorded — matches Python's safe-divide.
    pub fn compression_ratio(&self) -> f64 {
        let orig = self.total_original_tokens();
        if orig == 0 {
            1.0
        } else {
            self.total_compressed_tokens() as f64 / orig as f64
        }
    }

    pub fn tokens_saved(&self) -> usize {
        self.total_original_tokens()
            .saturating_sub(self.total_compressed_tokens())
    }

    /// Percent of tokens saved. `0.0` when no tokens were recorded.
    pub fn savings_percentage(&self) -> f64 {
        let orig = self.total_original_tokens();
        if orig == 0 {
            0.0
        } else {
            (self.tokens_saved() as f64 / orig as f64) * 100.0
        }
    }

    /// Human-readable summary. Format matches Python's `summary()`
    /// output verbatim, down to the thousand-separators and rounding,
    /// so existing log scrapers keep working when the router moves
    /// in-process.
    pub fn summary(&self) -> String {
        let total_orig = self.total_original_tokens();
        let total_comp = self.total_compressed_tokens();
        let pct = self.savings_percentage();
        if self.strategy_used == CompressionStrategy::Mixed {
            // Distinct strategies, sorted to match Python's set-formatting
            // when display happens to round-trip the same order. Python's
            // `{strategies}` print order is not guaranteed; we sort so
            // the message is at least stable across runs.
            let mut strategies: Vec<&'static str> = self
                .routing_log
                .iter()
                .map(|r| r.strategy.as_str())
                .collect();
            strategies.sort_unstable();
            strategies.dedup();
            // Match Python's `{...}` set repr: `{'a', 'b'}`.
            let formatted = if strategies.is_empty() {
                "set()".to_string()
            } else {
                let inner = strategies
                    .iter()
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{inner}}}")
            };
            format!(
                "Mixed content: {sections} sections, routed to {fmt}. \
                 {orig}→{comp} tokens ({pct:.0}% saved)",
                sections = self.sections_processed,
                fmt = formatted,
                orig = thousands(total_orig),
                comp = thousands(total_comp),
                pct = pct,
            )
        } else {
            format!(
                "Pure {strategy}: {orig}→{comp} tokens ({pct:.0}% saved)",
                strategy = self.strategy_used.as_str(),
                orig = thousands(total_orig),
                comp = thousands(total_comp),
                pct = pct,
            )
        }
    }
}

/// Python-style `{:,}` thousand-separator formatting for a `usize`.
/// Avoids pulling a heavy formatter crate just for this one call site.
fn thousands(mut n: usize) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut groups: Vec<String> = Vec::new();
    while n > 0 {
        let g = n % 1000;
        n /= 1000;
        if n > 0 {
            groups.push(format!("{g:03}"));
        } else {
            groups.push(g.to_string());
        }
    }
    groups.reverse();
    groups.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_string_tags_match_python() {
        let cases = [
            (CompressionStrategy::CodeAware, "code_aware"),
            (CompressionStrategy::SmartCrusher, "smart_crusher"),
            (CompressionStrategy::Search, "search"),
            (CompressionStrategy::Log, "log"),
            (CompressionStrategy::Kompress, "kompress"),
            (CompressionStrategy::Text, "text"),
            (CompressionStrategy::Diff, "diff"),
            (CompressionStrategy::Html, "html"),
            (CompressionStrategy::Mixed, "mixed"),
            (CompressionStrategy::Passthrough, "passthrough"),
        ];
        for (s, tag) in cases {
            assert_eq!(s.as_str(), tag);
            assert_eq!(tag.parse::<CompressionStrategy>(), Ok(s));
        }
        assert_eq!("nope".parse::<CompressionStrategy>(), Err(()));
    }

    #[test]
    fn routing_decision_ratio_handles_zero() {
        let d = RoutingDecision {
            content_type: ContentType::PlainText,
            strategy: CompressionStrategy::Text,
            original_tokens: 0,
            compressed_tokens: 0,
            confidence: 1.0,
            section_index: 0,
        };
        assert_eq!(d.compression_ratio(), 1.0);
    }

    #[test]
    fn routing_decision_ratio_normal_case() {
        let d = RoutingDecision {
            content_type: ContentType::JsonArray,
            strategy: CompressionStrategy::SmartCrusher,
            original_tokens: 200,
            compressed_tokens: 50,
            confidence: 1.0,
            section_index: 0,
        };
        assert_eq!(d.compression_ratio(), 0.25);
    }

    #[test]
    fn router_result_aggregates_log() {
        let mut r =
            RouterCompressionResult::new("out".into(), "in".into(), CompressionStrategy::Mixed);
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::JsonArray,
            strategy: CompressionStrategy::SmartCrusher,
            original_tokens: 100,
            compressed_tokens: 25,
            confidence: 1.0,
            section_index: 0,
        });
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::SourceCode,
            strategy: CompressionStrategy::CodeAware,
            original_tokens: 100,
            compressed_tokens: 75,
            confidence: 0.9,
            section_index: 1,
        });
        r.sections_processed = 2;
        assert_eq!(r.total_original_tokens(), 200);
        assert_eq!(r.total_compressed_tokens(), 100);
        assert_eq!(r.compression_ratio(), 0.5);
        assert_eq!(r.tokens_saved(), 100);
        assert_eq!(r.savings_percentage(), 50.0);
    }

    #[test]
    fn router_result_handles_empty_log() {
        let r = RouterCompressionResult::new(
            "out".into(),
            "in".into(),
            CompressionStrategy::Passthrough,
        );
        assert_eq!(r.compression_ratio(), 1.0);
        assert_eq!(r.savings_percentage(), 0.0);
        assert_eq!(r.tokens_saved(), 0);
    }

    #[test]
    fn tokens_saved_floors_at_zero() {
        // Compressed > original is degenerate but possible (e.g. compressor
        // adds metadata). Saturating-sub keeps `tokens_saved` non-negative.
        let mut r =
            RouterCompressionResult::new("out".into(), "in".into(), CompressionStrategy::Text);
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::PlainText,
            strategy: CompressionStrategy::Text,
            original_tokens: 50,
            compressed_tokens: 80,
            confidence: 1.0,
            section_index: 0,
        });
        assert_eq!(r.tokens_saved(), 0);
    }

    #[test]
    fn summary_pure_strategy_format() {
        let mut r = RouterCompressionResult::new(
            "out".into(),
            "in".into(),
            CompressionStrategy::SmartCrusher,
        );
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::JsonArray,
            strategy: CompressionStrategy::SmartCrusher,
            original_tokens: 1500,
            compressed_tokens: 750,
            confidence: 1.0,
            section_index: 0,
        });
        assert_eq!(
            r.summary(),
            "Pure smart_crusher: 1,500→750 tokens (50% saved)"
        );
    }

    #[test]
    fn summary_mixed_strategy_format() {
        let mut r =
            RouterCompressionResult::new("out".into(), "in".into(), CompressionStrategy::Mixed);
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::JsonArray,
            strategy: CompressionStrategy::SmartCrusher,
            original_tokens: 5_000,
            compressed_tokens: 1_000,
            confidence: 1.0,
            section_index: 0,
        });
        r.routing_log.push(RoutingDecision {
            content_type: ContentType::SourceCode,
            strategy: CompressionStrategy::CodeAware,
            original_tokens: 5_000,
            compressed_tokens: 4_000,
            confidence: 0.9,
            section_index: 1,
        });
        r.sections_processed = 2;
        let s = r.summary();
        assert!(
            s.starts_with("Mixed content: 2 sections, routed to "),
            "got: {s}"
        );
        assert!(s.contains("'smart_crusher'"));
        assert!(s.contains("'code_aware'"));
        assert!(s.ends_with("10,000→5,000 tokens (50% saved)"));
    }

    #[test]
    fn thousands_formatting() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(1), "1");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(12_345), "12,345");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }
}
