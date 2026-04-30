//! Compression pipeline — formal orchestrator for lossless → lossy → CCR.
//!
//! # Why this module exists
//!
//! Before Phase 3g each compressor (SmartCrusher, DiffCompressor,
//! LogCompressor, SearchCompressor, TagProtector) carried its own
//! ad-hoc decision tree: parse, score, drop, optionally emit a CCR
//! marker. SmartCrusher's 3c.2 refactor made the decision explicit
//! inside that one transform; the rest still decide privately. That
//! makes the system hard to reason about (which transforms ran in
//! what order, how much each saved, why CCR fired or didn't) and
//! impossible to extend without copying the same scaffolding.
//!
//! This module replaces that scaffolding with two traits and an
//! orchestrator:
//!
//! * [`LosslessTransform`] — preserves all information; can run
//!   first and stop early if structural compression suffices.
//! * [`LossyTransform`] — drops bytes deliberately. Runs only after
//!   the lossless pass. Reports a calibrated `confidence` so callers
//!   and telemetry can tell which transforms were trusted vs taken
//!   on faith.
//! * [`CompressionPipeline`] — content-type-keyed dispatch over the
//!   two transform sets, with budget gates between phases.
//!
//! # The pipeline contract
//!
//! ```text
//! input → DetectContentType (Magika, already shipped)
//!       → LosslessTransforms[content_type]   stop if structural compression suffices
//!       → BudgetCheck
//!       → LossyTransforms[content_type]      includes ProseFieldCompressor (Phase 3g PR4)
//!       → BudgetCheck
//!       → CCR for everything beyond budget   (Phase 3g PR3+)
//! ```
//!
//! PR1 ships traits + orchestrator + one real impl per trait
//! ([`JsonMinifier`] for lossless, [`LineImportanceFilter`] for lossy).
//! PR2 wraps existing structural transforms (Diff/Log/Search/Tag) in
//! the trait shape. PR3 refactors SmartCrusher to use the orchestrator
//! and starts retiring Python orchestration glue. PR4 adds
//! ProseFieldCompressor (the parser/model boundary primitive). PR5
//! migrates the remaining compressors and lets us delete the Python
//! `ContentRouter` strategy dispatch.
//!
//! # No regex
//!
//! By project convention (and feedback memory), nothing in this module
//! uses the `regex` crate. JsonMinifier is `serde_json` round-trip;
//! LineImportanceFilter walks `str::lines()` and consumes the existing
//! `signals::LineImportanceDetector` trait (which uses aho-corasick
//! plus an ASCII word-boundary post-filter, also no regex).

pub mod json_minifier;
pub mod line_importance_filter;
pub mod orchestrator;
pub mod traits;
pub mod wrappers;

pub use json_minifier::JsonMinifier;
pub use line_importance_filter::{LineImportanceFilter, LineImportanceFilterConfig};
pub use orchestrator::{
    CompressionPipeline, CompressionPipelineBuilder, PipelineConfig, PipelineResult,
};
pub use traits::{
    CompressionContext, LosslessTransform, LossyTransform, TransformError, TransformResult,
};
pub use wrappers::{DiffCompressorWrapper, LogCompressorWrapper, SearchCompressorWrapper};
