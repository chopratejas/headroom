//! Compression transforms — Rust ports of `headroom.transforms.*`.
//!
//! # Guiding principle: information preservation > aggressive compression
//!
//! When in doubt, prefer keeping bytes. The fixtures lock the Python
//! algorithm's exact behavior, so this crate cannot drop information that
//! Python keeps. But the inverse is also true — we MUST drop everything
//! Python drops, even when it feels lossy. Stage 3a's faithful port is
//! parity-bound. A follow-up stage (token-budget-aware compression) is
//! where we earn the right to keep more.
//!
//! Observability is the escape hatch: every transform returns a sidecar
//! `Stats` struct with the granular metrics Python doesn't emit (e.g. which
//! files were dropped, how many context lines were trimmed, per-file hunk
//! drop counts). These flow through `tracing` spans for OTel scraping in
//! prod and are returned alongside the parity-equal output for tests.

pub mod diff_compressor;

pub use diff_compressor::{
    DiffCompressionResult, DiffCompressor, DiffCompressorConfig, DiffCompressorStats,
};
