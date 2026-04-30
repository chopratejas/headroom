//! Wrappers that adapt existing structural transforms (DiffCompressor,
//! LogCompressor, SearchCompressor) to the Phase 3g pipeline trait
//! shape ([`LosslessTransform`] / [`LossyTransform`]).
//!
//! Each wrapper threads the underlying compressor's native config
//! through unchanged — the wrapping is deliberately thin so the parity
//! contract of the wrapped transform is preserved. The wrappers map
//! `cache_key → reversible_via`, surface skip/error states as
//! [`TransformError`], and self-declare their `applies_to` content
//! types so the orchestrator can dispatch correctly.
//!
//! TagProtector wrapper is deferred until PR #324 lands the Rust port
//! — once that merges, a fourth file `tag_protector_wrapper.rs` slots
//! in here as a [`LosslessTransform`] (placeholder substitution is
//! reversible by construction).
//!
//! [`LosslessTransform`]: super::traits::LosslessTransform
//! [`LossyTransform`]: super::traits::LossyTransform
//! [`TransformError`]: super::traits::TransformError

pub mod diff_compressor_wrapper;
pub mod log_compressor_wrapper;
pub mod search_compressor_wrapper;

pub use diff_compressor_wrapper::DiffCompressorWrapper;
pub use log_compressor_wrapper::LogCompressorWrapper;
pub use search_compressor_wrapper::SearchCompressorWrapper;
