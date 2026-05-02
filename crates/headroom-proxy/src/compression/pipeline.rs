//! Build the per-process [`CompressionPipeline`] + [`CcrStore`].
//!
//! Constructed once at proxy startup (when `--compression` is on)
//! and shared across every request. Both are `Arc`-cheap to clone
//! into request handlers.
//!
//! # What's registered
//!
//! Default reformat + offload set from `headroom-core`'s pipeline:
//!
//! - **`JsonMinifier`** (reformat, JsonArray) — strip whitespace.
//! - **`LogTemplate`** (reformat, BuildOutput) — Drain-style template
//!   miner that collapses repeated log lines.
//! - **`JsonOffload`** wrapping SmartCrusher (offload, JsonArray) —
//!   schema-dedup'd JSON array compression with CCR backup.
//! - **`LogOffload`** (offload, BuildOutput) — line-template offload.
//! - **`DiffOffload`** (offload, GitDiff) — diff-aware compression.
//! - **`DiffNoise`** (offload, GitDiff) — drops lockfile + whitespace
//!   diffs.
//!
//! `SearchOffload` is intentionally NOT registered (deprecated
//! upstream — see headroom_core::transforms::pipeline::offloads
//! head doc for rationale).
//!
//! # CCR store
//!
//! In-memory `DashMap`-backed store. Compressed offloads stash
//! their original payload here; if the LLM later calls a
//! `headroom_retrieve`-shaped tool the handler can look up the
//! payload by cache key.

use std::sync::Arc;

use headroom_core::ccr::{CcrStore, InMemoryCcrStore};
use headroom_core::transforms::pipeline::{
    CompressionPipeline, DiffNoise, DiffOffload, JsonMinifier, JsonOffload, LogOffload,
    LogTemplate, PipelineConfig,
};

/// Build the proxy's shared compression pipeline + CCR store. Both
/// are returned wrapped in `Arc` so handler code clones cheaply.
///
/// Infallible — every component has a `Default` constructor.
pub fn build_pipeline() -> (Arc<CompressionPipeline>, Arc<dyn CcrStore>) {
    let cfg = PipelineConfig::default();
    let pipeline = CompressionPipeline::builder()
        // Reformats — lossless first pass.
        .with_reformat(JsonMinifier)
        .with_reformat(LogTemplate::new(cfg.reformat.log_template))
        // Offloads — bloat-gated CCR write.
        .with_offload(JsonOffload::new(cfg.offload.json))
        .with_offload(LogOffload::new(cfg.bloat.log))
        .with_offload(DiffOffload::new(cfg.bloat.diff))
        .with_offload(DiffNoise::new(cfg.offload.diff_noise))
        .build();

    let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
    (Arc::new(pipeline), store)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_builds() {
        let (_p, _s) = build_pipeline();
    }
}
