//! Build the per-process `IntelligentContextManager`.
//!
//! Constructed once at proxy startup, stored in `AppState`, shared
//! across every request via `Arc`. The `MessageScorer` inside has a
//! `Mutex<HashMap>` embedding cache; contention is low because we
//! aren't wiring an `EmbeddingProvider` yet (that's a follow-up).
//!
//! # Tokenizer choice
//!
//! `IntelligentContextManager` needs a `Tokenizer` to count tokens
//! for the `should_apply` budget gate. We use `TiktokenCounter` with
//! the `gpt-4o-mini` (`o200k_base`) encoding because:
//!
//! - It's the most modern tiktoken vocabulary that's broadly
//!   compatible with both OpenAI and Anthropic message shapes.
//! - It's strictly more accurate than the `EstimatingCounter`
//!   (chars/4) default that the PyO3 binding hardcodes.
//! - For Anthropic specifically, the actual tokenizer is bespoke and
//!   not publicly available; tiktoken-based counting is the
//!   industry-standard close-enough estimate. Errors are O(±5%)
//!   which doesn't change `should_apply` decisions.
//!
//! # CCR store
//!
//! An `InMemoryCcrStore` is constructed alongside the manager. When
//! ICM drops messages, their original JSON gets stashed under a
//! content-hash key and a marker is inserted into the surviving
//! message stream. If the LLM later calls a `ccr_retrieve` tool, the
//! handler can serve the dropped content from this store. (The
//! retrieval-handling tool is a separate concern; we just construct
//! the store so drops are recoverable.)

use std::sync::Arc;

use headroom_core::ccr::{CcrStore, InMemoryCcrStore};
use headroom_core::context::{IcmConfig, IntelligentContextManager};
use headroom_core::tokenizer::{TiktokenCounter, Tokenizer};

/// Construct the proxy's shared ICM. Returns `Arc` for cheap cloning
/// into request handlers.
///
/// Errors only on tokenizer construction — `gpt-4o-mini` is always
/// available since tiktoken-rs ships its vocabulary, but we bubble
/// up the error path for symmetry with the rest of the proxy's
/// fallible startup.
pub fn build_icm() -> Result<Arc<IntelligentContextManager>, String> {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(
        TiktokenCounter::for_model("gpt-4o-mini")
            .map_err(|e| format!("init TiktokenCounter for gpt-4o-mini: {e}"))?,
    );
    let ccr: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
    Ok(Arc::new(IntelligentContextManager::new(
        IcmConfig::default(),
        tokenizer,
        Some(ccr),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_icm_succeeds() {
        let icm = build_icm().expect("ICM should build");
        // Smoke: should_apply on an empty list under a generous
        // budget returns false (no work).
        assert!(!icm.should_apply(&[], 128_000, 4_000));
    }
}
