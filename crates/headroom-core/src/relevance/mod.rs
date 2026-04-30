//! Relevance scoring — Rust port of `headroom/relevance/`.
//!
//! Used by SmartCrusher's planning layer to decide which items in a tool
//! output match the user's query (the user's recent prompts plus the
//! assistant's tool-call argument JSON, joined). Items above a relevance
//! threshold are pinned into `keep_indices`.
//!
//! # Scorer ladder
//!
//! 1. **BM25** (`bm25`): keyword overlap with TF-IDF + length
//!    normalization. No ML deps. Excellent for exact-match cases (UUIDs,
//!    field=value filters). Tool-call arguments are usually literal
//!    keywords that appear verbatim in the response, so BM25 catches
//!    most cases.
//! 2. **Embedding** (future commit): sentence-transformer ONNX model
//!    for semantic matching when query and items use different
//!    vocabularies.
//! 3. **Hybrid** (future commit): combines BM25 and embedding signals.
//!
//! Each scorer implements the `RelevanceScorer` trait — same surface
//! as Python's abstract base class.

mod base;
mod bm25;
#[cfg(feature = "fastembed")]
mod embedding;
#[cfg(not(feature = "fastembed"))]
mod embedding_stub;
mod hybrid;

pub use base::{default_batch_score, RelevanceScore, RelevanceScorer};
pub use bm25::BM25Scorer;
#[cfg(feature = "fastembed")]
pub use embedding::EmbeddingScorer;
#[cfg(not(feature = "fastembed"))]
pub use embedding_stub::EmbeddingScorer;
pub use hybrid::HybridScorer;

/// Factory mirroring Python's `relevance.create_scorer` (`__init__.py:72`).
///
/// Returns a boxed trait object so callers don't have to know which
/// concrete scorer they got. `tier`:
///
/// - `"hybrid"` (default) — `HybridScorer` (BM25 + embedding fusion;
///   gracefully falls back to BM25 + boost when embeddings stubbed).
/// - `"bm25"` — `BM25Scorer` (pure keyword).
/// - `"embedding"` — `EmbeddingScorer` (available only when the
///   `fastembed` feature is enabled; otherwise returns `Err` to mirror
///   Python's optional-backend behavior).
pub fn create_scorer(tier: &str) -> Result<Box<dyn RelevanceScorer + Send + Sync>, String> {
    match tier.to_lowercase().as_str() {
        "bm25" => Ok(Box::new(BM25Scorer::default())),
        "hybrid" => Ok(Box::new(HybridScorer::default())),
        "embedding" => {
            let s = EmbeddingScorer::default();
            if s.is_available() {
                Ok(Box::new(s))
            } else {
                Err(
                    "EmbeddingScorer requires the ONNX backend (not yet implemented in Rust)"
                        .to_string(),
                )
            }
        }
        other => Err(format!(
            "Unknown scorer tier: {}. Valid tiers: bm25, embedding, hybrid",
            other
        )),
    }
}
