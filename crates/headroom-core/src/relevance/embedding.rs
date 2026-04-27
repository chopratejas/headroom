//! Embedding-based relevance scorer (sentence-transformers).
//!
//! # Status: STUB — real ONNX implementation lands in a follow-up commit.
//!
//! Direct port of `headroom/relevance/embedding.py`. Python's
//! `EmbeddingScorer.is_available()` returns `False` when
//! `sentence-transformers` is not installed; this Rust stub mirrors
//! that "not available" path exactly. `HybridScorer` already handles
//! this case via its BM25-only fallback (with a small score boost),
//! so the planning layer can call into hybrid right now and behave
//! parity-equal with a Python deployment that has no ML deps.
//!
//! When the real ONNX implementation lands:
//! - Add `ort` (ONNX Runtime) and reuse the existing `tokenizers` dep.
//! - Auto-download `sentence-transformers/all-MiniLM-L6-v2` via `hf-hub`.
//! - Mean-pool the token embeddings, L2-normalize, cosine-similarity.
//! - Flip `is_available()` to `true` and `score()`/`score_batch()` to
//!   the inference path.
//!
//! No public-API change required at the call site — `HybridScorer` can
//! continue to reach for `EmbeddingScorer::new()` and check
//! `is_available()` on it.

use super::base::{RelevanceScore, RelevanceScorer};

/// Stub scorer. Always reports `is_available() == false`.
///
/// Construct one when you want to express "embedding scoring desired,
/// fall back if unavailable". Production code goes through
/// `HybridScorer` which handles the fallback transparently.
pub struct EmbeddingScorer {
    /// Stored for future ONNX work. Currently unused since the stub
    /// can't actually load a model.
    pub model_name: String,
}

impl Default for EmbeddingScorer {
    fn default() -> Self {
        EmbeddingScorer {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        }
    }
}

impl EmbeddingScorer {
    pub fn new(model_name: impl Into<String>) -> Self {
        EmbeddingScorer {
            model_name: model_name.into(),
        }
    }
}

impl RelevanceScorer for EmbeddingScorer {
    fn score(&self, _item: &str, _context: &str) -> RelevanceScore {
        // Defensive: `HybridScorer` checks `is_available()` first so
        // this branch shouldn't be reached. Returning empty rather
        // than panicking keeps the trait safe to call directly.
        RelevanceScore::empty("Embedding: ONNX backend not yet implemented")
    }

    fn score_batch(&self, items: &[&str], _context: &str) -> Vec<RelevanceScore> {
        items
            .iter()
            .map(|_| RelevanceScore::empty("Embedding: ONNX backend not yet implemented"))
            .collect()
    }

    fn is_available(&self) -> bool {
        // Pinned to the Python "sentence-transformers not installed"
        // branch. Flips to `true` when the real ONNX impl lands.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_unavailable() {
        assert!(!EmbeddingScorer::default().is_available());
    }

    #[test]
    fn stub_score_returns_empty() {
        let s = EmbeddingScorer::default();
        let r = s.score("item text", "query");
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn stub_score_batch_one_per_item() {
        let s = EmbeddingScorer::default();
        let items = ["a", "b", "c"];
        let scores = s.score_batch(&items, "ctx");
        assert_eq!(scores.len(), 3);
        for sc in scores {
            assert_eq!(sc.score, 0.0);
        }
    }

    #[test]
    fn stores_model_name() {
        let s = EmbeddingScorer::new("custom/model");
        assert_eq!(s.model_name, "custom/model");
    }

    #[test]
    fn default_model_is_all_minilm() {
        assert_eq!(
            EmbeddingScorer::default().model_name,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }
}
