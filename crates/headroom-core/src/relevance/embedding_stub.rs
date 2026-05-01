//! Stub embedding scorer used when the `fastembed` feature is disabled.

use super::base::{default_batch_score, RelevanceScore, RelevanceScorer};

#[derive(Debug)]
pub struct EmbeddingScorer {
    pub model_name: String,
}

impl Default for EmbeddingScorer {
    fn default() -> Self {
        EmbeddingScorer {
            model_name: "fastembed-disabled".to_string(),
        }
    }
}

impl EmbeddingScorer {
    pub fn try_new() -> Result<Self, String> {
        Err("EmbeddingScorer requires the `fastembed` cargo feature".to_string())
    }
}

impl RelevanceScorer for EmbeddingScorer {
    fn score(&self, _item: &str, _context: &str) -> RelevanceScore {
        RelevanceScore::empty("Embedding: fastembed feature disabled")
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        default_batch_score(self, items, context)
    }

    fn is_available(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_embedding_is_unavailable() {
        let scorer = EmbeddingScorer::default();
        assert!(!scorer.is_available());
    }

    #[test]
    fn try_new_errors_without_feature() {
        let err = EmbeddingScorer::try_new().expect_err("feature gate should reject fastembed");
        assert!(err.contains("fastembed"));
    }
}
