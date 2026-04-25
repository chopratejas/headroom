//! HuggingFace `tokenizers`-crate adapter implementing [`Tokenizer`].
//!
//! Loads a `tokenizer.json` (HuggingFace's serialization format) and counts
//! tokens via real BPE / Unigram / WordPiece — whatever the file describes.
//! This closes the gap between OpenAI (tiktoken, byte-equal) and the
//! Anthropic/Google chars-per-token fallback: every other major family
//! (Cohere `command-*`, Llama-3.x, Mistral, Qwen, BERT, T5, …) publishes a
//! `tokenizer.json` on the HuggingFace Hub and the `tokenizers` crate is a
//! pure-Rust loader, so we don't have to estimate.
//!
//! # What's NOT here
//! - **No HuggingFace Hub auto-download.** Callers pass bytes or a path. A
//!   later stage can add `hf-hub` integration behind a Cargo feature; doing it
//!   here would drag in ureq/rustls, a `~/.cache/huggingface` dependency, and
//!   gated-model auth flows that don't belong in the core crate.
//! - **No tokenizer.json bundled in the binary.** Bundling Llama / Cohere
//!   tokenizers would add several MB of binary bloat for code paths most users
//!   don't hit.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use tokenizers::Tokenizer as HfInner;

use super::{Backend, Tokenizer};

#[derive(Debug, Error)]
pub enum HfTokenizerError {
    /// The bytes / file did not parse as a valid HuggingFace `tokenizer.json`,
    /// or the model component referenced an unsupported algorithm.
    #[error("failed to load tokenizer for `{name}`: {source}")]
    Load {
        name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Token counter backed by a HuggingFace `tokenizer.json`.
///
/// Cheap to clone — internally an `Arc<tokenizers::Tokenizer>`. Construct once
/// at startup, share across handlers.
#[derive(Clone)]
pub struct HfTokenizer {
    name: String,
    inner: Arc<HfInner>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("name", &self.name)
            .finish()
    }
}

impl HfTokenizer {
    /// Build from raw `tokenizer.json` bytes. Use this when the tokenizer is
    /// embedded via `include_bytes!` or fetched from a non-HF source.
    pub fn from_bytes(name: impl Into<String>, bytes: &[u8]) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_bytes(bytes).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    /// Build from a `tokenizer.json` on disk.
    pub fn from_file(
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, HfTokenizerError> {
        let name = name.into();
        let inner = HfInner::from_file(path.as_ref()).map_err(|e| HfTokenizerError::Load {
            name: name.clone(),
            source: e,
        })?;
        Ok(Self {
            name,
            inner: Arc::new(inner),
        })
    }

    /// The logical name this tokenizer was registered under (e.g.
    /// `"command-r-plus"`). Used in logs and metrics.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Tokenizer for HfTokenizer {
    fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        // `add_special_tokens=false` matches the spirit of
        // `tiktoken.encode_ordinary`: count *content* tokens, leaving
        // BOS/EOS/CLS/SEP padding to be added (or not) by the upstream API.
        // Different providers add different specials, so counting them here
        // would systematically over-charge. Documented for future readers.
        match self.inner.encode(text, false) {
            Ok(enc) => enc.len(),
            // `encode` only fails for malformed inputs that pass UTF-8 but
            // violate the tokenizer's constraints (e.g. a normalizer that
            // rejects certain code points). We degrade to "0 known tokens"
            // rather than panic — the proxy must keep flowing requests.
            Err(_) => 0,
        }
    }

    fn backend(&self) -> Backend {
        Backend::HuggingFace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid `tokenizer.json`: WordLevel model, Whitespace
    /// pre-tokenizer, three-token vocabulary. Lets us test the API surface
    /// without committing a multi-MB tokenizer fixture or hitting the
    /// HuggingFace Hub.
    const TINY_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
            "unk_token": "[UNK]"
        }
    }"#;

    fn tiny() -> HfTokenizer {
        HfTokenizer::from_bytes("tiny-test", TINY_TOKENIZER_JSON.as_bytes())
            .expect("tiny tokenizer.json parses")
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(tiny().count_text(""), 0);
    }

    #[test]
    fn known_vocab_matches_count() {
        let t = tiny();
        // Each word in the vocab is one token; whitespace splits them.
        assert_eq!(t.count_text("hello"), 1);
        assert_eq!(t.count_text("hello world"), 2);
        assert_eq!(t.count_text("hello world hello"), 3);
    }

    #[test]
    fn unknown_words_become_unk() {
        // OOV tokens collapse to [UNK] — still 1 per whitespace-split chunk.
        let t = tiny();
        assert_eq!(t.count_text("supercalifragilistic"), 1);
        assert_eq!(t.count_text("foo bar baz"), 3);
    }

    #[test]
    fn deterministic() {
        let t = tiny();
        let s = "hello world hello world";
        let first = t.count_text(s);
        for _ in 0..100 {
            assert_eq!(t.count_text(s), first);
        }
    }

    #[test]
    fn unicode_does_not_panic() {
        let t = tiny();
        for s in ["héllo wörld", "你好世界", "🦀 ferris", "\n\t\r"] {
            // We only assert non-panic and a reasonable upper bound; the exact
            // count depends on Whitespace pre-tokenizer behavior, which is
            // tokenizers-crate internal.
            let n = t.count_text(s);
            assert!(n < s.len() * 4 + 10, "absurd count {n} for {s:?}");
        }
    }

    #[test]
    fn invalid_bytes_returns_error() {
        let r = HfTokenizer::from_bytes("bad", b"not a tokenizer.json");
        assert!(matches!(r, Err(HfTokenizerError::Load { .. })));
    }

    #[test]
    fn name_round_trips() {
        let t = tiny();
        assert_eq!(t.name(), "tiny-test");
    }

    #[test]
    fn backend_is_huggingface() {
        assert_eq!(tiny().backend(), Backend::HuggingFace);
    }

    #[test]
    fn clone_shares_inner() {
        let a = tiny();
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.inner, &b.inner));
    }

    #[test]
    fn from_file_loads_a_real_file() {
        // Round-trip via a temp file to cover the on-disk constructor.
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!(
            "headroom-hf-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_TOKENIZER_JSON.as_bytes()).unwrap();
        drop(f);

        let t = HfTokenizer::from_file("from-file", &path).expect("loads");
        assert_eq!(t.count_text("hello world"), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
