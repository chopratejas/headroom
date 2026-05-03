//! Compression interceptor for LLM-shaped requests.
//!
//! # Phase A lockdown (PR-A1)
//!
//! Per `REALIGNMENT/03-phase-A-lockdown.md`, the
//! `IntelligentContextManager`-driven path that previously ran on
//! every `/v1/messages` request is gone. Today this module is a
//! tracking shell: it owns the path-matcher (`is_compressible_path`)
//! and the Anthropic decision stub (`compress_anthropic_request`)
//! that always returns `Outcome::NoCompression`.
//!
//! Phase B PR-B2 reintroduces real compression, but with two
//! invariants the deleted code violated:
//!
//! 1. The cache hot zone (system, tools, historical messages,
//!    reasoning items, thinking signatures, redacted_thinking,
//!    compaction items) is never modified.
//! 2. Compression is append-only: only the live zone is rewritten.
//!
//! # Provider matrix (current + planned)
//!
//! | Provider     | Path                  | Status |
//! |--------------|-----------------------|--------|
//! | Anthropic    | `POST /v1/messages`   | passthrough (PR-A1) → live-zone (PR-B2) |
//! | OpenAI       | `POST /v1/chat/completions` | follow-up |
//! | Google       | `POST /v1beta/...`    | follow-up |
//! | Bedrock      | varied                | follow-up |
//!
//! # Failure-mode contract
//!
//! Compression must NEVER break a request. Even when Phase B brings
//! a real dispatcher back, every error path falls through to the
//! original body being forwarded unchanged.

pub mod anthropic;
pub mod live_zone_anthropic;
pub mod model_limits;

// PR-A4 helper for cache-control floor derivation lives on the
// passthrough-stub module so PR-B2's live-zone dispatcher can call
// it without dragging in the rest of `anthropic.rs`. The stub
// itself stays through B1 → B2 transition for parallel review;
// `compress_anthropic_request` is sourced from the live-zone module.
pub use anthropic::resolve_frozen_count;
pub use live_zone_anthropic::{compress_anthropic_request, Outcome, PassthroughReason};

/// Does this request path target an LLM endpoint we know how to
/// compress? Cheap pre-filter before buffering the body. Phase B
/// reuses this to gate which paths get the live-zone dispatcher.
pub fn is_compressible_path(path: &str) -> bool {
    // Exact-match the Anthropic Messages endpoint. Future providers
    // get their own arms here. Avoid prefix-matching to keep the
    // compression scope explicit — `/v1/messages/123` (a
    // hypothetical future per-message endpoint) shouldn't accidentally
    // get its body parsed as a chat-completions request.
    path == "/v1/messages"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_messages_path_matches() {
        assert!(is_compressible_path("/v1/messages"));
    }

    #[test]
    fn other_paths_skip() {
        assert!(!is_compressible_path("/v1/messages/123"));
        assert!(!is_compressible_path("/v1/chat/completions"));
        assert!(!is_compressible_path("/healthz"));
        assert!(!is_compressible_path("/"));
        assert!(!is_compressible_path(""));
    }
}
