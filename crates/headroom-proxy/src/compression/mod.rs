//! Compression interceptor for LLM-shaped requests.
//!
//! The proxy is a streaming reverse proxy by default. When
//! `--compression` is enabled and a request hits a known LLM
//! provider path, we buffer the body, run
//! `IntelligentContextManager` over the message list, and forward
//! the (possibly trimmed) body upstream. Everything else stays
//! streaming, including:
//!
//! - WebSocket upgrades — handled in the catch-all before
//!   `forward_http` is called; never reach this module.
//! - Non-LLM paths (any URL not matching a known provider).
//! - Non-JSON content types (skip; we don't speculate at body
//!   contents we don't know how to parse).
//! - Streaming SSE responses — only the request body is touched;
//!   responses pass through untouched.
//!
//! # Provider matrix (current + planned)
//!
//! | Provider     | Path                  | Status |
//! |--------------|-----------------------|--------|
//! | Anthropic    | `POST /v1/messages`   | ✅ this module |
//! | OpenAI       | `POST /v1/chat/completions` | follow-up |
//! | Google       | `POST /v1beta/...`    | follow-up |
//! | Bedrock      | varied                | follow-up |
//!
//! # Failure-mode contract
//!
//! Compression must NEVER break a request. Every error path —
//! parse failure, missing field, body too large, unknown model —
//! falls through to the original body being forwarded unchanged.
//! Operators see what happened in `tracing` warnings; clients see
//! their request go through.

pub mod anthropic;
pub mod icm;
pub mod model_limits;

pub use anthropic::{maybe_compress, Outcome, PassthroughReason};
pub use icm::build_icm;

/// Does this request path target an LLM endpoint we know how to
/// compress? Cheap pre-filter before buffering the body.
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
