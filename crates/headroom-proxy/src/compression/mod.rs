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
//! - Non-LLM paths (any URL not classified by [`classify_path`]).
//! - Non-JSON content types (skip; we don't speculate at body
//!   contents we don't know how to parse).
//! - Streaming SSE responses — only the request body is touched;
//!   responses pass through untouched.
//!
//! # Provider matrix
//!
//! | Provider     | Path                              | Status |
//! |--------------|-----------------------------------|--------|
//! | Anthropic    | `POST /v1/messages`               | ✅ |
//! | OpenAI       | `POST /v1/chat/completions`       | ✅ |
//! | OpenAI       | `POST /v1/responses` (Codex)      | follow-up |
//! | Google       | `POST /v1beta/...`                | follow-up |
//! | Bedrock      | varied                            | follow-up |
//!
//! # Failure-mode contract
//!
//! Compression must NEVER break a request. Every error path —
//! parse failure, missing field, body too large, unknown model —
//! falls through to the original body being forwarded unchanged.

pub mod anthropic;
pub mod icm;
pub mod messages;
pub mod model_limits;
pub mod openai;

pub use icm::build_icm;
pub use messages::{Outcome, PassthroughReason};

/// Which compressible LLM endpoint a request matches, if any. Used
/// by the proxy gate to decide whether to buffer and which provider
/// shape to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleEndpoint {
    /// `POST /v1/messages` (Anthropic Messages API).
    AnthropicMessages,
    /// `POST /v1/chat/completions` (OpenAI Chat Completions API,
    /// also matches OpenRouter and other proxies that present the
    /// OpenAI surface).
    OpenAIChatCompletions,
}

/// Classify a request path into a known provider endpoint, or
/// `None` when no compression should be applied. Exact-match only —
/// no prefix matching, so `/v1/messages/123` (a hypothetical
/// per-message endpoint) doesn't accidentally get treated as a chat
/// request.
pub fn classify_path(path: &str) -> Option<CompressibleEndpoint> {
    match path {
        "/v1/messages" => Some(CompressibleEndpoint::AnthropicMessages),
        "/v1/chat/completions" => Some(CompressibleEndpoint::OpenAIChatCompletions),
        _ => None,
    }
}

/// Convenience: would `classify_path(path)` return `Some(_)`? Kept
/// for callers that don't need the discriminant — primarily existing
/// telemetry/logging code paths.
pub fn is_compressible_path(path: &str) -> bool {
    classify_path(path).is_some()
}

/// Dispatch a buffered body to the right provider's compressor.
/// Returns the same provider-agnostic [`Outcome`] regardless of
/// which arm ran.
pub fn maybe_compress(
    body: &bytes::Bytes,
    icm: &headroom_core::context::IntelligentContextManager,
    endpoint: CompressibleEndpoint,
) -> Outcome {
    match endpoint {
        CompressibleEndpoint::AnthropicMessages => anthropic::maybe_compress(body, icm),
        CompressibleEndpoint::OpenAIChatCompletions => openai::maybe_compress(body, icm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_messages_path_matches() {
        assert_eq!(
            classify_path("/v1/messages"),
            Some(CompressibleEndpoint::AnthropicMessages)
        );
        assert!(is_compressible_path("/v1/messages"));
    }

    #[test]
    fn openai_chat_completions_matches() {
        assert_eq!(
            classify_path("/v1/chat/completions"),
            Some(CompressibleEndpoint::OpenAIChatCompletions)
        );
        assert!(is_compressible_path("/v1/chat/completions"));
    }

    #[test]
    fn other_paths_skip() {
        assert_eq!(classify_path("/v1/messages/123"), None);
        assert_eq!(classify_path("/v1/responses"), None); // follow-up
        assert_eq!(
            classify_path("/v1beta/models/gemini-pro:generateContent"),
            None
        );
        assert_eq!(classify_path("/healthz"), None);
        assert_eq!(classify_path("/"), None);
        assert_eq!(classify_path(""), None);
    }
}
