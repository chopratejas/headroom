//! Compression interceptor for LLM-shaped requests — **token mode**.
//!
//! # The single rule
//!
//! Find the last user message. **Freeze every byte before it**
//! exactly as the client sent it. In the new turn (last-user-message
//! onward), compress only the text content of tool outputs:
//!
//! - Anthropic `/v1/messages`: `tool_result` blocks inside user messages
//! - OpenAI `/v1/chat/completions`: messages with `role:tool`
//! - OpenAI `/v1/responses`: `function_call_output` items
//!
//! User text, system prompts, assistant text, structured tool calls,
//! reasoning items, images/audio/files, and every prefix message
//! pass through byte-identical.
//!
//! # Why
//!
//! This is the only design that gives prefix cache + accuracy at
//! the same time:
//!
//! - Provider prefix caches (Anthropic `cache_control`, OpenAI
//!   automatic 1024-token prefix, OpenAI `prompt_cache_key`) are
//!   *positional*. Anything before the cut point that changes byte
//!   shape busts the cache. We never change the prefix.
//! - User input is sacred — paraphrasing or shortening it changes
//!   the meaning of the question. Same for system prompts.
//! - The big tokens in agent traffic are tool outputs: file reads,
//!   search results, git diffs, build logs. Those are precisely
//!   what the [`CompressionPipeline`] is built to compress, with
//!   CCR backup so the model can retrieve the original if needed.
//!
//! [`CompressionPipeline`]: headroom_core::transforms::pipeline::CompressionPipeline
//!
//! # Provider matrix
//!
//! | Provider     | Path                              | Status |
//! |--------------|-----------------------------------|--------|
//! | Anthropic    | `POST /v1/messages`               | ✅ |
//! | OpenAI       | `POST /v1/chat/completions`       | ✅ |
//! | OpenAI       | `POST /v1/responses`              | ✅ |
//! | Google       | `POST /v1beta/models/...`         | follow-up |
//! | Bedrock      | varied                            | follow-up |
//!
//! # Failure-mode contract
//!
//! Compression must NEVER break a request. Every error path —
//! parse failure, missing field, body too large — falls through to
//! the original body being forwarded unchanged.

pub mod anthropic;
pub mod model_limits;
pub mod openai;
pub mod pipeline;
pub mod responses;
pub mod walker;

pub use pipeline::build_pipeline;

use bytes::Bytes;
use std::sync::Arc;

use headroom_core::ccr::CcrStore;
use headroom_core::transforms::pipeline::CompressionPipeline;

/// Which compressible LLM endpoint a request matches, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleEndpoint {
    /// `POST /v1/messages` (Anthropic Messages API).
    AnthropicMessages,
    /// `POST /v1/chat/completions` (OpenAI Chat Completions).
    OpenAIChatCompletions,
    /// `POST /v1/responses` (OpenAI Responses / Codex API).
    OpenAIResponses,
}

/// Classify a request path. Exact-match only — no prefixing.
pub fn classify_path(path: &str) -> Option<CompressibleEndpoint> {
    match path {
        "/v1/messages" => Some(CompressibleEndpoint::AnthropicMessages),
        "/v1/chat/completions" => Some(CompressibleEndpoint::OpenAIChatCompletions),
        "/v1/responses" => Some(CompressibleEndpoint::OpenAIResponses),
        _ => None,
    }
}

/// Convenience: would `classify_path(path)` return `Some(_)`?
pub fn is_compressible_path(path: &str) -> bool {
    classify_path(path).is_some()
}

/// What happened. Logged at request level.
#[derive(Debug)]
pub enum Outcome {
    /// Body was unchanged; send original buffered bytes.
    Passthrough { reason: PassthroughReason },
    /// Compressed body to forward in place of the original.
    Compressed {
        body: Bytes,
        bytes_before: usize,
        bytes_after: usize,
        steps_applied: Vec<String>,
    },
}

/// Why the body was passed through unchanged.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// JSON parse failed — body wasn't JSON or was malformed.
    NotJson,
    /// The relevant field (`messages` / `input`) wasn't present
    /// or wasn't the expected shape.
    NoMessages,
    /// `previous_response_id` is set on /v1/responses — the server
    /// holds the conversation, so this body is just one new turn
    /// and isn't a useful compression target.
    StatefulMode,
    /// Last user turn couldn't be found (e.g. only assistant or
    /// tool messages). Conservative skip.
    NoUserAnchor,
    /// Walked the new turn but found nothing eligible to compress.
    NothingToCompress,
    /// Re-serialization of the modified body failed (defense in depth).
    SerializeFailed,
}

/// Dispatch a buffered body to the right provider's token-mode walker.
pub fn maybe_compress(
    body: &Bytes,
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    endpoint: CompressibleEndpoint,
) -> Outcome {
    match endpoint {
        CompressibleEndpoint::AnthropicMessages => anthropic::maybe_compress(body, pipeline, store),
        CompressibleEndpoint::OpenAIChatCompletions => {
            openai::maybe_compress(body, pipeline, store)
        }
        CompressibleEndpoint::OpenAIResponses => responses::maybe_compress(body, pipeline, store),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_path_matches() {
        assert_eq!(
            classify_path("/v1/messages"),
            Some(CompressibleEndpoint::AnthropicMessages)
        );
    }

    #[test]
    fn openai_chat_path_matches() {
        assert_eq!(
            classify_path("/v1/chat/completions"),
            Some(CompressibleEndpoint::OpenAIChatCompletions)
        );
    }

    #[test]
    fn openai_responses_path_matches() {
        assert_eq!(
            classify_path("/v1/responses"),
            Some(CompressibleEndpoint::OpenAIResponses)
        );
    }

    #[test]
    fn other_paths_skip() {
        assert_eq!(classify_path("/v1/messages/123"), None);
        assert_eq!(classify_path("/healthz"), None);
        assert_eq!(classify_path("/"), None);
        assert_eq!(classify_path(""), None);
    }
}
