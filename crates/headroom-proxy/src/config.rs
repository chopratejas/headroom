//! Configuration for the proxy: CLI flags + env vars.

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use url::Url;

const DEFAULT_OPENAI_API_URL: &str = "https://api.openai.com";
const DEFAULT_ANTHROPIC_API_URL: &str = "https://api.anthropic.com";
const DEFAULT_GEMINI_API_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_CLOUDCODE_API_URL: &str = "https://cloudcode-pa.googleapis.com";
pub const MAX_REQUEST_ARRAY_LENGTH: usize = 10_000;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "headroom-proxy",
    version,
    about = "Headroom transparent reverse proxy"
)]
pub struct CliArgs {
    /// Address the proxy listens on (e.g. 0.0.0.0:8787).
    #[arg(long, env = "HEADROOM_PROXY_LISTEN", default_value = "0.0.0.0:8787")]
    pub listen: SocketAddr,

    /// Upstream base URL the proxy forwards to (e.g. http://127.0.0.1:8788).
    /// REQUIRED — there is no default; we want operators to be explicit.
    #[arg(long, env = "HEADROOM_PROXY_UPSTREAM")]
    pub upstream: Url,

    /// Custom OpenAI API URL for native OpenAI execution (env: OPENAI_TARGET_API_URL).
    #[arg(long, env = "OPENAI_TARGET_API_URL")]
    pub openai_api_url: Option<Url>,

    /// Custom Anthropic API URL for native Anthropic execution (env: ANTHROPIC_TARGET_API_URL).
    #[arg(long, env = "ANTHROPIC_TARGET_API_URL")]
    pub anthropic_api_url: Option<Url>,

    /// Custom Gemini API URL for native Gemini execution (env: GEMINI_TARGET_API_URL).
    #[arg(long, env = "GEMINI_TARGET_API_URL")]
    pub gemini_api_url: Option<Url>,

    /// Custom Databricks API URL for native Databricks execution (env: DATABRICKS_TARGET_API_URL).
    #[arg(long, env = "DATABRICKS_TARGET_API_URL")]
    pub databricks_api_url: Option<Url>,

    /// Custom Cloud Code API URL for native Cloud Code execution (env: CLOUDCODE_TARGET_API_URL).
    #[arg(long, env = "CLOUDCODE_TARGET_API_URL")]
    pub cloudcode_api_url: Option<Url>,

    /// Enable request-level response caching for eligible native buffered routes.
    #[arg(
        long,
        env = "HEADROOM_RESPONSE_CACHE_ENABLED",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub response_cache_enabled: bool,

    /// Maximum response-cache entries retained in-memory.
    #[arg(
        long,
        env = "HEADROOM_RESPONSE_CACHE_MAX_ENTRIES",
        default_value_t = 1024
    )]
    pub response_cache_max_entries: usize,

    /// TTL for cached native responses.
    #[arg(
        long,
        env = "HEADROOM_RESPONSE_CACHE_TTL",
        default_value = "300s",
        value_parser = parse_duration
    )]
    pub response_cache_ttl: Duration,

    /// End-to-end timeout for a single upstream request (long, since LLM
    /// streams may run for many minutes).
    #[arg(long, default_value = "600s", value_parser = parse_duration)]
    pub upstream_timeout: Duration,

    /// TCP/TLS connect timeout for upstream.
    #[arg(long, default_value = "10s", value_parser = parse_duration)]
    pub upstream_connect_timeout: Duration,

    /// Max body size for buffered cases (does NOT bound streaming bodies).
    #[arg(long, default_value = "100MB", value_parser = parse_bytes)]
    pub max_body_bytes: u64,

    /// Log level / filter (RUST_LOG-style). Default: info.
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Rewrite the outgoing Host header to the upstream host (default).
    /// Pair with --no-rewrite-host to preserve the client-supplied Host.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub rewrite_host: bool,

    /// Convenience flag matching the spec; sets rewrite_host=false when present.
    #[arg(long = "no-rewrite-host", default_value_t = false)]
    pub no_rewrite_host: bool,

    /// Maximum time to wait for in-flight requests to finish on shutdown.
    #[arg(long, default_value = "30s", value_parser = parse_duration)]
    pub graceful_shutdown_timeout: Duration,

    /// Enable the first native Rust route for `/v1/chat/completions`.
    /// Disabled by default so the proxy stays passthrough-first until cutover.
    #[arg(long, env = "HEADROOM_NATIVE_OPENAI_CHAT", default_value_t = false)]
    pub native_openai_chat: bool,

    /// Run non-streaming native OpenAI chat requests in shadow comparison mode
    /// against the passthrough path. Requires `--native-openai-chat`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_OPENAI_CHAT_SHADOW",
        default_value_t = false
    )]
    pub native_openai_chat_shadow: bool,

    /// Enable the first native Rust route for `/v1/messages`.
    /// Disabled by default so Anthropic traffic stays passthrough-first.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_ANTHROPIC_MESSAGES",
        default_value_t = false
    )]
    pub native_anthropic_messages: bool,

    /// Run non-streaming native Anthropic messages requests in shadow
    /// comparison mode against the passthrough path. Requires
    /// `--native-anthropic-messages`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_ANTHROPIC_MESSAGES_SHADOW",
        default_value_t = false
    )]
    pub native_anthropic_messages_shadow: bool,

    /// Enable the native Rust route for Anthropic `/v1/messages/count_tokens`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_ANTHROPIC_COUNT_TOKENS",
        default_value_t = false
    )]
    pub native_anthropic_count_tokens: bool,

    /// Enable the first native Rust route for Gemini
    /// `/v1beta/models/{model}:generateContent`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GEMINI_GENERATE_CONTENT",
        default_value_t = false
    )]
    pub native_gemini_generate_content: bool,

    /// Run native Gemini generateContent requests in shadow comparison mode
    /// against the passthrough path. Requires `--native-gemini-generate-content`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GEMINI_GENERATE_CONTENT_SHADOW",
        default_value_t = false
    )]
    pub native_gemini_generate_content_shadow: bool,

    /// Enable the first native Rust route for Gemini
    /// `/v1beta/models/{model}:countTokens`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GEMINI_COUNT_TOKENS",
        default_value_t = false
    )]
    pub native_gemini_count_tokens: bool,

    /// Run native Gemini countTokens requests in shadow comparison mode
    /// against the passthrough path. Requires `--native-gemini-count-tokens`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GEMINI_COUNT_TOKENS_SHADOW",
        default_value_t = false
    )]
    pub native_gemini_count_tokens_shadow: bool,

    /// Enable the native Rust route for Google Cloud Code Assist streaming
    /// aliases `/v1internal:streamGenerateContent` and
    /// `/v1/v1internal:streamGenerateContent`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GOOGLE_CLOUDCODE_STREAM",
        default_value_t = false
    )]
    pub native_google_cloudcode_stream: bool,

    /// Run native Google Cloud Code Assist streaming alias requests in shadow
    /// comparison mode against the passthrough path. Requires
    /// `--native-google-cloudcode-stream`.
    #[arg(
        long,
        env = "HEADROOM_NATIVE_GOOGLE_CLOUDCODE_STREAM_SHADOW",
        default_value_t = false
    )]
    pub native_google_cloudcode_stream_shadow: bool,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("invalid duration `{s}`: {e}"))
}

fn parse_bytes(s: &str) -> Result<u64, String> {
    s.parse::<bytesize::ByteSize>()
        .map(|b| b.as_u64())
        .map_err(|e| format!("invalid byte size `{s}`: {e}"))
}

/// Resolved configuration used by the running server.
#[derive(Debug, Clone)]
pub struct Config {
    pub listen: SocketAddr,
    pub upstream: Url,
    pub openai_api_url: Url,
    pub anthropic_api_url: Url,
    pub gemini_api_url: Url,
    pub databricks_api_url: Url,
    pub cloudcode_api_url: Url,
    pub response_cache_enabled: bool,
    pub response_cache_max_entries: usize,
    pub response_cache_ttl: Duration,
    pub upstream_timeout: Duration,
    pub upstream_connect_timeout: Duration,
    pub max_body_bytes: u64,
    pub log_level: String,
    pub rewrite_host: bool,
    pub graceful_shutdown_timeout: Duration,
    pub savings_path: Option<PathBuf>,
    pub native_openai_chat: bool,
    pub native_openai_chat_shadow: bool,
    pub native_anthropic_messages: bool,
    pub native_anthropic_messages_shadow: bool,
    pub native_anthropic_count_tokens: bool,
    pub native_gemini_generate_content: bool,
    pub native_gemini_generate_content_shadow: bool,
    pub native_gemini_count_tokens: bool,
    pub native_gemini_count_tokens_shadow: bool,
    pub native_google_cloudcode_stream: bool,
    pub native_google_cloudcode_stream_shadow: bool,
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Self {
        let rewrite_host = if args.no_rewrite_host {
            false
        } else {
            args.rewrite_host
        };
        let openai_api_url =
            normalize_api_url(args.openai_api_url.unwrap_or_else(default_openai_api_url));
        let anthropic_api_url = normalize_api_url(
            args.anthropic_api_url
                .unwrap_or_else(default_anthropic_api_url),
        );
        let gemini_api_url =
            normalize_api_url(args.gemini_api_url.unwrap_or_else(default_gemini_api_url));
        let databricks_api_url = normalize_api_url(
            args.databricks_api_url
                .unwrap_or_else(|| args.upstream.clone()),
        );
        let cloudcode_api_url = normalize_api_url(
            args.cloudcode_api_url
                .unwrap_or_else(default_cloudcode_api_url),
        );
        Self {
            listen: args.listen,
            upstream: args.upstream,
            openai_api_url,
            anthropic_api_url,
            gemini_api_url,
            databricks_api_url,
            cloudcode_api_url,
            response_cache_enabled: args.response_cache_enabled,
            response_cache_max_entries: args.response_cache_max_entries,
            response_cache_ttl: args.response_cache_ttl,
            upstream_timeout: args.upstream_timeout,
            upstream_connect_timeout: args.upstream_connect_timeout,
            max_body_bytes: args.max_body_bytes,
            log_level: args.log_level,
            rewrite_host,
            graceful_shutdown_timeout: args.graceful_shutdown_timeout,
            savings_path: Some(default_savings_path()),
            native_openai_chat: args.native_openai_chat,
            native_openai_chat_shadow: args.native_openai_chat && args.native_openai_chat_shadow,
            native_anthropic_messages: args.native_anthropic_messages,
            native_anthropic_messages_shadow: args.native_anthropic_messages
                && args.native_anthropic_messages_shadow,
            native_anthropic_count_tokens: args.native_anthropic_count_tokens,
            native_gemini_generate_content: args.native_gemini_generate_content,
            native_gemini_generate_content_shadow: args.native_gemini_generate_content
                && args.native_gemini_generate_content_shadow,
            native_gemini_count_tokens: args.native_gemini_count_tokens,
            native_gemini_count_tokens_shadow: args.native_gemini_count_tokens
                && args.native_gemini_count_tokens_shadow,
            native_google_cloudcode_stream: args.native_google_cloudcode_stream,
            native_google_cloudcode_stream_shadow: args.native_google_cloudcode_stream
                && args.native_google_cloudcode_stream_shadow,
        }
    }

    /// Test/library helper.
    pub fn for_test(upstream: Url) -> Self {
        Self {
            listen: "127.0.0.1:0".parse().unwrap(),
            openai_api_url: upstream.clone(),
            anthropic_api_url: upstream.clone(),
            gemini_api_url: upstream.clone(),
            databricks_api_url: upstream.clone(),
            cloudcode_api_url: upstream.clone(),
            response_cache_enabled: true,
            response_cache_max_entries: 1024,
            response_cache_ttl: Duration::from_secs(300),
            upstream,
            upstream_timeout: Duration::from_secs(60),
            upstream_connect_timeout: Duration::from_secs(5),
            max_body_bytes: 100 * 1024 * 1024,
            log_level: "warn".into(),
            rewrite_host: true,
            graceful_shutdown_timeout: Duration::from_secs(5),
            savings_path: None,
            native_openai_chat: false,
            native_openai_chat_shadow: false,
            native_anthropic_messages: false,
            native_anthropic_messages_shadow: false,
            native_anthropic_count_tokens: false,
            native_gemini_generate_content: false,
            native_gemini_generate_content_shadow: false,
            native_gemini_count_tokens: false,
            native_gemini_count_tokens_shadow: false,
            native_google_cloudcode_stream: false,
            native_google_cloudcode_stream_shadow: false,
        }
    }
}

fn default_savings_path() -> PathBuf {
    let configured = std::env::var("HEADROOM_SAVINGS_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(path) = configured {
        return PathBuf::from(path);
    }
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".headroom")
        .join("proxy_savings.json")
}

fn default_openai_api_url() -> Url {
    Url::parse(DEFAULT_OPENAI_API_URL).expect("default OpenAI API URL must be valid")
}

fn default_anthropic_api_url() -> Url {
    Url::parse(DEFAULT_ANTHROPIC_API_URL).expect("default Anthropic API URL must be valid")
}

fn default_gemini_api_url() -> Url {
    Url::parse(DEFAULT_GEMINI_API_URL).expect("default Gemini API URL must be valid")
}

fn default_cloudcode_api_url() -> Url {
    Url::parse(DEFAULT_CLOUDCODE_API_URL).expect("default Cloud Code API URL must be valid")
}

fn normalize_api_url(mut url: Url) -> Url {
    let trimmed = url.path().trim_end_matches('/').to_string();
    let normalized = if trimmed == "/v1" {
        ""
    } else {
        trimmed.as_str()
    };
    url.set_path(normalized);
    url.set_query(None);
    url.set_fragment(None);
    url
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_openai_chat_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_openai_chat);
    }

    #[test]
    fn native_openai_chat_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-openai-chat",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_openai_chat);
    }

    #[test]
    fn openai_api_url_defaults_to_openai() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert_eq!(config.openai_api_url.as_str(), "https://api.openai.com/");
    }

    #[test]
    fn openai_api_url_normalizes_v1_suffix() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--openai-api-url",
            "https://example.test/v1/",
        ]);

        let config = Config::from_cli(args);

        assert_eq!(config.openai_api_url.as_str(), "https://example.test/");
    }

    #[test]
    fn anthropic_api_url_defaults_to_anthropic() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert_eq!(
            config.anthropic_api_url.as_str(),
            "https://api.anthropic.com/"
        );
    }

    #[test]
    fn anthropic_api_url_normalizes_v1_suffix() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--anthropic-api-url",
            "https://example.test/v1/",
        ]);

        let config = Config::from_cli(args);

        assert_eq!(config.anthropic_api_url.as_str(), "https://example.test/");
    }

    #[test]
    fn gemini_api_url_defaults_to_google() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert_eq!(
            config.gemini_api_url.as_str(),
            "https://generativelanguage.googleapis.com/"
        );
    }

    #[test]
    fn gemini_api_url_normalizes_v1_suffix() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--gemini-api-url",
            "https://example.test/v1/",
        ]);

        let config = Config::from_cli(args);

        assert_eq!(config.gemini_api_url.as_str(), "https://example.test/");
    }

    #[test]
    fn databricks_api_url_defaults_to_upstream() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert_eq!(config.databricks_api_url.as_str(), "http://127.0.0.1:8788/");
    }

    #[test]
    fn databricks_api_url_normalizes_v1_suffix() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--databricks-api-url",
            "https://example.test/v1/",
        ]);

        let config = Config::from_cli(args);

        assert_eq!(config.databricks_api_url.as_str(), "https://example.test/");
    }

    #[test]
    fn cloudcode_api_url_defaults_to_google() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert_eq!(
            config.cloudcode_api_url.as_str(),
            "https://cloudcode-pa.googleapis.com/"
        );
    }

    #[test]
    fn cloudcode_api_url_normalizes_v1_suffix() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--cloudcode-api-url",
            "https://example.test/v1/",
        ]);

        let config = Config::from_cli(args);

        assert_eq!(config.cloudcode_api_url.as_str(), "https://example.test/");
    }

    #[test]
    fn native_openai_chat_shadow_requires_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-openai-chat-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(!config.native_openai_chat);
        assert!(!config.native_openai_chat_shadow);
    }

    #[test]
    fn native_openai_chat_shadow_can_be_enabled_with_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-openai-chat",
            "--native-openai-chat-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_openai_chat);
        assert!(config.native_openai_chat_shadow);
    }

    #[test]
    fn native_anthropic_messages_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_anthropic_messages);
    }

    #[test]
    fn native_anthropic_messages_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-anthropic-messages",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_anthropic_messages);
    }

    #[test]
    fn native_anthropic_messages_shadow_requires_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-anthropic-messages-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(!config.native_anthropic_messages);
        assert!(!config.native_anthropic_messages_shadow);
    }

    #[test]
    fn native_anthropic_messages_shadow_can_be_enabled_with_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-anthropic-messages",
            "--native-anthropic-messages-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_anthropic_messages);
        assert!(config.native_anthropic_messages_shadow);
    }

    #[test]
    fn native_anthropic_count_tokens_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_anthropic_count_tokens);
    }

    #[test]
    fn native_anthropic_count_tokens_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-anthropic-count-tokens",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_anthropic_count_tokens);
    }

    #[test]
    fn native_gemini_generate_content_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_gemini_generate_content);
    }

    #[test]
    fn native_gemini_generate_content_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-generate-content",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_gemini_generate_content);
    }

    #[test]
    fn native_gemini_generate_content_shadow_requires_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-generate-content-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(!config.native_gemini_generate_content);
        assert!(!config.native_gemini_generate_content_shadow);
    }

    #[test]
    fn native_gemini_generate_content_shadow_can_be_enabled_with_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-generate-content",
            "--native-gemini-generate-content-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_gemini_generate_content);
        assert!(config.native_gemini_generate_content_shadow);
    }

    #[test]
    fn native_gemini_count_tokens_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_gemini_count_tokens);
    }

    #[test]
    fn native_gemini_count_tokens_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-count-tokens",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_gemini_count_tokens);
    }

    #[test]
    fn native_gemini_count_tokens_shadow_requires_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-count-tokens-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(!config.native_gemini_count_tokens);
        assert!(!config.native_gemini_count_tokens_shadow);
    }

    #[test]
    fn native_gemini_count_tokens_shadow_can_be_enabled_with_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-gemini-count-tokens",
            "--native-gemini-count-tokens-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_gemini_count_tokens);
        assert!(config.native_gemini_count_tokens_shadow);
    }

    #[test]
    fn native_google_cloudcode_stream_defaults_to_disabled() {
        let args = CliArgs::parse_from(["headroom-proxy", "--upstream", "http://127.0.0.1:8788"]);

        let config = Config::from_cli(args);

        assert!(!config.native_google_cloudcode_stream);
    }

    #[test]
    fn native_google_cloudcode_stream_can_be_enabled_explicitly() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-google-cloudcode-stream",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_google_cloudcode_stream);
    }

    #[test]
    fn native_google_cloudcode_stream_shadow_requires_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-google-cloudcode-stream-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(!config.native_google_cloudcode_stream);
        assert!(!config.native_google_cloudcode_stream_shadow);
    }

    #[test]
    fn native_google_cloudcode_stream_shadow_can_be_enabled_with_native_route() {
        let args = CliArgs::parse_from([
            "headroom-proxy",
            "--upstream",
            "http://127.0.0.1:8788",
            "--native-google-cloudcode-stream",
            "--native-google-cloudcode-stream-shadow",
        ]);

        let config = Config::from_cli(args);

        assert!(config.native_google_cloudcode_stream);
        assert!(config.native_google_cloudcode_stream_shadow);
    }
}
