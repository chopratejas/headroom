//! Configuration for the proxy: CLI flags + env vars.

use clap::Parser;
use std::net::SocketAddr;
use std::time::Duration;
use url::Url;

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

    /// Enable Headroom compression on LLM-shaped requests
    /// (currently: `POST /v1/messages` for Anthropic). When off,
    /// the proxy stays a pure streaming passthrough.
    ///
    /// Off by default so existing operators get unchanged behaviour
    /// and the integration-test harness doesn't need to opt out
    /// per-test. Operators wanting to demo the compressor pass
    /// `--compression` (or set `HEADROOM_PROXY_COMPRESSION=1`).
    #[arg(
        long = "compression",
        env = "HEADROOM_PROXY_COMPRESSION",
        default_value_t = false
    )]
    pub compression: bool,

    /// Maximum body size to buffer for compression. Bodies larger
    /// than this get forwarded unchanged. Defaults to `--max-body-bytes`
    /// when unset, so operators only need to tune one knob unless
    /// they have a specific reason to cap compression separately.
    #[arg(long, value_parser = parse_bytes)]
    pub compression_max_body_bytes: Option<u64>,
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
    pub upstream_timeout: Duration,
    pub upstream_connect_timeout: Duration,
    pub max_body_bytes: u64,
    pub log_level: String,
    pub rewrite_host: bool,
    pub graceful_shutdown_timeout: Duration,
    /// Master switch for the LLM compression interceptor. When `false`,
    /// the proxy is pure streaming passthrough and never buffers a body.
    pub compression: bool,
    /// Effective ceiling for compression-time body buffering.
    /// Inherits `max_body_bytes` when not overridden. Bodies larger
    /// than this still forward, just unchanged.
    pub compression_max_body_bytes: u64,
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Self {
        let rewrite_host = if args.no_rewrite_host {
            false
        } else {
            args.rewrite_host
        };
        let compression_max_body_bytes = args
            .compression_max_body_bytes
            .unwrap_or(args.max_body_bytes);
        Self {
            listen: args.listen,
            upstream: args.upstream,
            upstream_timeout: args.upstream_timeout,
            upstream_connect_timeout: args.upstream_connect_timeout,
            max_body_bytes: args.max_body_bytes,
            log_level: args.log_level,
            rewrite_host,
            graceful_shutdown_timeout: args.graceful_shutdown_timeout,
            compression: args.compression,
            compression_max_body_bytes,
        }
    }

    /// Test/library helper. Compression off by default — match
    /// production-default behaviour so existing tests stay unchanged.
    pub fn for_test(upstream: Url) -> Self {
        Self {
            listen: "127.0.0.1:0".parse().unwrap(),
            upstream,
            upstream_timeout: Duration::from_secs(60),
            upstream_connect_timeout: Duration::from_secs(5),
            max_body_bytes: 100 * 1024 * 1024,
            log_level: "warn".into(),
            rewrite_host: true,
            graceful_shutdown_timeout: Duration::from_secs(5),
            compression: false,
            compression_max_body_bytes: 100 * 1024 * 1024,
        }
    }
}
