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
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Self {
        let rewrite_host = if args.no_rewrite_host {
            false
        } else {
            args.rewrite_host
        };
        Self {
            listen: args.listen,
            upstream: args.upstream,
            upstream_timeout: args.upstream_timeout,
            upstream_connect_timeout: args.upstream_connect_timeout,
            max_body_bytes: args.max_body_bytes,
            log_level: args.log_level,
            rewrite_host,
            graceful_shutdown_timeout: args.graceful_shutdown_timeout,
        }
    }

    /// Test/library helper.
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
        }
    }
}
