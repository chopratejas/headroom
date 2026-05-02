//! Configuration for the proxy: CLI flags + env vars.

use clap::{Parser, ValueEnum};
use std::net::SocketAddr;
use std::time::Duration;
use url::Url;

/// Compression mode policy for the `/v1/messages` endpoint.
///
/// Drives whether `compress_anthropic_request` does any work. PR-A1
/// (Phase A lockdown) wires the flag in but both modes currently
/// passthrough — `live_zone` parses-but-warns until Phase B PR-B2
/// fills in the live-zone-only block dispatcher.
///
/// We do NOT add an `icm` mode (the deleted code path) or a
/// `passthrough` alias for `off` — those names are misleading. The
/// only legal values are `off` (compression disabled) and `live_zone`
/// (compress only the live-zone blocks; not yet implemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum CompressionMode {
    /// Compression disabled. Body forwards byte-equal to upstream.
    /// This is the default; Phase B will switch the default to
    /// `live_zone` once that mode is implemented.
    Off,
    /// Compress only live-zone blocks (latest user message,
    /// latest tool/function/shell/patch outputs). NOT YET IMPLEMENTED:
    /// in PR-A1 this falls through to passthrough behaviour with a
    /// loud warning. Phase B PR-B2 wires in the actual dispatcher.
    LiveZone,
}

/// Policy for automatically deriving `frozen_message_count` from the
/// customer's `cache_control` markers (PR-A4).
///
/// When `enabled` (default), the live-zone dispatcher will walk
/// `messages[*].content[*].cache_control` and bump the floor below
/// which compression is forbidden. When `disabled`, the floor stays
/// at 0 regardless of markers — Phase B's dispatcher will then treat
/// every message as live-zone, which is dangerous in production but
/// useful for benchmarking the cache-control machinery.
///
/// `system` and `tools[*]` markers never bump `frozen_count` because
/// those fields are *always* part of the cache hot zone (invariant I2);
/// they're guaranteed-immutable independently of marker placement.
///
/// Source priority: CLI flag → `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`
/// env var → default (`enabled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum CacheControlAutoFrozen {
    /// Walk customer `cache_control` markers and derive
    /// `frozen_message_count` automatically. Default.
    Enabled,
    /// Ignore customer `cache_control` markers when deriving
    /// `frozen_message_count`; the function returns 0 regardless of
    /// what the body contains. Intended for benchmarking and the
    /// "no automatic floor" testing path; not for production use.
    Disabled,
}

impl CacheControlAutoFrozen {
    /// Stable snake_case name suitable for log fields. Mirrors
    /// `CompressionMode::as_str` so the two policy fields render
    /// identically in JSON tracing output.
    pub fn as_str(self) -> &'static str {
        match self {
            CacheControlAutoFrozen::Enabled => "enabled",
            CacheControlAutoFrozen::Disabled => "disabled",
        }
    }

    /// Convenience: is the auto-frozen derivation switched on? Most
    /// callers want the boolean rather than pattern-matching on the
    /// enum.
    pub fn is_enabled(self) -> bool {
        matches!(self, CacheControlAutoFrozen::Enabled)
    }
}

impl CompressionMode {
    /// Stable snake_case name suitable for log fields. Avoids relying
    /// on `Debug` (which renders `Off`/`LiveZone`) or `Display`
    /// (which we don't implement to keep `ValueEnum` the single
    /// source of truth for stringification).
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionMode::Off => "off",
            CompressionMode::LiveZone => "live_zone",
        }
    }
}

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

    /// Compression mode policy for `/v1/messages`.
    ///
    /// `off` (default): byte-faithful passthrough on every request.
    /// `live_zone`: reserved for Phase B; in PR-A1 this parses-but-
    /// warns and behaves identically to `off`. The flag exists so
    /// Phase B can flip the default with one config change.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_COMPRESSION_MODE`
    /// env var → default (`off`).
    #[arg(
        long = "compression-mode",
        env = "HEADROOM_PROXY_COMPRESSION_MODE",
        value_enum,
        default_value_t = CompressionMode::Off,
    )]
    pub compression_mode: CompressionMode,

    /// Whether to derive `frozen_message_count` from customer
    /// `cache_control` markers in the request body (PR-A4).
    ///
    /// `enabled` (default): walk `messages[*].content[*].cache_control`
    /// and bump the floor for live-zone compression so any message
    /// the customer cache-pinned is left untouched. `disabled`: skip
    /// the walk; the floor stays at 0. The off switch exists for
    /// benchmark setups that want to measure compression independent
    /// of marker placement; it is NOT recommended for production.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`
    /// env var → default (`enabled`).
    #[arg(
        long = "cache-control-auto-frozen",
        env = "HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN",
        value_enum,
        default_value_t = CacheControlAutoFrozen::Enabled,
    )]
    pub cache_control_auto_frozen: CacheControlAutoFrozen,
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
    /// Policy mode for compression on `/v1/messages`. PR-A1 lockdown:
    /// both `Off` and `LiveZone` result in byte-faithful passthrough;
    /// `LiveZone` additionally emits a `tracing::warn!` per request
    /// because the dispatcher isn't implemented yet (Phase B PR-B2
    /// fills this in).
    pub compression_mode: CompressionMode,
    /// Whether the live-zone dispatcher derives `frozen_message_count`
    /// automatically from customer `cache_control` markers. PR-A4
    /// adds the derivation function (`compute_frozen_count`); Phase
    /// B's dispatcher consumes the resolved value here.
    pub cache_control_auto_frozen: CacheControlAutoFrozen,
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
            compression_mode: args.compression_mode,
            cache_control_auto_frozen: args.cache_control_auto_frozen,
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
            compression_mode: CompressionMode::Off,
            // Match production default so the cache-control walker is
            // exercised under test without per-test opt-in.
            cache_control_auto_frozen: CacheControlAutoFrozen::Enabled,
        }
    }
}
