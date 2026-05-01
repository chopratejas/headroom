//! Core reverse-proxy router and HTTP forwarding handler.

use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::extract::{ConnectInfo, Path, State, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::Router;
use base64::engine::general_purpose::URL_SAFE;
use base64::Engine as _;
use bytes::Bytes;
use futures_util::{Stream, StreamExt as _, TryStreamExt};
use headroom_core::tokenizer::{get_tokenizer, Tokenizer};
use headroom_core::transforms::{
    smart_crusher::SmartCrusher, DiffCompressor, DiffCompressorConfig, LogCompressor,
    LogCompressorConfig, SearchCompressor, SearchCompressorConfig, TextCompressor,
    TextCompressorConfig,
};
use headroom_runtime::{ExecutionContext, PipelineDispatcher, PipelineMetadata, PipelineStage};
use http_body_util::BodyExt;
use serde_json::{Map, Value};
use tokio::sync::oneshot;

use crate::config::Config;
use crate::config::MAX_REQUEST_ARRAY_LENGTH;
use crate::error::ProxyError;
use crate::headers::{build_forward_request_headers, filter_response_headers};
use crate::health::{health, healthz, healthz_upstream, livez, readyz};
use crate::metrics::{metrics, ProxyMetrics};
use crate::product_store::ProductStore;
use crate::request_log_store::RequestLogStore;
use crate::state_store::{file_backend, sqlite_backend, uses_sqlite_backend};
use crate::telemetry_store::{telemetry_enabled, TelemetryStore};
use crate::websocket::{ws_handler, WebSocketSessionRegistry};

const ANTIGRAVITY_DAILY_API_URL: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const DASHBOARD_HTML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../headroom/dashboard/templates/dashboard.html"
));
const DASHBOARD_STATS_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

struct BufferedForwardResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    ttfb_ms: f64,
}

struct StreamingForwardResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: ResponseByteStream,
    ttfb_ms: f64,
}

type ResponseByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[derive(Clone, Copy)]
enum ShadowProvider {
    OpenAi,
    Anthropic,
    Gemini,
    GoogleCloudCodeStream,
}

/// Shared state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: reqwest::Client,
    pub metrics: Arc<ProxyMetrics>,
    pub runtime: Arc<PipelineDispatcher>,
    pub product: ProductStore,
    pub telemetry: TelemetryStore,
    request_logs: RequestLogStore,
    compression_cache: Arc<Mutex<CompressionCache>>,
    response_cache: Arc<Mutex<ResponseCache>>,
    stats_snapshot: Arc<Mutex<Option<StatsSnapshot>>>,
    pub ws_sessions: WebSocketSessionRegistry,
    pub started_at: Instant,
}

#[derive(Clone)]
struct StatsSnapshot {
    expires_at: Instant,
    payload: Value,
}

impl AppState {
    pub fn new(config: Config) -> Result<Self, ProxyError> {
        Self::new_with_runtime(config, Arc::new(PipelineDispatcher::new()))
    }

    pub fn new_with_runtime(
        config: Config,
        runtime: Arc<PipelineDispatcher>,
    ) -> Result<Self, ProxyError> {
        let response_cache_ttl_seconds = config.response_cache_ttl.as_secs();
        let response_cache_max_entries = config.response_cache_max_entries;
        let savings_path = config.savings_path.clone();
        let request_log_backend = savings_path
            .clone()
            .map(|path| build_state_backend(path, "request_log", None));
        let product_backend = savings_path
            .clone()
            .map(|path| build_state_backend(path, "product", Some(derive_product_store_path)));
        let telemetry_backend = savings_path
            .map(|path| build_state_backend(path, "telemetry", Some(derive_telemetry_store_path)));
        let client = reqwest::Client::builder()
            .connect_timeout(config.upstream_connect_timeout)
            .timeout(config.upstream_timeout)
            // Don't auto-follow redirects: pass them through verbatim.
            .redirect(reqwest::redirect::Policy::none())
            // Pool needs to be allowed to be idle for long-lived streams.
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            // Both HTTP/1.1 and HTTP/2 negotiated via ALPN.
            .build()
            .map_err(ProxyError::Upstream)?;
        Ok(Self {
            config: Arc::new(config),
            client,
            metrics: Arc::new(ProxyMetrics::default()),
            runtime,
            product: ProductStore::new_with_backend(product_backend),
            telemetry: TelemetryStore::new_with_backend(telemetry_backend),
            request_logs: RequestLogStore::new_with_backend(request_log_backend),
            compression_cache: Arc::new(Mutex::new(CompressionCache::default())),
            response_cache: Arc::new(Mutex::new(ResponseCache::new(
                response_cache_ttl_seconds,
                response_cache_max_entries,
            ))),
            stats_snapshot: Arc::new(Mutex::new(None)),
            ws_sessions: WebSocketSessionRegistry::default(),
            started_at: Instant::now(),
        })
    }

    pub fn local_state_backends(&self) -> Value {
        serde_json::json!({
            "request_log": self.request_logs.backend_info(),
            "product": self.product.backend_info(),
            "telemetry": self.telemetry.backend_info(),
        })
    }
}

fn derive_product_store_path(path: &std::path::PathBuf) -> std::path::PathBuf {
    path.with_extension("product.json")
}

fn derive_telemetry_store_path(path: &std::path::PathBuf) -> std::path::PathBuf {
    path.with_extension("telemetry.json")
}

fn build_state_backend(
    path: std::path::PathBuf,
    sqlite_key: &'static str,
    file_path_mapper: Option<fn(&std::path::PathBuf) -> std::path::PathBuf>,
) -> crate::state_store::SharedStateBackend {
    if uses_sqlite_backend(&path) {
        sqlite_backend(path, sqlite_key)
    } else {
        let storage_path = file_path_mapper.map(|mapper| mapper(&path)).unwrap_or(path);
        file_backend(storage_path)
    }
}

#[derive(Debug, Clone)]
struct CachedCompression {
    compressed: String,
    compression_ratio: f64,
    strategy: String,
    created_at: Instant,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CompressionCacheStats {
    cache_hits: usize,
    cache_misses: usize,
    cache_skip_hits: usize,
    cache_evictions: usize,
    cache_size: usize,
    cache_skip_size: usize,
}

#[derive(Debug)]
struct CompressionCache {
    results: HashMap<u64, CachedCompression>,
    skips: HashMap<u64, Instant>,
    cache_hits: usize,
    cache_misses: usize,
    cache_skip_hits: usize,
    cache_evictions: usize,
    ttl_seconds: u64,
}

impl Default for CompressionCache {
    fn default() -> Self {
        Self {
            results: HashMap::new(),
            skips: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            cache_skip_hits: 0,
            cache_evictions: 0,
            ttl_seconds: 1800,
        }
    }
}

#[derive(Debug, Clone)]
struct CachedResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
    created_at: Instant,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResponseCacheStats {
    cache_hits: usize,
    cache_misses: usize,
    cache_evictions: usize,
    cache_size: usize,
}

#[derive(Debug)]
struct ResponseCache {
    entries: HashMap<u64, CachedResponse>,
    order: VecDeque<u64>,
    cache_hits: usize,
    cache_misses: usize,
    cache_evictions: usize,
    ttl_seconds: u64,
    max_entries: usize,
}

impl ResponseCache {
    fn new(ttl_seconds: u64, max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            ttl_seconds,
            max_entries,
        }
    }

    fn get(&mut self, key: u64) -> Option<CachedResponse> {
        let Some(entry) = self.entries.get(&key).cloned() else {
            self.cache_misses += 1;
            return None;
        };
        if entry.created_at.elapsed().as_secs() < self.ttl_seconds {
            self.cache_hits += 1;
            self.touch(key);
            return Some(entry);
        }
        self.entries.remove(&key);
        self.order.retain(|existing| *existing != key);
        self.cache_evictions += 1;
        self.cache_misses += 1;
        None
    }

    fn put(&mut self, key: u64, status: StatusCode, headers: HeaderMap, body: Bytes) {
        if self.max_entries == 0 || self.ttl_seconds == 0 {
            return;
        }
        self.entries.insert(
            key,
            CachedResponse {
                status,
                headers,
                body,
                created_at: Instant::now(),
            },
        );
        self.touch(key);
        while self.entries.len() > self.max_entries {
            let Some(evicted_key) = self.order.pop_front() else {
                break;
            };
            if self.entries.remove(&evicted_key).is_some() {
                self.cache_evictions += 1;
            }
        }
    }

    fn clear(&mut self) -> usize {
        let previous_size = self.entries.len();
        self.entries.clear();
        self.order.clear();
        previous_size
    }

    fn stats(&self) -> ResponseCacheStats {
        ResponseCacheStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            cache_evictions: self.cache_evictions,
            cache_size: self.entries.len(),
        }
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|existing| *existing != key);
        self.order.push_back(key);
    }
}

impl CompressionCache {
    fn is_skipped(&mut self, key: u64) -> bool {
        let Some(created_at) = self.skips.get(&key).copied() else {
            return false;
        };
        if created_at.elapsed().as_secs() < self.ttl_seconds {
            self.cache_skip_hits += 1;
            return true;
        }
        self.skips.remove(&key);
        self.cache_evictions += 1;
        false
    }

    fn get(&mut self, key: u64) -> Option<CachedCompression> {
        let Some(entry) = self.results.get(&key).cloned() else {
            self.cache_misses += 1;
            return None;
        };
        if entry.created_at.elapsed().as_secs() < self.ttl_seconds {
            self.cache_hits += 1;
            return Some(entry);
        }
        self.results.remove(&key);
        self.cache_evictions += 1;
        self.cache_misses += 1;
        None
    }

    fn put(&mut self, key: u64, compressed: String, compression_ratio: f64, strategy: String) {
        self.results.insert(
            key,
            CachedCompression {
                compressed,
                compression_ratio,
                strategy,
                created_at: Instant::now(),
            },
        );
    }

    fn mark_skip(&mut self, key: u64) {
        self.skips.insert(key, Instant::now());
    }

    fn move_to_skip(&mut self, key: u64) {
        if self.results.remove(&key).is_some() {
            self.cache_evictions += 1;
        }
        self.skips.insert(key, Instant::now());
    }

    fn clear(&mut self) -> (usize, usize) {
        let previous_size = self.results.len();
        let previous_skip_size = self.skips.len();
        self.results.clear();
        self.skips.clear();
        (previous_size, previous_skip_size)
    }

    fn stats(&self) -> CompressionCacheStats {
        CompressionCacheStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            cache_skip_hits: self.cache_skip_hits,
            cache_evictions: self.cache_evictions,
            cache_size: self.results.len(),
            cache_skip_size: self.skips.len(),
        }
    }
}

/// Build the axum app. Health/readiness endpoints are intercepted; everything
/// else hits the catch-all forwarder. WebSocket upgrades are handled inside
/// the catch-all handler when an `Upgrade: websocket` header is present.
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/livez", get(livez))
        .route("/metrics", get(metrics))
        .route("/readyz", get(readyz))
        .route("/healthz", get(healthz))
        .route("/healthz/upstream", get(healthz_upstream))
        .route("/stats", get(admin_stats))
        .route("/stats-history", get(admin_stats_history))
        .route("/transformations/feed", get(admin_transformations_feed))
        .route("/subscription-window", get(admin_subscription_window))
        .route("/quota", get(admin_quota))
        .route("/debug/memory", get(admin_debug_memory))
        .route("/debug/tasks", get(admin_debug_tasks))
        .route("/debug/ws-sessions", get(admin_debug_ws_sessions))
        .route("/debug/warmup", get(admin_debug_warmup))
        .route("/dashboard", get(admin_dashboard))
        .route("/cache/clear", post(admin_cache_clear))
        .route("/v1/compress", post(compress_messages))
        .route("/v1/telemetry", get(telemetry_stats))
        .route("/v1/telemetry/export", get(telemetry_export))
        .route("/v1/telemetry/import", post(telemetry_import))
        .route("/v1/telemetry/tools", get(telemetry_tools))
        .route(
            "/v1/telemetry/tools/{signature_hash}",
            get(telemetry_tool_detail),
        )
        .route("/v1/retrieve", post(ccr_retrieve))
        .route("/v1/retrieve/stats", get(ccr_retrieve_stats))
        .route("/v1/retrieve/{hash_key}", get(ccr_retrieve_get))
        .route("/v1/retrieve/tool_call", post(ccr_retrieve_tool_call))
        .route("/v1/feedback", get(ccr_feedback))
        .route("/v1/feedback/{tool_name}", get(ccr_feedback_for_tool))
        .route("/v1/toin/stats", get(toin_stats))
        .route("/v1/toin/patterns", get(toin_patterns))
        .route("/v1/toin/pattern/{hash_prefix}", get(toin_pattern_detail))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/messages/batches", post(anthropic_batches_create))
        .route("/v1/messages/batches", get(anthropic_batches_list))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/batches", post(openai_batches_create))
        .route("/v1/batches", get(openai_batches_list))
        .route("/v1/batches/{batch_id}", get(openai_batches_get))
        .route("/v1/batches/{batch_id}/cancel", post(openai_batches_cancel))
        .route("/v1/models", get(model_metadata_list))
        .route("/v1/models/{model_id}", get(model_metadata_get))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/moderations", post(openai_moderations))
        .route("/v1/images/generations", post(openai_images_generations))
        .route(
            "/v1/audio/transcriptions",
            post(openai_audio_transcriptions),
        )
        .route("/v1/audio/speech", post(openai_audio_speech))
        .fallback(any(catch_all))
        .with_state(state)
}

async fn anthropic_messages(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !state.config.native_anthropic_messages {
        return forward_http(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }

    forward_native_anthropic_messages(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn admin_stats(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let cached = uri_query_param(req.uri(), "cached")
        .map(|value| matches!(value.as_str(), "1" | "true" | "True" | "TRUE"))
        .unwrap_or(false);
    let payload = if cached {
        cached_stats_payload(&state)
    } else {
        build_stats_payload(&state)
    };
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_stats",
        "admin_stats",
        product_passthrough_metadata("stats"),
        StatusCode::OK,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

fn cached_stats_payload(state: &AppState) -> Value {
    let now = Instant::now();
    {
        let snapshot = state
            .stats_snapshot
            .lock()
            .expect("stats snapshot poisoned");
        if let Some(snapshot) = snapshot.as_ref() {
            if now < snapshot.expires_at {
                return snapshot.payload.clone();
            }
        }
    }

    let payload = build_stats_payload(state);
    let snapshot = StatsSnapshot {
        expires_at: now + DASHBOARD_STATS_CACHE_TTL,
        payload: payload.clone(),
    };
    *state
        .stats_snapshot
        .lock()
        .expect("stats snapshot poisoned") = Some(snapshot);
    payload
}

async fn admin_stats_history(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let format = uri_query_param(req.uri(), "format").unwrap_or_else(|| "json".to_string());
    let series = uri_query_param(req.uri(), "series").unwrap_or_else(|| "history".to_string());
    let history_mode =
        uri_query_param(req.uri(), "history_mode").unwrap_or_else(|| "compact".to_string());
    let request_log_entries = state.request_logs.snapshot();
    let storage_path = state.request_logs.storage_path();
    if format == "csv" {
        return local_text_route(
            state,
            client_addr,
            req,
            "proxy.admin_stats_history",
            "admin_stats_history",
            product_passthrough_metadata("stats-history"),
            StatusCode::OK,
            "text/csv; charset=utf-8",
            Some((
                "content-disposition",
                format!("attachment; filename=\"headroom-stats-history-{series}.csv\""),
            )),
            build_stats_history_csv(&request_log_entries, &series, storage_path.as_deref()),
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_stats_history",
        "admin_stats_history",
        product_passthrough_metadata("stats-history"),
        StatusCode::OK,
        build_stats_history_payload(&request_log_entries, &history_mode, storage_path.as_deref()),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_transformations_feed(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let limit = uri_query_param(req.uri(), "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.min(100))
        .unwrap_or(20);
    let payload = serde_json::json!({
        "transformations": state.request_logs.recent_with_messages(limit),
        "log_full_messages": false,
    });
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_transformations_feed",
        "admin_transformations_feed",
        product_passthrough_metadata("transformations/feed"),
        StatusCode::OK,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_subscription_window(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let payload = serde_json::json!({
        "error": "Subscription tracking is not enabled"
    });
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_subscription_window",
        "admin_subscription_window",
        product_passthrough_metadata("subscription-window"),
        StatusCode::SERVICE_UNAVAILABLE,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_quota(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_quota",
        "admin_quota",
        product_passthrough_metadata("quota"),
        StatusCode::OK,
        serde_json::json!({}),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_debug_memory(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !is_loopback_addr(client_addr) {
        return StatusCode::NOT_FOUND.into_response();
    }
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_debug_memory",
        "admin_debug_memory",
        product_passthrough_metadata("debug/memory"),
        StatusCode::OK,
        build_debug_memory_payload(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_debug_tasks(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !is_loopback_addr(client_addr) {
        return StatusCode::NOT_FOUND.into_response();
    }
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.admin_debug_tasks",
        "admin_debug_tasks",
        product_passthrough_metadata("debug/tasks"),
        StatusCode::OK,
        Value::Array(state.ws_sessions.debug_tasks_snapshot()),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_debug_ws_sessions(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !is_loopback_addr(client_addr) {
        return StatusCode::NOT_FOUND.into_response();
    }
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.admin_debug_ws_sessions",
        "admin_debug_ws_sessions",
        product_passthrough_metadata("debug/ws-sessions"),
        StatusCode::OK,
        Value::Array(state.ws_sessions.snapshot()),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_debug_warmup(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !is_loopback_addr(client_addr) {
        return StatusCode::NOT_FOUND.into_response();
    }
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.admin_debug_warmup",
        "admin_debug_warmup",
        product_passthrough_metadata("debug/warmup"),
        StatusCode::OK,
        serde_json::json!({
            "runtime": build_runtime_debug_payload(&state),
        }),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_dashboard(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    local_text_route(
        state,
        client_addr,
        req,
        "proxy.admin_dashboard",
        "admin_dashboard",
        product_passthrough_metadata("dashboard"),
        StatusCode::OK,
        "text/html; charset=utf-8",
        None,
        DASHBOARD_HTML.to_string(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn admin_cache_clear(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let (previous_size, previous_skip_size) = state
        .compression_cache
        .lock()
        .expect("compression cache poisoned")
        .clear();
    let previous_response_size = state
        .response_cache
        .lock()
        .expect("response cache poisoned")
        .clear();
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.admin_cache_clear",
        "admin_cache_clear",
        product_passthrough_metadata("cache/clear"),
        StatusCode::OK,
        serde_json::json!({
            "status": "cache cleared",
            "previous_size": previous_size,
            "previous_skip_size": previous_skip_size,
            "previous_response_size": previous_response_size,
        }),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn telemetry_stats(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let payload = state.telemetry.stats();
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.telemetry_stats",
        "telemetry_stats",
        product_passthrough_metadata("telemetry"),
        StatusCode::OK,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn telemetry_export(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let payload = state.telemetry.export_stats();
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.telemetry_export",
        "telemetry_export",
        product_passthrough_metadata("telemetry/export"),
        StatusCode::OK,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn telemetry_import(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let (parts, body) = req.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return ProxyError::InvalidRequest(format!(
                "failed to read telemetry import body: {err}"
            ))
            .into_response()
        }
    };
    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(value) => value,
        Err(err) => {
            return ProxyError::InvalidRequest(format!("invalid telemetry import payload: {err}"))
                .into_response()
        }
    };
    let current_stats = state.telemetry.import_stats(&payload);
    let response = serde_json::json!({
        "status": "imported",
        "current_stats": current_stats,
    });
    let req = Request::from_parts(parts, Body::empty());
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.telemetry_import",
        "telemetry_import",
        product_passthrough_metadata("telemetry/import"),
        StatusCode::OK,
        response,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn telemetry_tools(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let payload = state.telemetry.tool_list();
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.telemetry_tools",
        "telemetry_tools",
        product_passthrough_metadata("telemetry/tools"),
        StatusCode::OK,
        payload,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn telemetry_tool_detail(
    Path(signature_hash): Path<String>,
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    telemetry_tool_detail_with_signature(state, client_addr, req, signature_hash).await
}

async fn telemetry_tool_detail_with_signature(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    signature_hash: String,
) -> Response<Body> {
    let (status, response) = match state.telemetry.tool_detail(&signature_hash) {
        Some(detail) => (StatusCode::OK, detail),
        None => (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "error": format!("telemetry tool {signature_hash} not found")
            }),
        ),
    };
    let mut metadata = product_passthrough_metadata("telemetry/tools/detail");
    metadata.insert("signature_hash", signature_hash.clone());
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.telemetry_tool_detail",
        "telemetry_tool_detail",
        metadata,
        status,
        response,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_retrieve(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let (parts, body) = req.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return ProxyError::InvalidRequest(format!("failed to read CCR retrieve body: {err}"))
                .into_response()
        }
    };
    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(value) => value,
        Err(err) => {
            return ProxyError::InvalidRequest(format!("invalid CCR retrieve payload: {err}"))
                .into_response()
        }
    };
    let Some(hash_key) = payload.get("hash").and_then(Value::as_str) else {
        return ProxyError::InvalidRequest("hash required".to_string()).into_response();
    };
    let response = if let Some(query) = payload.get("query").and_then(Value::as_str) {
        if let Some(tool_name) = state.product.tool_name_for_hash(hash_key) {
            state.telemetry.record_retrieval(&tool_name, "search");
        }
        (StatusCode::OK, state.product.search(hash_key, query))
    } else if let Some(entry) = state.product.retrieve(hash_key) {
        if let Some(tool_name) = entry.get("tool_name").and_then(Value::as_str) {
            state.telemetry.record_retrieval(tool_name, "full");
        }
        (StatusCode::OK, entry)
    } else {
        (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "detail": "Entry not found or expired (TTL: 5 minutes)"
            }),
        )
    };
    local_json_route(
        state,
        client_addr,
        Request::from_parts(parts, Body::empty()),
        "proxy.ccr_retrieve",
        "ccr_retrieve",
        product_passthrough_metadata("retrieve"),
        response.0,
        response.1,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_retrieve_stats(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.ccr_retrieve_stats",
        "ccr_retrieve_stats",
        product_passthrough_metadata("retrieve/stats"),
        StatusCode::OK,
        state.product.ccr_stats(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_retrieve_get(
    Path(hash_key): Path<String>,
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    ccr_retrieve_get_with_hash(state, client_addr, req, hash_key).await
}

async fn ccr_retrieve_get_with_hash(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    hash_key: String,
) -> Response<Body> {
    let query = uri_query_param(req.uri(), "query");
    let response = if let Some(query) = query.as_deref() {
        if let Some(tool_name) = state.product.tool_name_for_hash(&hash_key) {
            state.telemetry.record_retrieval(&tool_name, "search");
        }
        (StatusCode::OK, state.product.search(&hash_key, query))
    } else if let Some(entry) = state.product.retrieve(&hash_key) {
        if let Some(tool_name) = entry.get("tool_name").and_then(Value::as_str) {
            state.telemetry.record_retrieval(tool_name, "full");
        }
        (StatusCode::OK, entry)
    } else {
        (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "detail": "Entry not found or expired"
            }),
        )
    };
    let mut metadata = product_passthrough_metadata("retrieve/item");
    metadata.insert("hash_key", hash_key);
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.ccr_retrieve_get",
        "ccr_retrieve_get",
        metadata,
        response.0,
        response.1,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_retrieve_tool_call(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let (parts, body) = req.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return ProxyError::InvalidRequest(format!("failed to read CCR tool call body: {err}"))
                .into_response()
        }
    };
    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(value) => value,
        Err(err) => {
            return ProxyError::InvalidRequest(format!("invalid CCR tool call payload: {err}"))
                .into_response()
        }
    };
    let tool_call = payload
        .get("tool_call")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    let provider = payload
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or("anthropic");
    let Some((hash_key, query, tool_call_id)) = parse_ccr_tool_call(&tool_call, provider) else {
        return ProxyError::InvalidRequest(
            "Invalid tool call or not a headroom_retrieve call".to_string(),
        )
        .into_response();
    };
    let retrieval_data = if let Some(query) = query.as_deref() {
        if let Some(tool_name) = state.product.tool_name_for_hash(&hash_key) {
            state.telemetry.record_retrieval(&tool_name, "search");
        }
        state.product.search(&hash_key, query)
    } else if let Some(entry) = state.product.retrieve(&hash_key) {
        if let Some(tool_name) = entry.get("tool_name").and_then(Value::as_str) {
            state.telemetry.record_retrieval(tool_name, "full");
        }
        entry
    } else {
        serde_json::json!({
            "error": "Entry not found or expired (TTL: 5 minutes)",
            "hash": hash_key,
        })
    };
    let result_content =
        serde_json::to_string_pretty(&retrieval_data).unwrap_or_else(|_| "{}".to_string());
    let tool_result = match provider {
        "anthropic" => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result_content,
        }),
        "openai" => serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        }),
        _ => serde_json::json!({
            "tool_call_id": tool_call_id,
            "content": result_content,
        }),
    };
    local_json_route(
        state,
        client_addr,
        Request::from_parts(parts, Body::empty()),
        "proxy.ccr_retrieve_tool_call",
        "ccr_retrieve_tool_call",
        product_passthrough_metadata("retrieve/tool_call"),
        StatusCode::OK,
        serde_json::json!({
            "tool_result": tool_result,
            "success": retrieval_data.get("error").is_none(),
            "data": retrieval_data,
        }),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_feedback(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.ccr_feedback",
        "ccr_feedback",
        product_passthrough_metadata("feedback"),
        StatusCode::OK,
        state.product.feedback_stats(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn ccr_feedback_for_tool(
    Path(tool_name): Path<String>,
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    ccr_feedback_for_tool_name(state, client_addr, req, tool_name).await
}

async fn ccr_feedback_for_tool_name(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    tool_name: String,
) -> Response<Body> {
    let mut metadata = product_passthrough_metadata("feedback/tool");
    metadata.insert("tool_name", tool_name.clone());
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.ccr_feedback_tool",
        "ccr_feedback_tool",
        metadata,
        StatusCode::OK,
        state.product.feedback_for_tool(&tool_name),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn toin_stats(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.toin_stats",
        "toin_stats",
        product_passthrough_metadata("toin/stats"),
        StatusCode::OK,
        state.product.toin_stats(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn toin_patterns(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let limit = uri_query_param(req.uri(), "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    local_json_route(
        state.clone(),
        client_addr,
        req,
        "proxy.toin_patterns",
        "toin_patterns",
        product_passthrough_metadata("toin/patterns"),
        StatusCode::OK,
        Value::Array(state.product.toin_patterns(limit)),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn toin_pattern_detail(
    Path(hash_prefix): Path<String>,
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    toin_pattern_detail_with_prefix(state, client_addr, req, hash_prefix).await
}

async fn toin_pattern_detail_with_prefix(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    hash_prefix: String,
) -> Response<Body> {
    let mut metadata = product_passthrough_metadata("toin/pattern/detail");
    metadata.insert("hash_prefix", hash_prefix.clone());
    let response = match state.product.toin_pattern_detail(&hash_prefix) {
        Some(pattern) => (StatusCode::OK, pattern),
        None => (
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "detail": format!("No TOIN pattern found with hash starting with: {hash_prefix}")
            }),
        ),
    };
    local_json_route(
        state,
        client_addr,
        req,
        "proxy.toin_pattern_detail",
        "toin_pattern_detail",
        metadata,
        response.0,
        response.1,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn compress_messages(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    native_compress_messages(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn anthropic_count_tokens(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !state.config.native_anthropic_count_tokens {
        return forward_http(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }

    forward_native_anthropic_count_tokens(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn openai_chat(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    if !state.config.native_openai_chat {
        return forward_http(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }

    forward_native_openai_chat(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn anthropic_batches_create(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_native_anthropic_batch_create(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn anthropic_batches_list(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_anthropic_passthrough_route(
        state,
        client_addr,
        req,
        "proxy.anthropic_batch_list",
        "anthropic_batch_list",
        "messages/batches",
        PipelineMetadata::new(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_embeddings(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_utility_route(
        state,
        client_addr,
        req,
        "proxy.openai_embeddings",
        "openai_embeddings",
        "embeddings",
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn model_metadata_list(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let provider = model_metadata_provider(req.headers());
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("model_metadata_provider", provider);
    let upstream_base = match provider_target_base_url(provider, req.headers(), &state.config) {
        Ok(base) => base,
        Err(error) => return error.into_response(),
    };
    forward_http_with_route_target(
        state,
        client_addr,
        req,
        "proxy.model_metadata_list",
        "model_metadata_list",
        Some((None, route_metadata)),
        Some(provider),
        Some(upstream_base),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn model_metadata_get(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(model_id): Path<String>,
    req: Request<Body>,
) -> Response<Body> {
    let provider = model_metadata_provider(req.headers());
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("model_metadata_provider", provider);
    route_metadata.insert("model_id", model_id.clone());
    let upstream_base = match provider_target_base_url(provider, req.headers(), &state.config) {
        Ok(base) => base,
        Err(error) => return error.into_response(),
    };
    forward_http_with_route_target(
        state,
        client_addr,
        req,
        "proxy.model_metadata_get",
        "model_metadata_get",
        Some((Some(model_id), route_metadata)),
        Some(provider),
        Some(upstream_base),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_batches_create(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_native_openai_batch_create(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

async fn openai_batches_list(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_passthrough_route(
        state,
        client_addr,
        req,
        "proxy.openai_batch_list",
        "openai_batch_list",
        "batches",
        PipelineMetadata::new(),
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_batches_get(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(batch_id): Path<String>,
    req: Request<Body>,
) -> Response<Body> {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("batch_id", batch_id);
    forward_openai_passthrough_route(
        state,
        client_addr,
        req,
        "proxy.openai_batch_get",
        "openai_batch_get",
        "batches",
        route_metadata,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_batches_cancel(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    Path(batch_id): Path<String>,
    req: Request<Body>,
) -> Response<Body> {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("batch_id", batch_id);
    forward_openai_passthrough_route(
        state,
        client_addr,
        req,
        "proxy.openai_batch_cancel",
        "openai_batch_cancel",
        "batches",
        route_metadata,
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_moderations(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_utility_route(
        state,
        client_addr,
        req,
        "proxy.openai_moderations",
        "openai_moderations",
        "moderations",
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_images_generations(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_utility_route(
        state,
        client_addr,
        req,
        "proxy.openai_images_generations",
        "openai_images_generations",
        "images/generations",
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_audio_transcriptions(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_utility_route(
        state,
        client_addr,
        req,
        "proxy.openai_audio_transcriptions",
        "openai_audio_transcriptions",
        "audio/transcriptions",
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

async fn openai_audio_speech(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    forward_openai_utility_route(
        state,
        client_addr,
        req,
        "proxy.openai_audio_speech",
        "openai_audio_speech",
        "audio/speech",
    )
    .await
    .unwrap_or_else(|e| e.into_response())
}

/// Catch-all handler. If the request is a WebSocket upgrade, hand off to the
/// ws module; otherwise forward as plain HTTP.
async fn catch_all(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    ws: Option<WebSocketUpgrade>,
    req: Request<Body>,
) -> Response<Body> {
    if is_openai_responses_family_request(req.method(), req.uri().path()) {
        let (req, route_metadata, upstream_base) =
            match prepare_openai_responses_request(req, &state.config.openai_api_url) {
                Ok(rewritten) => rewritten,
                Err(error) => return error.into_response(),
            };
        if is_websocket_upgrade(req.headers()) {
            if let Some(ws) = ws {
                return crate::websocket::ws_handler_with_base(
                    ws,
                    state,
                    client_addr,
                    req,
                    upstream_base,
                )
                .await;
            }
        }
        return forward_http_with_route_target(
            state,
            client_addr,
            req,
            "proxy.openai_responses",
            "openai_responses",
            Some((None, route_metadata)),
            Some("openai"),
            Some(upstream_base),
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if let Some((operation, route_mode, route_details, provider)) =
        model_metadata_route(req.method(), req.uri().path(), req.headers())
    {
        let upstream_base = match provider_target_base_url(provider, req.headers(), &state.config) {
            Ok(base) => base,
            Err(error) => return error.into_response(),
        };
        return forward_http_with_route_target(
            state,
            client_addr,
            req,
            operation,
            route_mode,
            route_details,
            Some(provider),
            Some(upstream_base),
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if let Some((operation, route_mode, route_details)) =
        anthropic_batch_route(req.method(), req.uri().path())
    {
        return forward_http_with_route(
            state,
            client_addr,
            req,
            operation,
            route_mode,
            route_details,
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if let Some((model, route_metadata)) =
        databricks_invocations_route(req.method(), req.uri().path())
    {
        let databricks_base_url =
            normalize_provider_base_url(state.config.databricks_api_url.clone());
        return forward_http_with_route_target(
            state,
            client_addr,
            req,
            "proxy.databricks_invocations",
            "databricks_invocations",
            Some((Some(model), route_metadata)),
            Some("databricks"),
            Some(databricks_base_url),
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    let telemetry_tool_detail = req
        .uri()
        .path()
        .strip_prefix("/v1/telemetry/tools/")
        .map(str::to_string);
    if let Some(signature_hash) = telemetry_tool_detail {
        if req.method() == Method::GET
            && !signature_hash.is_empty()
            && !signature_hash.contains('/')
        {
            return telemetry_tool_detail_with_signature(state, client_addr, req, signature_hash)
                .await;
        }
    }
    let ccr_retrieve_hash = req
        .uri()
        .path()
        .strip_prefix("/v1/retrieve/")
        .map(str::to_string);
    if let Some(hash_key) = ccr_retrieve_hash {
        if req.method() == Method::GET
            && !hash_key.is_empty()
            && !hash_key.contains('/')
            && hash_key != "stats"
            && hash_key != "tool_call"
        {
            return ccr_retrieve_get_with_hash(state, client_addr, req, hash_key).await;
        }
    }
    let feedback_tool = req
        .uri()
        .path()
        .strip_prefix("/v1/feedback/")
        .map(str::to_string);
    if let Some(tool_name) = feedback_tool {
        if req.method() == Method::GET && !tool_name.is_empty() && !tool_name.contains('/') {
            return ccr_feedback_for_tool_name(state, client_addr, req, tool_name).await;
        }
    }
    let toin_pattern = req
        .uri()
        .path()
        .strip_prefix("/v1/toin/pattern/")
        .map(str::to_string);
    if let Some(hash_prefix) = toin_pattern {
        if req.method() == Method::GET && !hash_prefix.is_empty() && !hash_prefix.contains('/') {
            return toin_pattern_detail_with_prefix(state, client_addr, req, hash_prefix).await;
        }
    }
    if let Some((operation, route_mode, route_details)) =
        product_passthrough_route(req.method(), req.uri().path())
    {
        return forward_http_with_route(
            state,
            client_addr,
            req,
            operation,
            route_mode,
            route_details,
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if let Some((operation, route_mode, route_details)) =
        gemini_passthrough_route(req.method(), req.uri().path())
    {
        let gemini_base_url = normalize_provider_base_url(state.config.gemini_api_url.clone());
        return forward_http_with_route_target(
            state,
            client_addr,
            req,
            operation,
            route_mode,
            route_details,
            Some("gemini"),
            Some(gemini_base_url),
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if let Some((operation, route_mode, route_details)) =
        openai_batch_route(req.method(), req.uri().path())
    {
        return forward_http_with_route(
            state,
            client_addr,
            req,
            operation,
            route_mode,
            route_details,
        )
        .await
        .unwrap_or_else(|e| e.into_response());
    }
    if is_gemini_generate_content_request(req.method(), req.uri().path())
        && state.config.native_gemini_generate_content
    {
        return forward_native_gemini_generate_content(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }
    if is_gemini_count_tokens_request(req.method(), req.uri().path())
        && state.config.native_gemini_count_tokens
    {
        return forward_native_gemini_count_tokens(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }
    if is_google_cloudcode_stream_request(req.method(), req.uri().path())
        && state.config.native_google_cloudcode_stream
    {
        return forward_native_google_cloudcode_stream(state, client_addr, req)
            .await
            .unwrap_or_else(|e| e.into_response());
    }

    if is_websocket_upgrade(req.headers()) {
        if let Some(ws) = ws {
            return ws_handler(ws, state, client_addr, req).await;
        }
        // Header says websocket but axum didn't extract it (likely missing
        // Sec-WebSocket-Key) — fall through to HTTP forwarding which will
        // surface the upstream error.
    }
    forward_http(state, client_addr, req)
        .await
        .unwrap_or_else(|e| e.into_response())
}

fn is_gemini_generate_content_request(method: &Method, path: &str) -> bool {
    is_gemini_model_request(method, path, ":generateContent")
        || is_gemini_model_request(method, path, ":streamGenerateContent")
}

fn model_metadata_route(
    method: &Method,
    path: &str,
    headers: &HeaderMap,
) -> Option<(
    &'static str,
    &'static str,
    Option<(Option<String>, PipelineMetadata)>,
    &'static str,
)> {
    if *method != Method::GET || !path.starts_with("/v1/models") {
        return None;
    }

    let provider = model_metadata_provider(headers);
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("model_metadata_provider", provider);

    if path == "/v1/models" {
        return Some((
            "proxy.model_metadata_list",
            "model_metadata_list",
            Some((None, route_metadata)),
            provider,
        ));
    }

    let model_id = path.strip_prefix("/v1/models/")?;
    if model_id.is_empty() || model_id.contains('/') {
        return None;
    }

    route_metadata.insert("model_id", model_id.to_string());
    Some((
        "proxy.model_metadata_get",
        "model_metadata_get",
        Some((Some(model_id.to_string()), route_metadata)),
        provider,
    ))
}

fn openai_batch_route(
    method: &Method,
    path: &str,
) -> Option<(
    &'static str,
    &'static str,
    Option<(Option<String>, PipelineMetadata)>,
)> {
    if path == "/v1/batches" {
        return match *method {
            Method::POST => Some((
                "proxy.openai_batch_create",
                "openai_batch_create",
                Some((None, openai_passthrough_metadata("batches"))),
            )),
            Method::GET => Some((
                "proxy.openai_batch_list",
                "openai_batch_list",
                Some((None, openai_passthrough_metadata("batches"))),
            )),
            _ => None,
        };
    }

    let batch_tail = path.strip_prefix("/v1/batches/")?;
    if batch_tail.is_empty() {
        return None;
    }

    if let Some(batch_id) = batch_tail.strip_suffix("/cancel") {
        if *method != Method::POST || batch_id.is_empty() || batch_id.contains('/') {
            return None;
        }
        let mut route_metadata = openai_passthrough_metadata("batches");
        route_metadata.insert("batch_id", batch_id.to_string());
        return Some((
            "proxy.openai_batch_cancel",
            "openai_batch_cancel",
            Some((None, route_metadata)),
        ));
    }

    if *method != Method::GET || batch_tail.contains('/') {
        return None;
    }

    let mut route_metadata = openai_passthrough_metadata("batches");
    route_metadata.insert("batch_id", batch_tail.to_string());
    Some((
        "proxy.openai_batch_get",
        "openai_batch_get",
        Some((None, route_metadata)),
    ))
}

fn anthropic_batch_route(
    method: &Method,
    path: &str,
) -> Option<(
    &'static str,
    &'static str,
    Option<(Option<String>, PipelineMetadata)>,
)> {
    if path == "/v1/messages/batches" {
        return match *method {
            Method::POST => Some((
                "proxy.anthropic_batch_create",
                "anthropic_batch_create",
                Some((None, anthropic_passthrough_metadata("messages/batches"))),
            )),
            Method::GET => Some((
                "proxy.anthropic_batch_list",
                "anthropic_batch_list",
                Some((None, anthropic_passthrough_metadata("messages/batches"))),
            )),
            _ => None,
        };
    }

    let batch_tail = path.strip_prefix("/v1/messages/batches/")?;
    if batch_tail.is_empty() {
        return None;
    }

    if let Some(batch_id) = batch_tail.strip_suffix("/results") {
        if *method != Method::GET || batch_id.is_empty() || batch_id.contains('/') {
            return None;
        }
        let mut route_metadata = anthropic_passthrough_metadata("messages/batches");
        route_metadata.insert("batch_id", batch_id.to_string());
        route_metadata.insert("result_stream", true);
        return Some((
            "proxy.anthropic_batch_results",
            "anthropic_batch_results",
            Some((None, route_metadata)),
        ));
    }

    if let Some(batch_id) = batch_tail.strip_suffix("/cancel") {
        if *method != Method::POST || batch_id.is_empty() || batch_id.contains('/') {
            return None;
        }
        let mut route_metadata = anthropic_passthrough_metadata("messages/batches");
        route_metadata.insert("batch_id", batch_id.to_string());
        return Some((
            "proxy.anthropic_batch_cancel",
            "anthropic_batch_cancel",
            Some((None, route_metadata)),
        ));
    }

    if *method != Method::GET || batch_tail.contains('/') {
        return None;
    }

    let mut route_metadata = anthropic_passthrough_metadata("messages/batches");
    route_metadata.insert("batch_id", batch_tail.to_string());
    Some((
        "proxy.anthropic_batch_get",
        "anthropic_batch_get",
        Some((None, route_metadata)),
    ))
}

fn databricks_invocations_route(method: &Method, path: &str) -> Option<(String, PipelineMetadata)> {
    if *method != Method::POST {
        return None;
    }

    let suffix = "/invocations";
    let model = path
        .strip_prefix("/serving-endpoints/")?
        .strip_suffix(suffix)?;

    if model.is_empty() || model.contains('/') {
        return None;
    }

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", "serving-endpoints/invocations");
    Some((model.to_string(), route_metadata))
}

fn gemini_passthrough_route(
    method: &Method,
    path: &str,
) -> Option<(
    &'static str,
    &'static str,
    Option<(Option<String>, PipelineMetadata)>,
)> {
    if path == "/v1beta/models" && *method == Method::GET {
        return Some((
            "proxy.gemini_list_models",
            "gemini_list_models",
            Some((None, gemini_passthrough_metadata("models"))),
        ));
    }

    if let Some(model_path) = path.strip_prefix("/v1beta/models/") {
        if let Some(model) = model_path.strip_suffix(":embedContent") {
            if *method == Method::POST && !model.is_empty() && !model.contains('/') {
                return Some((
                    "proxy.gemini_embed_content",
                    "gemini_embed_content",
                    Some((
                        Some(model.to_string()),
                        gemini_passthrough_metadata("embedContent"),
                    )),
                ));
            }
            return None;
        }
        if let Some(model) = model_path.strip_suffix(":batchEmbedContents") {
            if *method == Method::POST && !model.is_empty() && !model.contains('/') {
                return Some((
                    "proxy.gemini_batch_embed_contents",
                    "gemini_batch_embed_contents",
                    Some((
                        Some(model.to_string()),
                        gemini_passthrough_metadata("batchEmbedContents"),
                    )),
                ));
            }
            return None;
        }
        if let Some(model) = model_path.strip_suffix(":batchGenerateContent") {
            if *method == Method::POST && !model.is_empty() && !model.contains('/') {
                return Some((
                    "proxy.gemini_batch_generate_content",
                    "gemini_batch_generate_content",
                    Some((
                        Some(model.to_string()),
                        gemini_passthrough_metadata("batchGenerateContent"),
                    )),
                ));
            }
            return None;
        }
        if !model_path.is_empty()
            && !model_path.contains('/')
            && !model_path.contains(':')
            && *method == Method::GET
        {
            let mut metadata = gemini_passthrough_metadata("models");
            metadata.insert("model_id", model_path.to_string());
            return Some((
                "proxy.gemini_get_model",
                "gemini_get_model",
                Some((Some(model_path.to_string()), metadata)),
            ));
        }
    }

    if let Some(batch_name) = path.strip_prefix("/v1beta/batches/") {
        if let Some(batch_name) = batch_name.strip_suffix(":cancel") {
            if *method == Method::POST && !batch_name.is_empty() && !batch_name.contains('/') {
                let mut metadata = gemini_passthrough_metadata("batches");
                metadata.insert("batch_name", batch_name.to_string());
                return Some((
                    "proxy.gemini_batch_cancel",
                    "gemini_batch_cancel",
                    Some((None, metadata)),
                ));
            }
            return None;
        }
        if !batch_name.is_empty() && !batch_name.contains('/') {
            let mut metadata = gemini_passthrough_metadata("batches");
            metadata.insert("batch_name", batch_name.to_string());
            return match *method {
                Method::GET => Some((
                    "proxy.gemini_batch_get",
                    "gemini_batch_get",
                    Some((None, metadata)),
                )),
                Method::DELETE => Some((
                    "proxy.gemini_batch_delete",
                    "gemini_batch_delete",
                    Some((None, metadata)),
                )),
                _ => None,
            };
        }
    }

    if path == "/v1beta/cachedContents" {
        return match *method {
            Method::POST => Some((
                "proxy.gemini_create_cached_content",
                "gemini_create_cached_content",
                Some((None, gemini_passthrough_metadata("cachedContents"))),
            )),
            Method::GET => Some((
                "proxy.gemini_list_cached_contents",
                "gemini_list_cached_contents",
                Some((None, gemini_passthrough_metadata("cachedContents"))),
            )),
            _ => None,
        };
    }

    if let Some(cache_id) = path.strip_prefix("/v1beta/cachedContents/") {
        if cache_id.is_empty() || cache_id.contains('/') {
            return None;
        }
        let mut metadata = gemini_passthrough_metadata("cachedContents");
        metadata.insert("cache_id", cache_id.to_string());
        return match *method {
            Method::GET => Some((
                "proxy.gemini_get_cached_content",
                "gemini_get_cached_content",
                Some((None, metadata)),
            )),
            Method::DELETE => Some((
                "proxy.gemini_delete_cached_content",
                "gemini_delete_cached_content",
                Some((None, metadata)),
            )),
            _ => None,
        };
    }

    None
}

fn product_passthrough_route(
    _method: &Method,
    _path: &str,
) -> Option<(
    &'static str,
    &'static str,
    Option<(Option<String>, PipelineMetadata)>,
)> {
    None
}

fn is_openai_responses_family_request(method: &Method, path: &str) -> bool {
    matches!(*method, Method::GET | Method::POST | Method::DELETE)
        && canonical_openai_responses_path(path).is_some()
}

fn is_gemini_count_tokens_request(method: &Method, path: &str) -> bool {
    is_gemini_model_request(method, path, ":countTokens")
}

fn is_gemini_model_request(method: &Method, path: &str, suffix: &str) -> bool {
    *method == Method::POST && path.starts_with("/v1beta/models/") && path.ends_with(suffix)
}

fn is_google_cloudcode_stream_request(method: &Method, path: &str) -> bool {
    *method == Method::POST
        && (path == "/v1internal:streamGenerateContent"
            || path == "/v1/v1internal:streamGenerateContent")
}

fn is_websocket_upgrade(headers: &HeaderMap) -> bool {
    let upgrade = headers
        .get(http::header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    let connection = headers
        .get(http::header::CONNECTION)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|t| t.trim().eq_ignore_ascii_case("upgrade"))
        })
        .unwrap_or(false);
    upgrade && connection
}

fn canonical_openai_responses_path(path: &str) -> Option<String> {
    for prefix in [
        "/v1/responses",
        "/v1/codex/responses",
        "/backend-api/responses",
        "/backend-api/codex/responses",
    ] {
        if let Some(suffix) = path.strip_prefix(prefix) {
            if suffix.is_empty() || suffix.starts_with('/') {
                return Some(format!("/v1/responses{suffix}"));
            }
        }
    }
    None
}

fn prepare_openai_responses_request(
    req: Request<Body>,
    default_openai_base: &url::Url,
) -> Result<(Request<Body>, PipelineMetadata, url::Url), ProxyError> {
    let (mut parts, body) = req.into_parts();
    let original_path = parts.uri.path().to_string();
    let canonical_path = canonical_openai_responses_path(&original_path)
        .ok_or_else(|| ProxyError::InvalidRequest("unsupported responses path".to_string()))?;
    let query = parts
        .uri
        .query()
        .map(|value| format!("?{value}"))
        .unwrap_or_default();
    parts.uri = format!("{canonical_path}{query}")
        .parse::<Uri>()
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid rewritten uri: {e}")))?;

    let chatgpt_account_id = resolve_chatgpt_account_id(&parts.headers);
    if let Some(account_id) = &chatgpt_account_id {
        parts.headers.insert(
            HeaderName::from_static("chatgpt-account-id"),
            HeaderValue::from_str(account_id)
                .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?,
        );
    }
    let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
    apply_openai_responses_websocket_defaults(&mut parts.headers, openai_api_key.as_deref())?;
    let target_path = chatgpt_responses_path(&canonical_path);
    let upstream_base = if chatgpt_account_id.is_some() {
        url::Url::parse("https://chatgpt.com")
            .map_err(|e| ProxyError::InvalidUpstream(format!("invalid ChatGPT base URL: {e}")))?
    } else {
        normalize_provider_base_url(default_openai_base.clone())
    };
    let rewritten_path = if chatgpt_account_id.is_some() {
        target_path.clone()
    } else {
        canonical_path.clone()
    };
    parts.uri = format!("{rewritten_path}{query}")
        .parse::<Uri>()
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid rewritten uri: {e}")))?;

    let mut metadata = PipelineMetadata::new();
    metadata.insert("responses_api", true);
    metadata.insert("original_path", original_path.clone());
    metadata.insert("canonical_path", canonical_path.clone());
    metadata.insert("target_path", rewritten_path.clone());
    metadata.insert("codex_alias", original_path != canonical_path);
    metadata.insert("subpath", canonical_path != "/v1/responses");
    metadata.insert("chatgpt_auth", chatgpt_account_id.is_some());

    Ok((Request::from_parts(parts, body), metadata, upstream_base))
}

fn apply_openai_responses_websocket_defaults(
    headers: &mut HeaderMap,
    openai_api_key: Option<&str>,
) -> Result<(), ProxyError> {
    if !is_websocket_upgrade(headers) {
        return Ok(());
    }

    if !headers.contains_key(http::header::AUTHORIZATION) {
        if let Some(api_key) = openai_api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            headers.insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?,
            );
        }
    }

    if !headers.contains_key("openai-beta") {
        headers.insert(
            HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("responses_websockets=2026-02-06"),
        );
    }

    Ok(())
}

fn chatgpt_responses_path(canonical_path: &str) -> String {
    let suffix = canonical_path
        .strip_prefix("/v1/responses")
        .unwrap_or_default();
    format!("/backend-api/codex/responses{suffix}")
}

fn resolve_chatgpt_account_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("chatgpt-account-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| decode_openai_bearer_payload(headers).and_then(extract_chatgpt_account_id))
}

fn decode_openai_bearer_payload(headers: &HeaderMap) -> Option<Value> {
    let auth = headers.get(http::header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = auth.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") || token.matches('.').count() < 2 {
        return None;
    }
    let payload = token.split('.').nth(1)?;
    let padded = format!("{payload}{}", "=".repeat((4 - payload.len() % 4) % 4));
    let decoded = URL_SAFE.decode(padded).ok()?;
    serde_json::from_slice::<Value>(&decoded).ok()
}

fn extract_chatgpt_account_id(payload: Value) -> Option<String> {
    payload
        .get("https://api.openai.com/auth")
        .and_then(Value::as_object)
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Build the upstream URL by joining the configured base with the incoming
/// path-and-query. Preserves '?' and the query string verbatim.
pub(crate) fn build_upstream_url(base: &url::Url, uri: &Uri) -> Result<url::Url, ProxyError> {
    Ok(join_upstream_path(base, uri.path(), uri.query()))
}

/// Shared path-join helper used by HTTP and WebSocket handlers.
/// Appends `path` to `base`, preserving any base path prefix, then sets `query`.
pub(crate) fn join_upstream_path(base: &url::Url, path: &str, query: Option<&str>) -> url::Url {
    let mut joined = base.clone();
    // Strip trailing slash from base path so "http://x:1/api" + "/v1/foo"
    // yields "http://x:1/api/v1/foo" rather than "http://x:1/v1/foo".
    let base_path = joined.path().trim_end_matches('/').to_string();
    let combined = if path.is_empty() || path == "/" {
        if base_path.is_empty() {
            "/".to_string()
        } else {
            base_path
        }
    } else if base_path.is_empty() {
        path.to_string()
    } else {
        format!("{base_path}{path}")
    };
    joined.set_path(&combined);
    joined.set_query(query);
    joined
}

/// Forward an HTTP request to the upstream and stream the response back.
async fn forward_http(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    forward_http_with_route(
        state,
        client_addr,
        req,
        "proxy.forward_http",
        "passthrough",
        None,
    )
    .await
}

async fn forward_native_openai_chat(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let (model, messages_count, streaming) = parse_openai_chat_request(&body_bytes)?;
    let body_bytes = if streaming {
        ensure_openai_stream_include_usage(&body_bytes)?
    } else {
        body_bytes
    };
    let optimization_started = Instant::now();
    let optimized = optimize_openai_chat_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
    )?;
    let optimization_latency_ms = optimization_started.elapsed().as_secs_f64() * 1000.0;
    validate_request_array_length("messages", messages_count)?;

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert(
        "shadow_mode",
        if state.config.native_openai_chat_shadow {
            "compare"
        } else {
            "disabled"
        },
    );
    route_metadata.insert("messages_count", messages_count);
    route_metadata.insert("stream", streaming);

    if state.config.native_openai_chat_shadow {
        let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));
        return forward_native_openai_chat_with_shadow(
            state,
            client_addr,
            req,
            body_bytes,
            optimized.compression_status,
            model,
            route_metadata,
            streaming,
        )
        .await;
    }

    let openai_base_url =
        select_openai_chat_base_url(&parts.headers, &state.config.openai_api_url)?;
    if !streaming {
        return forward_buffered_native_route_with_response_cache(
            state,
            client_addr,
            parts,
            optimized.primary_body_bytes,
            "proxy.openai_chat",
            "native_openai_chat",
            "openai",
            model,
            route_metadata,
            openai_base_url,
            optimized.compression_status,
            optimization_latency_ms,
        )
        .await;
    }

    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));

    forward_http_with_route_target_with_compression(
        state,
        client_addr,
        req,
        "proxy.openai_chat",
        "native_openai_chat",
        Some((Some(model), route_metadata)),
        Some("openai"),
        Some(openai_base_url),
        optimized.compression_status,
    )
    .await
}

async fn forward_openai_utility_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    endpoint: &'static str,
) -> Result<Response<Body>, ProxyError> {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    let openai_base_url = normalize_provider_base_url(state.config.openai_api_url.clone());
    forward_http_with_route_target(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        Some((None, route_metadata)),
        Some("openai"),
        Some(openai_base_url),
    )
    .await
}

async fn forward_openai_passthrough_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    endpoint: &'static str,
    mut route_metadata: PipelineMetadata,
) -> Result<Response<Body>, ProxyError> {
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    let openai_base_url = normalize_provider_base_url(state.config.openai_api_url.clone());

    forward_http_with_route_target(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        Some((None, route_metadata)),
        Some("openai"),
        Some(openai_base_url),
    )
    .await
}

async fn forward_anthropic_passthrough_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    endpoint: &'static str,
    mut route_metadata: PipelineMetadata,
) -> Result<Response<Body>, ProxyError> {
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    let anthropic_base_url = normalize_provider_base_url(state.config.anthropic_api_url.clone());

    forward_http_with_route_target(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        Some((None, route_metadata)),
        Some("anthropic"),
        Some(anthropic_base_url),
    )
    .await
}

fn openai_passthrough_metadata(endpoint: &'static str) -> PipelineMetadata {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    route_metadata
}

fn anthropic_passthrough_metadata(endpoint: &'static str) -> PipelineMetadata {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    route_metadata
}

fn gemini_passthrough_metadata(endpoint: &'static str) -> PipelineMetadata {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    route_metadata
}

fn product_passthrough_metadata(endpoint: &'static str) -> PipelineMetadata {
    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("endpoint", endpoint);
    route_metadata
}

async fn forward_native_anthropic_messages(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let (model, messages_count, max_tokens, streaming) =
        parse_anthropic_messages_request(&body_bytes)?;
    let optimization_started = Instant::now();
    let optimized = optimize_anthropic_messages_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
    )?;
    let optimization_latency_ms = optimization_started.elapsed().as_secs_f64() * 1000.0;
    validate_request_array_length("messages", messages_count)?;

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert(
        "shadow_mode",
        if state.config.native_anthropic_messages_shadow {
            "compare"
        } else {
            "disabled"
        },
    );
    route_metadata.insert("messages_count", messages_count);
    route_metadata.insert("max_tokens", max_tokens);
    route_metadata.insert("stream", streaming);

    if state.config.native_anthropic_messages_shadow {
        let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));
        return forward_native_anthropic_messages_with_shadow(
            state,
            client_addr,
            req,
            body_bytes,
            optimized.compression_status,
            model,
            route_metadata,
            streaming,
        )
        .await;
    }

    let anthropic_base_url = normalize_provider_base_url(state.config.anthropic_api_url.clone());
    if !streaming {
        return forward_buffered_native_route_with_response_cache(
            state,
            client_addr,
            parts,
            optimized.primary_body_bytes,
            "proxy.anthropic_messages",
            "native_anthropic_messages",
            "anthropic",
            model,
            route_metadata,
            anthropic_base_url,
            optimized.compression_status,
            optimization_latency_ms,
        )
        .await;
    }

    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));
    forward_http_with_route_target_with_compression(
        state,
        client_addr,
        req,
        "proxy.anthropic_messages",
        "native_anthropic_messages",
        Some((Some(model), route_metadata)),
        Some("anthropic"),
        Some(anthropic_base_url),
        optimized.compression_status,
    )
    .await
}

async fn forward_native_anthropic_batch_create(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let optimized = optimize_anthropic_batch_create_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
    )?;
    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes));

    let mut route_metadata = anthropic_passthrough_metadata("messages/batches");
    route_metadata.insert("requests_count", optimized.requests_count);
    let anthropic_base_url = normalize_provider_base_url(state.config.anthropic_api_url.clone());
    forward_http_with_route_target_with_compression(
        state,
        client_addr,
        req,
        "proxy.anthropic_batch_create",
        "anthropic_batch_create",
        Some((None, route_metadata)),
        Some("anthropic"),
        Some(anthropic_base_url),
        optimized.compression_status,
    )
    .await
}

async fn forward_native_openai_batch_create(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let request_id = ensure_request_id(&parts.headers);
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let request_head = request_from_parts_head(&parts)?;
    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(payload) => payload,
        Err(error) => {
            return local_json_route(
                state,
                client_addr,
                request_head,
                "proxy.openai_batch_create",
                "openai_batch_create",
                openai_passthrough_metadata("batches"),
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": {
                        "message": format!("Invalid request body: {error}"),
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                }),
            )
            .await;
        }
    };
    let Some(object) = payload.as_object() else {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            openai_passthrough_metadata("batches"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "message": "Invalid request body: request body must be a JSON object",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            }),
        )
        .await;
    };

    let Some(input_file_id) = object
        .get("input_file_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            openai_passthrough_metadata("batches"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "message": "input_file_id is required",
                    "type": "invalid_request_error",
                    "code": "missing_parameter",
                }
            }),
        )
        .await;
    };

    let Some(endpoint) = object
        .get("endpoint")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            openai_passthrough_metadata("batches"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "message": "endpoint is required",
                    "type": "invalid_request_error",
                    "code": "missing_parameter",
                }
            }),
        )
        .await;
    };

    let mut route_metadata = openai_passthrough_metadata("batches");
    route_metadata.insert("endpoint_name", endpoint.to_string());
    let openai_base_url = normalize_provider_base_url(state.config.openai_api_url.clone());

    if endpoint != "/v1/chat/completions" {
        return forward_buffered_native_route_with_response_cache(
            state,
            client_addr,
            parts,
            body_bytes,
            "proxy.openai_batch_create",
            "openai_batch_create",
            "openai",
            "batch".to_string(),
            route_metadata,
            openai_base_url,
            "bypassed",
            0.0,
        )
        .await;
    }

    let optimization_started = Instant::now();
    let downloaded = download_openai_batch_file(
        &state,
        client_addr,
        &parts.headers,
        &request_id,
        &openai_base_url,
        input_file_id,
    )
    .await?;
    let Some(file_content) = downloaded else {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            route_metadata,
            StatusCode::NOT_FOUND,
            serde_json::json!({
                "error": {
                    "message": format!("Failed to download file {input_file_id}"),
                    "type": "invalid_request_error",
                    "code": "file_not_found",
                }
            }),
        )
        .await;
    };

    let optimized = optimize_openai_batch_jsonl_content(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &file_content,
    )?;
    if optimized.stats.total_requests == 0 {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            route_metadata,
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "message": "No valid requests found in input file",
                    "type": "invalid_request_error",
                    "code": "empty_file",
                }
            }),
        )
        .await;
    }

    let uploaded_file_id = upload_openai_batch_file(
        &state,
        client_addr,
        &parts.headers,
        &request_id,
        &openai_base_url,
        input_file_id,
        &optimized.content,
    )
    .await?;
    let Some(uploaded_file_id) = uploaded_file_id else {
        return local_json_route(
            state,
            client_addr,
            request_head,
            "proxy.openai_batch_create",
            "openai_batch_create",
            route_metadata,
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({
                "error": {
                    "message": "Failed to upload compressed file",
                    "type": "server_error",
                    "code": "upload_failed",
                }
            }),
        )
        .await;
    };

    let completion_window = object
        .get("completion_window")
        .cloned()
        .unwrap_or_else(|| Value::String("24h".to_string()));
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let batch_body = build_openai_batch_create_body(
        &uploaded_file_id,
        endpoint,
        completion_window,
        metadata,
        input_file_id,
        &optimized.stats,
    );
    let batch_body_bytes = serde_json::to_vec(&batch_body)
        .map(Bytes::from)
        .map_err(|error| ProxyError::InvalidRequest(format!("invalid JSON body: {error}")))?;

    route_metadata.insert("original_file_id", input_file_id.to_string());
    route_metadata.insert("compressed_file_id", uploaded_file_id.clone());
    route_metadata.insert("total_requests", optimized.stats.total_requests.to_string());
    route_metadata.insert("jsonl_errors", optimized.stats.errors.to_string());

    let mut response = forward_buffered_native_route_with_response_cache(
        state,
        client_addr,
        parts,
        batch_body_bytes,
        "proxy.openai_batch_create",
        "openai_batch_create",
        "openai",
        "batch".to_string(),
        route_metadata,
        openai_base_url,
        optimized.compression_status,
        optimization_started.elapsed().as_secs_f64() * 1000.0,
    )
    .await?;
    attach_openai_batch_response_headers(&mut response, &optimized.stats)?;
    Ok(response)
}

async fn forward_native_anthropic_count_tokens(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let request_id = ensure_request_id(&parts.headers);
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let (model, messages_count, system_present) =
        parse_anthropic_count_tokens_request(&body_bytes)?;
    validate_request_array_length("messages", messages_count)?;
    let req = Request::from_parts(parts, Body::from(body_bytes.clone()));

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert("messages_count", messages_count);
    route_metadata.insert("system_present", system_present);

    let anthropic_base_url = normalize_provider_base_url(state.config.anthropic_api_url.clone());
    let response = forward_http_with_route_target(
        state.clone(),
        client_addr,
        req,
        "proxy.anthropic_count_tokens",
        "native_anthropic_count_tokens",
        Some((Some(model.clone()), route_metadata)),
        Some("anthropic"),
        Some(anthropic_base_url),
    )
    .await?;
    log_native_provider_request(
        &state,
        &request_id,
        "anthropic",
        &model,
        "native_anthropic_count_tokens",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(response)
}

async fn forward_buffered_native_route_with_response_cache(
    state: AppState,
    client_addr: SocketAddr,
    parts: http::request::Parts,
    body_bytes: Bytes,
    operation: &'static str,
    route_mode: &'static str,
    provider: &'static str,
    model: String,
    route_metadata: PipelineMetadata,
    upstream_base: url::Url,
    compression_status: &'static str,
    optimization_latency_ms: f64,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let context = ExecutionContext::new(operation, request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());
    let upstream_url = build_upstream_url(&upstream_base, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", upstream_base.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );

    let cache_key = if response_cache_is_enabled(&state.config) {
        Some(hash_response_cache_request(
            &method,
            &uri,
            &parts.headers,
            &body_bytes,
            &upstream_base,
            provider,
        ))
    } else {
        None
    };
    let cached = cache_key.and_then(|key| {
        state
            .response_cache
            .lock()
            .expect("response cache poisoned")
            .get(key)
    });

    match &cached {
        Some(_) => {
            state.metrics.record_response_cache_hit();
            emit_stage(
                &state,
                PipelineStage::InputCached,
                &context,
                [("cache_status", "hit".to_string())],
            );
        }
        None if cache_key.is_some() => {
            state.metrics.record_response_cache_miss();
            emit_stage(
                &state,
                PipelineStage::InputCached,
                &context,
                [("cache_status", "miss".to_string())],
            );
        }
        None => {
            emit_stage(
                &state,
                PipelineStage::InputCached,
                &context,
                [("cache_status", "disabled".to_string())],
            );
        }
    }

    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", route_mode);
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", compression_status.to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );

    if let Some(cached) = cached {
        emit_stage(
            &state,
            PipelineStage::PreSend,
            &context,
            [
                ("upstream_url", "cache://response-cache".to_string()),
                ("provider_bypass", "true".to_string()),
            ],
        );
        let status = cached.status;
        let response = cached_response_to_http(cached, &request_id).inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;
        state
            .metrics
            .record_request_completed(status.as_u16(), start.elapsed().as_secs_f64());
        emit_stage(
            &state,
            PipelineStage::PostSend,
            &context,
            [
                ("upstream_status", status.as_u16().to_string()),
                ("upstream_url", "cache://response-cache".to_string()),
                ("response_source", "cache".to_string()),
            ],
        );
        emit_stage(
            &state,
            PipelineStage::ResponseReceived,
            &context,
            [
                ("response_status", status.as_u16().to_string()),
                ("latency_ms", start.elapsed().as_millis().to_string()),
                ("response_source", "cache".to_string()),
            ],
        );
        log_native_provider_request(
            &state,
            &request_id,
            provider,
            &model,
            route_mode,
            &body_bytes,
            true,
            optimization_latency_ms,
            0.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        return Ok(response);
    }

    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("provider_bypass", "false".to_string()),
        ],
    );

    let primary = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        body_bytes.clone(),
        &request_id,
        &upstream_base,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;

    if let Some(key) = cache_key.filter(|_| primary.status.is_success()) {
        state
            .response_cache
            .lock()
            .expect("response cache poisoned")
            .put(
                key,
                primary.status,
                primary.headers.clone(),
                primary.body.clone(),
            );
    }

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("response_source", "provider".to_string()),
        ],
    );

    let status = primary.status;
    let ttfb_ms = primary.ttfb_ms;
    let response = buffered_response_to_http(primary, &request_id).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(status.as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", status.as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("response_source", "provider".to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        provider,
        &model,
        route_mode,
        &body_bytes,
        false,
        optimization_latency_ms,
        ttfb_ms,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(response)
}

async fn forward_native_gemini_generate_content(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let request_id = ensure_request_id(&parts.headers);
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let model = gemini_model_from_path(parts.uri.path())
        .ok_or_else(|| ProxyError::InvalidRequest("invalid Gemini model path".to_string()))?;
    let (contents_count, system_instruction_present) =
        parse_gemini_generate_content_request(&body_bytes)?;
    validate_request_array_length("contents", contents_count)?;
    let streaming = is_gemini_streaming_query(parts.uri.query())
        || is_gemini_stream_generate_content_path(parts.uri.path());
    let optimized = optimize_gemini_generate_content_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
    )?;
    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert(
        "shadow_mode",
        if state.config.native_gemini_generate_content_shadow {
            "compare"
        } else {
            "disabled"
        },
    );
    route_metadata.insert("contents_count", contents_count);
    route_metadata.insert("system_instruction_present", system_instruction_present);
    route_metadata.insert("stream", streaming);

    if state.config.native_gemini_generate_content_shadow {
        return forward_native_gemini_generate_content_with_shadow(
            state,
            client_addr,
            req,
            model,
            route_metadata,
            streaming,
        )
        .await;
    }

    let gemini_base_url = normalize_provider_base_url(state.config.gemini_api_url.clone());
    let response = forward_http_with_route_target_with_compression(
        state.clone(),
        client_addr,
        req,
        "proxy.gemini_generate_content",
        "native_gemini_generate_content",
        Some((Some(model.clone()), route_metadata)),
        Some("gemini"),
        Some(gemini_base_url),
        optimized.compression_status,
    )
    .await?;
    log_native_provider_request(
        &state,
        &request_id,
        "gemini",
        &model,
        "native_gemini_generate_content",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(response)
}

async fn forward_native_gemini_count_tokens(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let request_id = ensure_request_id(&parts.headers);
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let model = gemini_model_from_suffix(parts.uri.path(), ":countTokens")
        .ok_or_else(|| ProxyError::InvalidRequest("invalid Gemini model path".to_string()))?;
    let (contents_count, system_instruction_present) =
        parse_gemini_generate_content_request(&body_bytes)?;
    validate_request_array_length("contents", contents_count)?;
    let optimized = optimize_gemini_count_tokens_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
    )?;
    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert(
        "shadow_mode",
        if state.config.native_gemini_count_tokens_shadow {
            "compare"
        } else {
            "disabled"
        },
    );
    route_metadata.insert("contents_count", contents_count);
    route_metadata.insert("system_instruction_present", system_instruction_present);

    if state.config.native_gemini_count_tokens_shadow {
        return forward_native_gemini_count_tokens_with_shadow(
            state,
            client_addr,
            req,
            body_bytes,
            optimized.compression_status,
            model,
            route_metadata,
        )
        .await;
    }

    let gemini_base_url = normalize_provider_base_url(state.config.gemini_api_url.clone());
    let response = forward_http_with_route_target_with_compression(
        state.clone(),
        client_addr,
        req,
        "proxy.gemini_count_tokens",
        "native_gemini_count_tokens",
        Some((Some(model.clone()), route_metadata)),
        Some("gemini"),
        Some(gemini_base_url),
        optimized.compression_status,
    )
    .await?;
    log_native_provider_request(
        &state,
        &request_id,
        "gemini",
        &model,
        "native_gemini_count_tokens",
        &optimized.primary_body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(response)
}

async fn forward_native_google_cloudcode_stream(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let (parts, body) = req.into_parts();
    validate_request_content_length(&parts.headers, state.config.max_body_bytes)?;
    let request_id = ensure_request_id(&parts.headers);
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let (model, request_contents_count, antigravity) =
        parse_google_cloudcode_stream_request(&parts.headers, &body_bytes)?;
    validate_request_array_length("request.contents", request_contents_count)?;
    let optimized = optimize_google_cloudcode_stream_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
        antigravity,
    )?;
    let req = Request::from_parts(parts, Body::from(optimized.primary_body_bytes.clone()));

    let mut route_metadata = PipelineMetadata::new();
    route_metadata.insert("native_route", true);
    route_metadata.insert(
        "shadow_mode",
        if state.config.native_google_cloudcode_stream_shadow {
            "compare"
        } else {
            "disabled"
        },
    );
    route_metadata.insert("request_contents_count", request_contents_count);
    route_metadata.insert("antigravity", antigravity);
    route_metadata.insert("stream", true);

    if state.config.native_google_cloudcode_stream_shadow {
        return forward_native_google_cloudcode_stream_with_shadow(
            state,
            client_addr,
            req,
            model,
            route_metadata,
            antigravity,
        )
        .await;
    }

    let cloudcode_base_url =
        resolve_cloudcode_base_url(antigravity, &state.config.cloudcode_api_url);
    let response = forward_http_with_route_target_with_compression(
        state.clone(),
        client_addr,
        req,
        "proxy.google_cloudcode_stream",
        "native_google_cloudcode_stream",
        Some((Some(model.clone()), route_metadata)),
        Some("cloudcode"),
        Some(cloudcode_base_url),
        optimized.compression_status,
    )
    .await?;
    log_native_provider_request(
        &state,
        &request_id,
        "cloudcode",
        &model,
        "native_google_cloudcode_stream",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(response)
}

async fn forward_native_anthropic_messages_with_shadow(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    original_body_bytes: Bytes,
    compression_status: &'static str,
    model: String,
    route_metadata: PipelineMetadata,
    streaming: bool,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?
        .to_bytes();
    let anthropic_base_url = normalize_provider_base_url(state.config.anthropic_api_url.clone());
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new("proxy.anthropic_messages", request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_url = build_upstream_url(&anthropic_base_url, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", anthropic_base_url.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", "native_anthropic_messages");
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", compression_status.to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("shadow_mode", "compare".to_string()),
        ],
    );

    if streaming {
        let primary = execute_streaming_forward(
            &state,
            client_addr,
            &method,
            &uri,
            &parts.headers,
            body_bytes.clone(),
            &request_id,
            &anthropic_base_url,
        )
        .await
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

        emit_stage(
            &state,
            PipelineStage::PostSend,
            &context,
            [
                ("upstream_status", primary.status.as_u16().to_string()),
                ("upstream_url", upstream_url.to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );

        let response = streaming_response_to_http_with_shadow(
            primary,
            &request_id,
            state.clone(),
            ShadowProvider::Anthropic,
            client_addr,
            method.clone(),
            uri.clone(),
            parts.headers.clone(),
            original_body_bytes.clone(),
        )
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;
        state
            .metrics
            .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
        emit_stage(
            &state,
            PipelineStage::ResponseReceived,
            &context,
            [
                ("response_status", response.status().as_u16().to_string()),
                ("latency_ms", start.elapsed().as_millis().to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );
        log_native_provider_request(
            &state,
            &request_id,
            "anthropic",
            &model,
            "native_anthropic_messages",
            &body_bytes,
            false,
            0.0,
            0.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        return Ok(response);
    }

    let primary = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        body_bytes.clone(),
        &request_id,
        &anthropic_base_url,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    let shadow = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        original_body_bytes.clone(),
        &request_id,
        &state.config.upstream,
    )
    .await;

    let shadow_matched = match shadow {
        Ok(ref shadow_response) => responses_match(&primary, shadow_response),
        Err(_) => false,
    };
    state
        .metrics
        .record_anthropic_shadow_comparison(shadow_matched);

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );

    let response = buffered_response_to_http(primary, &request_id).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", response.status().as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        "anthropic",
        &model,
        "native_anthropic_messages",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(response)
}

async fn native_compress_messages(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let bypass = req
        .headers()
        .get("x-headroom-bypass")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    if let Err(error) = validate_request_content_length(&parts.headers, state.config.max_body_bytes)
    {
        return local_json_route_with_compression(
            state,
            client_addr,
            Request::from_parts(parts, Body::empty()),
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::PAYLOAD_TOO_LARGE,
            serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": error.to_string(),
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await
        .map_err(ProxyError::from);
    }
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)?
        .to_bytes();
    let req = Request::from_parts(parts, Body::empty());

    if bypass {
        return match serde_json::from_slice::<Value>(&body_bytes) {
            Ok(body) => {
                let messages = body
                    .as_object()
                    .and_then(|object| object.get("messages"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([]));
                local_json_route_with_compression(
                    state,
                    client_addr,
                    req,
                    "proxy.compress_messages",
                    "compress_messages",
                    product_passthrough_metadata("compress"),
                    StatusCode::OK,
                    serde_json::json!({
                        "messages": messages,
                        "tokens_before": 0,
                        "tokens_after": 0,
                        "tokens_saved": 0,
                        "compression_ratio": 1.0,
                        "transforms_applied": [],
                        "ccr_hashes": [],
                    }),
                    [("compression_status", "bypassed".to_string())],
                )
                .await
            }
            Err(error) => {
                local_json_route_with_compression(
                    state,
                    client_addr,
                    req,
                    "proxy.compress_messages",
                    "compress_messages",
                    product_passthrough_metadata("compress"),
                    StatusCode::BAD_REQUEST,
                    serde_json::json!({
                        "error": format!("Invalid request body: {error}"),
                    }),
                    [("compression_status", "bypassed".to_string())],
                )
                .await
            }
        };
    }

    let payload = match serde_json::from_slice::<Value>(&body_bytes) {
        Ok(payload) => payload,
        Err(_) => {
            return local_json_route_with_compression(
                state,
                client_addr,
                req,
                "proxy.compress_messages",
                "compress_messages",
                product_passthrough_metadata("compress"),
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "error": {
                        "type": "invalid_request",
                        "message": "Invalid JSON in request body.",
                    }
                }),
                [("compression_status", "failed".to_string())],
            )
            .await;
        }
    };

    let Some(object) = payload.as_object() else {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Missing required field: messages",
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await;
    };

    let Some(messages_value) = object.get("messages") else {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Missing required field: messages",
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await;
    };

    let Some(model) = object.get("model").and_then(Value::as_str) else {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": "Missing required field: model",
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await;
    };

    let Some(messages) = messages_value.as_array() else {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({
                "error": {
                    "type": "compression_error",
                    "message": "messages must be an array",
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await;
    };
    if messages.len() > MAX_REQUEST_ARRAY_LENGTH {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": ProxyError::RequestArrayTooLarge {
                        field: "messages",
                        actual_length: messages.len(),
                        max_length: MAX_REQUEST_ARRAY_LENGTH,
                    }
                    .to_string(),
                }
            }),
            [("compression_status", "failed".to_string())],
        )
        .await;
    }

    if messages.is_empty() {
        return local_json_route_with_compression(
            state,
            client_addr,
            req,
            "proxy.compress_messages",
            "compress_messages",
            product_passthrough_metadata("compress"),
            StatusCode::OK,
            serde_json::json!({
                "messages": [],
                "tokens_before": 0,
                "tokens_after": 0,
                "tokens_saved": 0,
                "compression_ratio": 1.0,
                "transforms_applied": [],
                "ccr_hashes": [],
            }),
            [("compression_status", "bypassed".to_string())],
        )
        .await;
    }

    match build_compress_response(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        object,
        messages,
        model,
    ) {
        Ok(result) => {
            log_native_compress_request(
                &state,
                &request_id,
                model,
                &result.body,
                result.cache_hit,
                start.elapsed().as_secs_f64() * 1000.0,
            );
            let compression_status = if result.body["tokens_saved"].as_u64().unwrap_or(0) > 0 {
                "compressed"
            } else {
                "unchanged"
            };
            let tokens_before = result.body["tokens_before"]
                .as_u64()
                .unwrap_or(0)
                .to_string();
            let tokens_after = result.body["tokens_after"]
                .as_u64()
                .unwrap_or(0)
                .to_string();
            let transform_count = result.body["transforms_applied"]
                .as_array()
                .map(|items| count_logical_transforms(items))
                .unwrap_or(0)
                .to_string();
            local_json_route_with_compression(
                state,
                client_addr,
                req,
                "proxy.compress_messages",
                "compress_messages",
                product_passthrough_metadata("compress"),
                StatusCode::OK,
                result.body,
                [
                    ("compression_status", compression_status.to_string()),
                    ("tokens_before", tokens_before),
                    ("tokens_after", tokens_after),
                    ("transform_count", transform_count),
                ],
            )
            .await
        }
        Err(message) => {
            local_json_route_with_compression(
                state,
                client_addr,
                req,
                "proxy.compress_messages",
                "compress_messages",
                product_passthrough_metadata("compress"),
                StatusCode::SERVICE_UNAVAILABLE,
                serde_json::json!({
                    "error": {
                        "type": "compression_error",
                        "message": message,
                    }
                }),
                [("compression_status", "failed".to_string())],
            )
            .await
        }
    }
}

async fn forward_native_openai_chat_with_shadow(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    shadow_body_bytes: Bytes,
    compression_status: &'static str,
    model: String,
    route_metadata: PipelineMetadata,
    streaming: bool,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let primary_body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?
        .to_bytes();
    let openai_base_url =
        select_openai_chat_base_url(&parts.headers, &state.config.openai_api_url)?;
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new("proxy.openai_chat", request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_url = build_upstream_url(&openai_base_url, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", openai_base_url.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", "native_openai_chat");
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", compression_status.to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("shadow_mode", "compare".to_string()),
        ],
    );

    if streaming {
        let primary = execute_streaming_forward(
            &state,
            client_addr,
            &method,
            &uri,
            &parts.headers,
            primary_body_bytes.clone(),
            &request_id,
            &openai_base_url,
        )
        .await
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

        emit_stage(
            &state,
            PipelineStage::PostSend,
            &context,
            [
                ("upstream_status", primary.status.as_u16().to_string()),
                ("upstream_url", upstream_url.to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );

        let response = streaming_response_to_http_with_shadow(
            primary,
            &request_id,
            state.clone(),
            ShadowProvider::OpenAi,
            client_addr,
            method.clone(),
            uri.clone(),
            parts.headers.clone(),
            shadow_body_bytes.clone(),
        )
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;
        state
            .metrics
            .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
        emit_stage(
            &state,
            PipelineStage::ResponseReceived,
            &context,
            [
                ("response_status", response.status().as_u16().to_string()),
                ("latency_ms", start.elapsed().as_millis().to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );
        log_native_provider_request(
            &state,
            &request_id,
            "openai",
            &model,
            "native_openai_chat",
            &primary_body_bytes,
            false,
            0.0,
            0.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        return Ok(response);
    }

    let primary = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        primary_body_bytes.clone(),
        &request_id,
        &openai_base_url,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    let shadow = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        shadow_body_bytes.clone(),
        &request_id,
        &state.config.upstream,
    )
    .await;

    let shadow_matched = match shadow {
        Ok(ref shadow_response) => responses_match(&primary, shadow_response),
        Err(_) => false,
    };
    state
        .metrics
        .record_openai_shadow_comparison(shadow_matched);

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );

    let response = buffered_response_to_http(primary, &request_id).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", response.status().as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        "openai",
        &model,
        "native_openai_chat",
        &primary_body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(response)
}

async fn forward_native_gemini_generate_content_with_shadow(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    model: String,
    route_metadata: PipelineMetadata,
    streaming: bool,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?
        .to_bytes();
    let optimized = optimize_gemini_generate_content_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
    )?;
    let gemini_base_url = normalize_provider_base_url(state.config.gemini_api_url.clone());
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new("proxy.gemini_generate_content", request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_url = build_upstream_url(&gemini_base_url, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", gemini_base_url.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", "native_gemini_generate_content");
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [(
            "compression_status",
            optimized.compression_status.to_string(),
        )],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("shadow_mode", "compare".to_string()),
        ],
    );

    if streaming {
        let primary = execute_streaming_forward(
            &state,
            client_addr,
            &method,
            &uri,
            &parts.headers,
            optimized.primary_body_bytes.clone(),
            &request_id,
            &gemini_base_url,
        )
        .await
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

        emit_stage(
            &state,
            PipelineStage::PostSend,
            &context,
            [
                ("upstream_status", primary.status.as_u16().to_string()),
                ("upstream_url", upstream_url.to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );

        let response = streaming_response_to_http_with_shadow(
            primary,
            &request_id,
            state.clone(),
            ShadowProvider::Gemini,
            client_addr,
            method.clone(),
            uri.clone(),
            parts.headers.clone(),
            body_bytes.clone(),
        )
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;
        state
            .metrics
            .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
        emit_stage(
            &state,
            PipelineStage::ResponseReceived,
            &context,
            [
                ("response_status", response.status().as_u16().to_string()),
                ("latency_ms", start.elapsed().as_millis().to_string()),
                ("shadow_match", "pending".to_string()),
            ],
        );
        log_native_provider_request(
            &state,
            &request_id,
            "gemini",
            &model,
            "native_gemini_generate_content",
            &body_bytes,
            false,
            0.0,
            0.0,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        return Ok(response);
    }

    let primary = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        optimized.primary_body_bytes.clone(),
        &request_id,
        &gemini_base_url,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    let shadow = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        body_bytes.clone(),
        &request_id,
        &state.config.upstream,
    )
    .await;

    let shadow_matched = match shadow {
        Ok(ref shadow_response) => responses_match(&primary, shadow_response),
        Err(_) => false,
    };
    state
        .metrics
        .record_gemini_shadow_comparison(shadow_matched);

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );

    let response = buffered_response_to_http(primary, &request_id).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", response.status().as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        "gemini",
        &model,
        "native_gemini_generate_content",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(response)
}

async fn forward_native_gemini_count_tokens_with_shadow(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    shadow_body_bytes: Bytes,
    compression_status: &'static str,
    model: String,
    route_metadata: PipelineMetadata,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let primary_body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?
        .to_bytes();
    let gemini_base_url = normalize_provider_base_url(state.config.gemini_api_url.clone());
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new("proxy.gemini_count_tokens", request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_url = build_upstream_url(&gemini_base_url, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", gemini_base_url.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", "native_gemini_count_tokens");
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", compression_status.to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("shadow_mode", "compare".to_string()),
        ],
    );

    let primary = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        primary_body_bytes.clone(),
        &request_id,
        &gemini_base_url,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    let shadow = execute_buffered_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        shadow_body_bytes.clone(),
        &request_id,
        &state.config.upstream,
    )
    .await;

    let shadow_matched = match shadow {
        Ok(ref shadow_response) => responses_match(&primary, shadow_response),
        Err(_) => false,
    };
    state
        .metrics
        .record_gemini_count_tokens_shadow_comparison(shadow_matched);

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );

    let response = buffered_response_to_http(primary, &request_id).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", response.status().as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("shadow_match", shadow_matched.to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        "gemini",
        &model,
        "native_gemini_count_tokens",
        &primary_body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(response)
}

async fn forward_native_google_cloudcode_stream_with_shadow(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    model: String,
    route_metadata: PipelineMetadata,
    antigravity: bool,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();

    let (parts, body) = req.into_parts();
    let request_id = ensure_request_id(&parts.headers);
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let path_for_log = uri.path().to_string();
    let body_bytes = body
        .collect()
        .await
        .map_err(std::io::Error::other)
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?
        .to_bytes();
    let optimized = optimize_google_cloudcode_stream_body(
        &state.compression_cache,
        &state.product,
        &state.telemetry,
        &body_bytes,
        &model,
        antigravity,
    )?;
    let cloudcode_base_url =
        resolve_cloudcode_base_url(antigravity, &state.config.cloudcode_api_url);
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new("proxy.google_cloudcode_stream", request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider)
        .with_model(model.clone());

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_url = build_upstream_url(&cloudcode_base_url, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", cloudcode_base_url.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    let forwarded_host = parts
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    let mut routed_metadata = route_metadata.clone();
    routed_metadata.insert("route_mode", "native_google_cloudcode_stream");
    emit_stage_metadata(
        &state,
        PipelineStage::InputRouted,
        &context,
        routed_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [(
            "compression_status",
            optimized.compression_status.to_string(),
        )],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("shadow_mode", "compare".to_string()),
        ],
    );

    let primary = execute_streaming_forward(
        &state,
        client_addr,
        &method,
        &uri,
        &parts.headers,
        optimized.primary_body_bytes.clone(),
        &request_id,
        &cloudcode_base_url,
    )
    .await
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", primary.status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("shadow_match", "pending".to_string()),
        ],
    );

    let response = streaming_response_to_http_with_shadow(
        primary,
        &request_id,
        state.clone(),
        ShadowProvider::GoogleCloudCodeStream,
        client_addr,
        method.clone(),
        uri.clone(),
        parts.headers.clone(),
        body_bytes.clone(),
    )
    .inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    state
        .metrics
        .record_request_completed(response.status().as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", response.status().as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
            ("shadow_match", "pending".to_string()),
        ],
    );
    log_native_provider_request(
        &state,
        &request_id,
        "cloudcode",
        &model,
        "native_google_cloudcode_stream",
        &body_bytes,
        false,
        0.0,
        0.0,
        start.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(response)
}

async fn forward_http_with_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: Option<(Option<String>, PipelineMetadata)>,
) -> Result<Response<Body>, ProxyError> {
    forward_http_with_route_target(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        route_details,
        None,
        None,
    )
    .await
}

async fn forward_http_with_route_provider(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: Option<(Option<String>, PipelineMetadata)>,
    provider_override: Option<&'static str>,
) -> Result<Response<Body>, ProxyError> {
    forward_http_with_route_target(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        route_details,
        provider_override,
        None,
    )
    .await
}

async fn local_json_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: PipelineMetadata,
    status: StatusCode,
    payload: Value,
) -> Result<Response<Body>, ProxyError> {
    local_json_route_with_compression(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        route_details,
        status,
        payload,
        [("compression_status", "bypassed".to_string())],
    )
    .await
}

async fn local_json_route_with_compression<const N: usize>(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: PipelineMetadata,
    status: StatusCode,
    payload: Value,
    compression_metadata: [(&str, String); N],
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new(operation, request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider);
    let local_url = format!("local://headroom-proxy{}", uri);

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", "local".to_string()),
            ("upstream_url", local_url.clone()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                req.headers().contains_key(http::header::HOST).to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRouted,
        &context,
        [("route_mode", route_mode.to_string())],
    );
    emit_stage_metadata(&state, PipelineStage::InputRouted, &context, route_details);
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        compression_metadata,
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", local_url.clone()),
            ("forwarded_headers", "0".to_string()),
        ],
    );

    let body = serde_json::to_vec(&payload)
        .map_err(|err| {
            ProxyError::InvalidRequest(format!("failed to serialize local response: {err}"))
        })
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", status.as_u16().to_string()),
            ("upstream_url", local_url.clone()),
        ],
    );

    let mut response = Response::builder().status(status);
    {
        let headers = response.headers_mut().expect("builder has headers");
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            headers.insert(HeaderName::from_static("x-request-id"), value);
        }
    }
    let response = response
        .body(Body::from(body))
        .map_err(|err| ProxyError::InvalidHeader(err.to_string()))
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

    state
        .metrics
        .record_request_completed(status.as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", status.as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
        ],
    );

    Ok(response)
}

async fn local_text_route(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: PipelineMetadata,
    status: StatusCode,
    content_type: &'static str,
    extra_header: Option<(&'static str, String)>,
    body_text: String,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();
    let provider = provider_from_path(&path_for_log);
    let context = ExecutionContext::new(operation, request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider);
    let local_url = format!("local://headroom-proxy{}", uri);

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", "local".to_string()),
            ("upstream_url", local_url.clone()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                req.headers().contains_key(http::header::HOST).to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRouted,
        &context,
        [("route_mode", route_mode.to_string())],
    );
    emit_stage_metadata(&state, PipelineStage::InputRouted, &context, route_details);
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", "bypassed".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", local_url.clone()),
            ("forwarded_headers", "0".to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", status.as_u16().to_string()),
            ("upstream_url", local_url.clone()),
        ],
    );

    let mut response = Response::builder().status(status);
    {
        let headers = response.headers_mut().expect("builder has headers");
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static(content_type),
        );
        if let Some((name, value)) = extra_header {
            headers.insert(
                HeaderName::from_static(name),
                HeaderValue::from_str(&value)
                    .map_err(|error| ProxyError::InvalidHeader(error.to_string()))?,
            );
        }
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            headers.insert(HeaderName::from_static("x-request-id"), value);
        }
    }
    let response = response
        .body(Body::from(body_text))
        .map_err(|err| ProxyError::InvalidHeader(err.to_string()))?;

    state
        .metrics
        .record_request_completed(status.as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", status.as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
        ],
    );

    Ok(response)
}

async fn forward_http_with_route_target(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: Option<(Option<String>, PipelineMetadata)>,
    provider_override: Option<&'static str>,
    upstream_base_override: Option<url::Url>,
) -> Result<Response<Body>, ProxyError> {
    forward_http_with_route_target_with_compression(
        state,
        client_addr,
        req,
        operation,
        route_mode,
        route_details,
        provider_override,
        upstream_base_override,
        "bypassed",
    )
    .await
}

async fn forward_http_with_route_target_with_compression(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
    operation: &'static str,
    route_mode: &'static str,
    route_details: Option<(Option<String>, PipelineMetadata)>,
    provider_override: Option<&'static str>,
    upstream_base_override: Option<url::Url>,
    compression_status: &'static str,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    state.metrics.record_request_started();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();
    let provider = provider_override.unwrap_or_else(|| provider_from_path(&path_for_log));
    let mut context = ExecutionContext::new(operation, request_id.clone())
        .with_component("headroom-proxy")
        .with_provider(provider);
    let routed_details = route_details
        .as_ref()
        .map(|(_, metadata)| metadata.clone())
        .unwrap_or_default();
    if let Some((Some(model), _)) = &route_details {
        context = context.with_model(model.clone());
    }

    emit_stage(
        &state,
        PipelineStage::Setup,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("protocol", "http".to_string()),
        ],
    );

    let upstream_base = upstream_base_override
        .as_ref()
        .unwrap_or(&state.config.upstream);
    let upstream_url = build_upstream_url(upstream_base, &uri).inspect_err(|_| {
        state
            .metrics
            .record_request_failed(start.elapsed().as_secs_f64());
    })?;
    emit_stage(
        &state,
        PipelineStage::PreStart,
        &context,
        [
            ("upstream_base", upstream_base.to_string()),
            ("upstream_url", upstream_url.to_string()),
            ("rewrite_host", state.config.rewrite_host.to_string()),
        ],
    );

    // Forwarded-Host: prefer client's Host. Forwarded-Proto: assume http for
    // now (we don't terminate TLS in this binary; if a TLS terminator is in
    // front, it should rewrite this — which we'd handle by not overwriting
    // an existing one in a future change).
    let forwarded_host = req
        .headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    emit_stage(
        &state,
        PipelineStage::PostStart,
        &context,
        [
            (
                "forwarded_host_present",
                forwarded_host.is_some().to_string(),
            ),
            ("client_addr", client_addr.to_string()),
            ("max_body_bytes", state.config.max_body_bytes.to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputReceived,
        &context,
        [
            ("method", method.to_string()),
            ("path", path_for_log.clone()),
            ("query_present", uri.query().is_some().to_string()),
        ],
    );
    emit_stage(
        &state,
        PipelineStage::InputCached,
        &context,
        [("cache_status", "not_configured".to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRouted,
        &context,
        [("route_mode", route_mode.to_string())],
    );
    if !routed_details.is_empty() {
        emit_stage_metadata(&state, PipelineStage::InputRouted, &context, routed_details);
    }
    emit_stage(
        &state,
        PipelineStage::InputCompressed,
        &context,
        [("compression_status", compression_status.to_string())],
    );
    emit_stage(
        &state,
        PipelineStage::InputRemembered,
        &context,
        [("memory_status", "disabled".to_string())],
    );

    // Build the outgoing headers off the incoming ones, then optionally drop
    // Host (rewrite_host=true => let reqwest set its own Host for the upstream).
    let mut outgoing_headers = build_forward_request_headers(
        req.headers(),
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        &request_id,
    );
    if !state.config.rewrite_host {
        if let Some(h) = req.headers().get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }
    emit_stage(
        &state,
        PipelineStage::PreSend,
        &context,
        [
            ("upstream_url", upstream_url.to_string()),
            ("forwarded_headers", outgoing_headers.len().to_string()),
        ],
    );

    // Stream the request body through to reqwest. We don't buffer.
    let body_stream =
        TryStreamExt::map_err(req.into_body().into_data_stream(), std::io::Error::other);
    let reqwest_body = reqwest::Body::wrap_stream(body_stream);

    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;
    let upstream_resp = state
        .client
        .request(reqwest_method, upstream_url.clone())
        .headers(outgoing_headers)
        .body(reqwest_body)
        .send()
        .await
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

    let upstream_status = upstream_resp.status();
    emit_stage(
        &state,
        PipelineStage::PostSend,
        &context,
        [
            ("upstream_status", upstream_status.as_u16().to_string()),
            ("upstream_url", upstream_url.to_string()),
        ],
    );
    let status = StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let resp_headers = filter_response_headers(upstream_resp.headers());

    // Stream response body back without buffering. Wrap errors so mid-stream
    // upstream failures are logged rather than silently truncating the client.
    let rid = request_id.clone();
    let resp_stream = upstream_resp.bytes_stream().map(move |r| match r {
        Ok(b) => Ok(b),
        Err(e) => {
            tracing::warn!(request_id = %rid, error = %e, "upstream stream error mid-response");
            Err(e)
        }
    });
    let body = Body::from_stream(resp_stream);

    let mut response = Response::builder().status(status);
    {
        let h = response.headers_mut().expect("builder has headers");
        h.extend(resp_headers);
        // Echo X-Request-Id back to the client.
        if let Ok(v) = http::HeaderValue::from_str(&request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
    }
    let response = response
        .body(body)
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))
        .inspect_err(|_| {
            state
                .metrics
                .record_request_failed(start.elapsed().as_secs_f64());
        })?;

    tracing::info!(
        request_id = %request_id,
        method = %method,
        path = %path_for_log,
        upstream_status = upstream_status.as_u16(),
        latency_ms = start.elapsed().as_millis() as u64,
        protocol = "http",
        "forwarded"
    );
    state
        .metrics
        .record_request_completed(status.as_u16(), start.elapsed().as_secs_f64());
    emit_stage(
        &state,
        PipelineStage::ResponseReceived,
        &context,
        [
            ("response_status", status.as_u16().to_string()),
            ("latency_ms", start.elapsed().as_millis().to_string()),
        ],
    );

    Ok(response)
}

fn emit_stage<const N: usize>(
    state: &AppState,
    stage: PipelineStage,
    context: &ExecutionContext,
    metadata: [(&str, String); N],
) {
    let metadata = metadata.into_iter().collect::<PipelineMetadata>();
    emit_stage_metadata(state, stage, context, metadata);
}

fn emit_stage_metadata(
    state: &AppState,
    stage: PipelineStage,
    context: &ExecutionContext,
    metadata: PipelineMetadata,
) {
    state.runtime.emit(stage, context.clone(), metadata);
}

fn select_openai_chat_base_url(
    headers: &HeaderMap,
    default_base: &url::Url,
) -> Result<url::Url, ProxyError> {
    let azure_base = headers
        .get("x-headroom-base-url")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if headers.contains_key("api-key") {
        if let Some(base) = azure_base {
            let parsed = url::Url::parse(base).map_err(|e| {
                ProxyError::InvalidUpstream(format!("invalid x-headroom-base-url: {e}"))
            })?;
            return Ok(normalize_provider_base_url(parsed));
        }
    }
    Ok(normalize_provider_base_url(default_base.clone()))
}

fn normalize_provider_base_url(mut url: url::Url) -> url::Url {
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

fn resolve_cloudcode_base_url(is_antigravity: bool, default_base: &url::Url) -> url::Url {
    if is_antigravity {
        return url::Url::parse(ANTIGRAVITY_DAILY_API_URL)
            .expect("default antigravity cloudcode URL must be valid");
    }
    normalize_provider_base_url(default_base.clone())
}

fn provider_target_base_url(
    provider: &'static str,
    headers: &HeaderMap,
    config: &Config,
) -> Result<url::Url, ProxyError> {
    match provider {
        "openai" => select_openai_chat_base_url(headers, &config.openai_api_url),
        "anthropic" => Ok(normalize_provider_base_url(
            config.anthropic_api_url.clone(),
        )),
        "gemini" => Ok(normalize_provider_base_url(config.gemini_api_url.clone())),
        "cloudcode" => Ok(normalize_provider_base_url(
            config.cloudcode_api_url.clone(),
        )),
        _ => Ok(normalize_provider_base_url(config.upstream.clone())),
    }
}

fn uri_query_param(uri: &Uri, key: &str) -> Option<String> {
    uri.query().and_then(|query| {
        url::form_urlencoded::parse(query.as_bytes())
            .find(|(query_key, _)| query_key == key)
            .map(|(_, value)| value.into_owned())
    })
}

fn input_cost_per_token(model: &str) -> Option<f64> {
    let normalized = model
        .strip_prefix("openai/")
        .or_else(|| model.strip_prefix("anthropic/"))
        .or_else(|| model.strip_prefix("google/"))
        .unwrap_or(model);

    if normalized.starts_with("gpt-4o-mini") {
        return Some(0.15 / 1_000_000.0);
    }
    if normalized.starts_with("gpt-4o") {
        return Some(2.50 / 1_000_000.0);
    }
    if normalized.starts_with("claude-haiku-4-5") {
        return Some(0.80 / 1_000_000.0);
    }
    if normalized.starts_with("claude-opus-4") {
        return Some(15.00 / 1_000_000.0);
    }
    if normalized.starts_with("claude-sonnet-4-6")
        || normalized.starts_with("claude-sonnet-4-5")
        || normalized.starts_with("claude-sonnet-4-")
        || normalized.starts_with("claude-3-5-sonnet")
        || normalized.starts_with("claude-3.5-sonnet")
    {
        return Some(3.00 / 1_000_000.0);
    }
    if normalized.starts_with("gemini-1.5-pro") {
        return Some(1.25 / 1_000_000.0);
    }

    None
}

fn estimate_compression_savings_usd(model: &str, tokens_saved: u64) -> f64 {
    input_cost_per_token(model)
        .map(|cost| tokens_saved as f64 * cost)
        .unwrap_or(0.0)
}

fn estimate_input_cost_usd(model: &str, input_tokens: u64) -> f64 {
    input_cost_per_token(model)
        .map(|cost| input_tokens as f64 * cost)
        .unwrap_or(0.0)
}

fn empty_prefix_cache_stats() -> Value {
    serde_json::json!({
        "by_provider": {},
        "totals": {
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_write_5m_tokens": 0,
            "cache_write_1h_tokens": 0,
            "cache_write_5m_requests": 0,
            "cache_write_1h_requests": 0,
            "uncached_input_tokens": 0,
            "requests": 0,
            "hit_requests": 0,
            "hit_rate": 0.0,
            "request_hit_rate": 0.0,
            "bust_count": 0,
            "bust_write_tokens": 0,
            "savings_usd": 0.0,
            "write_premium_usd": 0.0,
            "net_savings_usd": 0.0,
            "observed_ttl_buckets": {
                "5m": {"tokens": 0, "requests": 0},
                "1h": {"tokens": 0, "requests": 0},
            },
            "observed_ttl_mix": {
                "5m_pct": 0.0,
                "1h_pct": 0.0,
                "active_buckets": [],
            },
        },
        "prefix_freeze": {
            "busts_avoided": 0,
            "tokens_preserved": 0,
            "compression_foregone_tokens": 0,
            "net_benefit_tokens": 0,
        },
        "compression_vs_cache": {
            "tokens_saved_by_compression": 0,
            "tokens_lost_to_cache_bust": 0,
            "cache_bust_count": 0,
            "net_tokens": 0,
        },
    })
}

fn build_stats_payload(state: &AppState) -> Value {
    let metrics = state.metrics.snapshot();
    let request_log_stats = aggregate_request_log_stats(
        &state.request_logs.snapshot(),
        state.request_logs.storage_path().as_deref(),
    );
    let prefix_cache_stats = empty_prefix_cache_stats();
    let prefix_cache_savings_usd = prefix_cache_stats["totals"]["net_savings_usd"]
        .as_f64()
        .unwrap_or(0.0);
    let compression_cache_stats = state
        .compression_cache
        .lock()
        .expect("compression cache poisoned")
        .stats();
    let response_cache_stats = state
        .response_cache
        .lock()
        .expect("response cache poisoned")
        .stats();
    let avg_latency_ms = if metrics.request_duration_count > 0 {
        metrics.request_duration_millis_total as f64 / metrics.request_duration_count as f64
    } else {
        0.0
    };
    let telemetry_stats = state.telemetry.stats();
    let ccr_stats = state.product.ccr_stats();
    let feedback_stats = state.product.feedback_stats();
    let toin_stats = state.product.toin_stats();
    let local_state_backends = state.local_state_backends();
    serde_json::json!({
        "summary": {
            "requests": metrics.requests_total,
            "errors": metrics.request_errors_total,
            "average_latency_ms": avg_latency_ms,
        },
        "savings": {
            "total_tokens": 0,
            "by_layer": {
                "cli_filtering": {
                    "tokens": 0,
                    "description": "Tokens avoided by CLI output filtering (rtk) before reaching context",
                },
                "compression": {
                    "tokens": 0,
                    "description": "Tokens removed by proxy compression (SmartCrusher, ContentRouter, etc.)",
                },
                "prefix_cache": {
                    "discount_usd": 0.0,
                    "description": "Cost discount from provider prefix caching.",
                }
            }
        },
        "requests": {
            "total": metrics.requests_total,
            "cached": compression_cache_stats.cache_hits
                + compression_cache_stats.cache_skip_hits
                + response_cache_stats.cache_hits,
            "rate_limited": 0,
            "failed": metrics.request_errors_total,
            "by_provider": request_log_stats.by_provider,
            "by_model": request_log_stats.by_model,
            "by_stack": request_log_stats.by_stack,
        },
        "tokens": {
            "input": request_log_stats.total_input_tokens,
            "output": request_log_stats.total_output_tokens,
            "saved": request_log_stats.total_tokens_saved,
            "cli_tokens_avoided": 0,
            "total_before_compression": request_log_stats.total_input_tokens_before_compression,
            "savings_percent": request_log_stats.savings_percent,
        },
        "latency": {
            "average_ms": avg_latency_ms,
            "min_ms": 0,
            "max_ms": avg_latency_ms,
            "total_requests": metrics.request_duration_count,
        },
        "overhead": {
            "average_ms": request_log_stats.overhead_average_ms,
            "min_ms": request_log_stats.overhead_min_ms,
            "max_ms": request_log_stats.overhead_max_ms,
        },
        "ttfb": {
            "average_ms": request_log_stats.ttfb_average_ms,
            "min_ms": request_log_stats.ttfb_min_ms,
            "max_ms": request_log_stats.ttfb_max_ms,
        },
        "pipeline_timing": {},
        "waste_signals": {},
        "savings_history": request_log_stats.savings_history,
        "display_session": request_log_stats.display_session,
        "persistent_savings": request_log_stats.persistent_savings,
        "prefix_cache": prefix_cache_stats,
        "compression_cache": {
            "mode": "proxy",
            "hits": compression_cache_stats.cache_hits,
            "misses": compression_cache_stats.cache_misses,
            "skip_hits": compression_cache_stats.cache_skip_hits,
            "evictions": compression_cache_stats.cache_evictions,
            "size": compression_cache_stats.cache_size,
            "skip_size": compression_cache_stats.cache_skip_size,
        },
        "response_cache": {
            "mode": "provider",
            "enabled": state.config.response_cache_enabled,
            "ttl_seconds": state.config.response_cache_ttl.as_secs(),
            "max_entries": state.config.response_cache_max_entries,
            "hits": response_cache_stats.cache_hits,
            "misses": response_cache_stats.cache_misses,
            "evictions": response_cache_stats.cache_evictions,
            "size": response_cache_stats.cache_size,
        },
        "cost": {
            "savings_usd": request_log_stats.total_compression_savings_usd,
            "compression_savings_usd": request_log_stats.total_compression_savings_usd,
            "cache_savings_usd": prefix_cache_savings_usd,
            "cli_tokens_avoided": 0,
            "total_input_cost_usd": request_log_stats.total_input_cost_usd,
            "per_model": request_log_stats.cost_per_model,
        },
        "compression": {
            "ccr_entries": ccr_stats["store"]["entry_count"].clone(),
            "ccr_max_entries": ccr_stats["store"]["max_entries"].clone(),
            "original_tokens_cached": ccr_stats["store"]["total_original_tokens"].clone(),
            "compressed_tokens_cached": ccr_stats["store"]["total_compressed_tokens"].clone(),
            "ccr_retrievals": ccr_stats["store"]["total_retrievals"].clone(),
        },
        "anon_telemetry_shipping": telemetry_enabled(),
        "telemetry": {
            "enabled": telemetry_stats["enabled"].clone(),
            "total_compressions": telemetry_stats["total_compressions"].clone(),
            "total_retrievals": telemetry_stats["total_retrievals"].clone(),
            "global_retrieval_rate": telemetry_stats["global_retrieval_rate"].clone(),
            "tool_signatures_tracked": telemetry_stats["tool_signatures_tracked"].clone(),
            "avg_compression_ratio": telemetry_stats["avg_compression_ratio"].clone(),
            "avg_token_reduction": telemetry_stats["avg_token_reduction"].clone(),
        },
        "otel": {},
        "langfuse": {},
        "feedback_loop": {
            "tools_tracked": feedback_stats["feedback"]["tools_tracked"].clone(),
            "total_compressions": feedback_stats["feedback"]["total_compressions"].clone(),
            "total_retrievals": feedback_stats["feedback"]["total_retrievals"].clone(),
            "global_retrieval_rate": feedback_stats["feedback"]["global_retrieval_rate"].clone(),
            "tools_with_high_retrieval": 0,
        },
        "toin": toin_stats,
        "local_state_backends": local_state_backends,
        "cli_filtering": {},
        "cache": Value::Null,
        "rate_limiter": Value::Null,
        "recent_requests": state.request_logs.recent(10),
        "log_full_messages": false,
        "upstream_status_classes": {
            "2xx": metrics.upstream_2xx_total,
            "4xx": metrics.upstream_4xx_total,
            "5xx": metrics.upstream_5xx_total,
            "in_flight": metrics.requests_in_flight,
        }
    })
}

struct RequestLogStats {
    total_input_tokens: u64,
    total_input_tokens_before_compression: u64,
    total_output_tokens: u64,
    total_tokens_saved: u64,
    savings_percent: f64,
    total_compression_savings_usd: f64,
    total_input_cost_usd: f64,
    overhead_average_ms: f64,
    overhead_min_ms: f64,
    overhead_max_ms: f64,
    ttfb_average_ms: f64,
    ttfb_min_ms: f64,
    ttfb_max_ms: f64,
    by_provider: serde_json::Map<String, Value>,
    by_model: serde_json::Map<String, Value>,
    by_stack: serde_json::Map<String, Value>,
    cost_per_model: serde_json::Map<String, Value>,
    savings_history: Vec<Value>,
    raw_history: Vec<Value>,
    lifetime: Value,
    display_session: Value,
    persistent_savings: Value,
}

fn aggregate_request_log_stats(
    entries: &[Value],
    storage_path: Option<&std::path::Path>,
) -> RequestLogStats {
    let mut total_input_tokens = 0_u64;
    let mut total_input_tokens_before_compression = 0_u64;
    let mut total_output_tokens = 0_u64;
    let mut total_tokens_saved = 0_u64;
    let mut total_compression_savings_usd = 0.0_f64;
    let mut total_input_cost_usd = 0.0_f64;
    let mut overhead_sum_ms = 0.0_f64;
    let mut overhead_count = 0_u64;
    let mut overhead_min_ms = f64::INFINITY;
    let mut overhead_max_ms = 0.0_f64;
    let mut ttfb_sum_ms = 0.0_f64;
    let mut ttfb_count = 0_u64;
    let mut ttfb_min_ms = f64::INFINITY;
    let mut ttfb_max_ms = 0.0_f64;
    let mut by_provider = BTreeMap::<String, u64>::new();
    let mut by_model = BTreeMap::<String, u64>::new();
    let mut by_stack = BTreeMap::<String, u64>::new();
    let mut cost_per_model = BTreeMap::<String, (u64, u64, u64, f64)>::new();
    let mut savings_history = Vec::new();
    let mut raw_history = Vec::new();
    let mut started_at: Option<String> = None;
    let mut last_activity_at: Option<String> = None;

    for entry in entries {
        let input_tokens_original = entry["input_tokens_original"].as_u64().unwrap_or(0);
        let input_tokens_optimized = entry["input_tokens_optimized"].as_u64().unwrap_or(0);
        let output_tokens = entry["output_tokens"].as_u64().unwrap_or(0);
        let tokens_saved = entry["tokens_saved"].as_u64().unwrap_or(0);
        let optimization_latency_ms = entry["optimization_latency_ms"].as_f64().unwrap_or(0.0);
        let ttfb_ms = entry["ttfb_ms"].as_f64().unwrap_or(0.0);
        let model_name = entry["model"].as_str().filter(|value| !value.is_empty());
        let compression_savings_usd = model_name
            .map(|model| estimate_compression_savings_usd(model, tokens_saved))
            .unwrap_or(0.0);
        let input_cost_usd = model_name
            .map(|model| estimate_input_cost_usd(model, input_tokens_optimized))
            .unwrap_or(0.0);
        let timestamp = entry["timestamp"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(current_timestamp_iso);
        let normalized_timestamp = normalize_timestamp_string(&timestamp);

        total_input_tokens_before_compression =
            total_input_tokens_before_compression.saturating_add(input_tokens_original);
        total_input_tokens = total_input_tokens.saturating_add(input_tokens_optimized);
        total_output_tokens = total_output_tokens.saturating_add(output_tokens);
        total_tokens_saved = total_tokens_saved.saturating_add(tokens_saved);
        total_compression_savings_usd += compression_savings_usd;
        total_input_cost_usd += input_cost_usd;
        if optimization_latency_ms > 0.0 {
            overhead_sum_ms += optimization_latency_ms;
            overhead_count += 1;
            overhead_min_ms = overhead_min_ms.min(optimization_latency_ms);
            overhead_max_ms = overhead_max_ms.max(optimization_latency_ms);
        }
        if ttfb_ms > 0.0 {
            ttfb_sum_ms += ttfb_ms;
            ttfb_count += 1;
            ttfb_min_ms = ttfb_min_ms.min(ttfb_ms);
            ttfb_max_ms = ttfb_max_ms.max(ttfb_ms);
        }

        if let Some(provider) = entry["provider"].as_str().filter(|value| !value.is_empty()) {
            *by_provider.entry(provider.to_string()).or_default() += 1;
        }
        if let Some(model) = model_name {
            *by_model.entry(model.to_string()).or_default() += 1;
            let aggregates = cost_per_model.entry(model.to_string()).or_default();
            aggregates.0 = aggregates.0.saturating_add(1);
            aggregates.1 = aggregates.1.saturating_add(tokens_saved);
            aggregates.2 = aggregates.2.saturating_add(input_tokens_optimized);
            aggregates.3 += compression_savings_usd;
        }
        if let Some(stack) = entry["tags"]["route_mode"]
            .as_str()
            .or_else(|| entry["tags"]["route"].as_str())
            .filter(|value| !value.is_empty())
        {
            *by_stack.entry(stack.to_string()).or_default() += 1;
        }

        started_at.get_or_insert_with(|| normalized_timestamp.clone());
        last_activity_at = Some(normalized_timestamp.clone());
        if tokens_saved > 0 {
            let point = serde_json::json!({
                "timestamp": normalized_timestamp,
                "total_tokens_saved": total_tokens_saved,
                "compression_savings_usd": round6(total_compression_savings_usd),
                "total_input_tokens": total_input_tokens,
                "total_input_cost_usd": round6(total_input_cost_usd),
            });
            raw_history.push(point.clone());
            savings_history.push(serde_json::json!([
                point["timestamp"].clone(),
                point["total_tokens_saved"].clone(),
            ]));
        }
    }

    if savings_history.len() > 100 {
        let keep_from = savings_history.len().saturating_sub(100);
        savings_history = savings_history.into_iter().skip(keep_from).collect();
    }

    let savings_percent = if total_input_tokens_before_compression > 0 {
        round2(total_tokens_saved as f64 / total_input_tokens_before_compression as f64 * 100.0)
    } else {
        0.0
    };
    let total_compression_savings_usd = round6(total_compression_savings_usd);
    let total_input_cost_usd = round6(total_input_cost_usd);
    let overhead_average_ms = if overhead_count > 0 {
        round2(overhead_sum_ms / overhead_count as f64)
    } else {
        0.0
    };
    let overhead_min_ms = if overhead_count > 0 {
        round2(overhead_min_ms)
    } else {
        0.0
    };
    let overhead_max_ms = if overhead_count > 0 {
        round2(overhead_max_ms)
    } else {
        0.0
    };
    let ttfb_average_ms = if ttfb_count > 0 {
        round2(ttfb_sum_ms / ttfb_count as f64)
    } else {
        0.0
    };
    let ttfb_min_ms = if ttfb_count > 0 {
        round2(ttfb_min_ms)
    } else {
        0.0
    };
    let ttfb_max_ms = if ttfb_count > 0 {
        round2(ttfb_max_ms)
    } else {
        0.0
    };

    let lifetime = serde_json::json!({
        "requests": entries.len(),
        "tokens_saved": total_tokens_saved,
        "compression_savings_usd": total_compression_savings_usd,
        "total_input_tokens": total_input_tokens,
        "total_input_cost_usd": total_input_cost_usd,
    });

    let display_session = serde_json::json!({
        "requests": entries.len(),
        "tokens_saved": total_tokens_saved,
        "compression_savings_usd": total_compression_savings_usd,
        "total_input_tokens": total_input_tokens,
        "total_input_cost_usd": total_input_cost_usd,
        "savings_percent": savings_percent,
        "started_at": started_at,
        "last_activity_at": last_activity_at,
    });

    let persistent_savings = serde_json::json!({
        "schema_version": 2,
        "storage_path": storage_path.map(|path| path.to_string_lossy().to_string()).unwrap_or_default(),
        "lifetime": lifetime.clone(),
        "display_session": display_session.clone(),
        "display_session_policy": {
            "rollover_inactivity_minutes": 0,
        },
        "history_points": raw_history.len(),
        "recent_history": raw_history.iter().cloned().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>(),
        "retention": {
            "max_history_points": 0,
            "max_history_age_days": 0,
            "max_response_history_points": 500,
        }
    });

    RequestLogStats {
        total_input_tokens,
        total_input_tokens_before_compression,
        total_output_tokens,
        total_tokens_saved,
        savings_percent,
        total_compression_savings_usd,
        total_input_cost_usd,
        overhead_average_ms,
        overhead_min_ms,
        overhead_max_ms,
        ttfb_average_ms,
        ttfb_min_ms,
        ttfb_max_ms,
        by_provider: by_provider
            .into_iter()
            .map(|(key, value)| (key, Value::from(value)))
            .collect(),
        by_model: by_model
            .into_iter()
            .map(|(key, value)| (key, Value::from(value)))
            .collect(),
        by_stack: by_stack
            .into_iter()
            .map(|(key, value)| (key, Value::from(value)))
            .collect(),
        cost_per_model: cost_per_model
            .into_iter()
            .map(
                |(model, (requests, tokens_saved, tokens_sent, savings_usd))| {
                    let denominator = tokens_saved.saturating_add(tokens_sent);
                    let reduction_pct = if denominator > 0 {
                        round2(tokens_saved as f64 / denominator as f64 * 100.0)
                    } else {
                        0.0
                    };
                    (
                        model,
                        serde_json::json!({
                            "requests": requests,
                            "tokens_saved": tokens_saved,
                            "tokens_sent": tokens_sent,
                            "savings_usd": round6(savings_usd),
                            "reduction_pct": reduction_pct,
                        }),
                    )
                },
            )
            .collect(),
        savings_history,
        raw_history,
        lifetime,
        display_session,
        persistent_savings,
    }
}

fn build_stats_history_payload(
    entries: &[Value],
    history_mode: &str,
    storage_path: Option<&std::path::Path>,
) -> Value {
    let stats = aggregate_request_log_stats(entries, storage_path);
    let history = history_for_response(&stats.raw_history, history_mode);
    let series = serde_json::json!({
        "hourly": build_history_rollup(&stats.raw_history, TimeBucket::Hour),
        "daily": build_history_rollup(&stats.raw_history, TimeBucket::Day),
        "weekly": build_history_rollup(&stats.raw_history, TimeBucket::Week),
        "monthly": build_history_rollup(&stats.raw_history, TimeBucket::Month),
    });
    serde_json::json!({
        "schema_version": 2,
        "generated_at": current_timestamp_iso(),
        "storage_path": storage_path.map(|path| path.to_string_lossy().to_string()).unwrap_or_default(),
        "lifetime": stats.lifetime,
        "display_session": stats.display_session,
        "display_session_policy": {
            "rollover_inactivity_minutes": 0,
        },
        "history": history,
        "series": series,
        "exports": {
            "default_format": "json",
            "available_formats": ["json", "csv"],
            "available_series": ["history", "hourly", "daily", "weekly", "monthly"],
        },
        "retention": {
            "max_history_points": 0,
            "max_history_age_days": 0,
            "max_response_history_points": 500,
        },
        "history_summary": {
            "mode": history_mode,
            "stored_points": stats.raw_history.len(),
            "returned_points": history.len(),
            "compacted": history.len() < stats.raw_history.len(),
        }
    })
}

fn build_stats_history_csv(
    entries: &[Value],
    series: &str,
    storage_path: Option<&std::path::Path>,
) -> String {
    let stats = aggregate_request_log_stats(entries, storage_path);
    if series == "history" {
        let mut csv =
            "timestamp,total_tokens_saved,compression_savings_usd,total_input_tokens,total_input_cost_usd\n"
                .to_string();
        for row in &stats.raw_history {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                row["timestamp"].as_str().unwrap_or_default(),
                row["total_tokens_saved"].as_u64().unwrap_or(0),
                row["compression_savings_usd"].as_f64().unwrap_or(0.0),
                row["total_input_tokens"].as_u64().unwrap_or(0),
                row["total_input_cost_usd"].as_f64().unwrap_or(0.0)
            ));
        }
        return csv;
    }

    let bucket = match series {
        "hourly" => TimeBucket::Hour,
        "daily" => TimeBucket::Day,
        "weekly" => TimeBucket::Week,
        "monthly" => TimeBucket::Month,
        _ => TimeBucket::Day,
    };
    let rows = build_history_rollup(&stats.raw_history, bucket);
    let mut csv = "timestamp,tokens_saved,compression_savings_usd_delta,total_tokens_saved,compression_savings_usd,total_input_tokens_delta,total_input_tokens,total_input_cost_usd_delta,total_input_cost_usd\n".to_string();
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            row["timestamp"].as_str().unwrap_or_default(),
            row["tokens_saved"].as_u64().unwrap_or(0),
            row["compression_savings_usd_delta"].as_f64().unwrap_or(0.0),
            row["total_tokens_saved"].as_u64().unwrap_or(0),
            row["compression_savings_usd"].as_f64().unwrap_or(0.0),
            row["total_input_tokens_delta"].as_u64().unwrap_or(0),
            row["total_input_tokens"].as_u64().unwrap_or(0),
            row["total_input_cost_usd_delta"].as_f64().unwrap_or(0.0),
            row["total_input_cost_usd"].as_f64().unwrap_or(0.0)
        ));
    }
    csv
}

#[derive(Clone, Copy)]
enum TimeBucket {
    Hour,
    Day,
    Week,
    Month,
}

fn history_for_response(history: &[Value], mode: &str) -> Vec<Value> {
    match mode {
        "none" => Vec::new(),
        "full" => history.to_vec(),
        _ => compact_history(history, 500),
    }
}

fn compact_history(history: &[Value], max_points: usize) -> Vec<Value> {
    if history.len() <= max_points {
        return history.to_vec();
    }

    let recent_points = min(max(max_points / 3, 50), max_points.saturating_sub(1));
    let recent = &history[history.len().saturating_sub(recent_points)..];
    let older = &history[..history.len().saturating_sub(recent.len())];
    let older_slots = max_points.saturating_sub(recent.len());
    if older_slots == 0 || older.is_empty() {
        return recent.to_vec();
    }

    let sampled_older: Vec<Value> = if older_slots == 1 {
        vec![older[0].clone()]
    } else {
        (0..older_slots)
            .map(|index| older[((older.len() - 1) * index) / (older_slots - 1)].clone())
            .collect()
    };

    let mut compacted = Vec::new();
    let mut seen_timestamps = HashSet::new();
    for point in sampled_older.into_iter().chain(recent.iter().cloned()) {
        let Some(timestamp) = point.get("timestamp").and_then(Value::as_str) else {
            continue;
        };
        if seen_timestamps.insert(timestamp.to_string()) {
            compacted.push(point);
        }
    }
    compacted
}

fn build_history_rollup(history: &[Value], bucket: TimeBucket) -> Vec<Value> {
    if history.is_empty() {
        return Vec::new();
    }

    let mut aggregated = BTreeMap::<u64, Value>::new();
    let mut prev_total_tokens = 0_u64;
    let mut prev_total_usd = 0.0_f64;
    let mut prev_total_input_tokens = 0_u64;
    let mut prev_total_input_cost_usd = 0.0_f64;

    for point in history {
        let Some(timestamp) = point["timestamp"].as_str().and_then(parse_timestamp_string) else {
            continue;
        };
        let bucket_key = bucket_start_timestamp(timestamp, bucket);
        let total_tokens_saved = point["total_tokens_saved"].as_u64().unwrap_or(0);
        let total_usd = point["compression_savings_usd"].as_f64().unwrap_or(0.0);
        let total_input_tokens = point["total_input_tokens"].as_u64().unwrap_or(0);
        let total_input_cost_usd = point["total_input_cost_usd"].as_f64().unwrap_or(0.0);
        let delta_tokens = total_tokens_saved.saturating_sub(prev_total_tokens);
        let delta_usd = (total_usd - prev_total_usd).max(0.0);
        let delta_input_tokens = total_input_tokens.saturating_sub(prev_total_input_tokens);
        let delta_input_cost_usd = (total_input_cost_usd - prev_total_input_cost_usd).max(0.0);

        prev_total_tokens = total_tokens_saved;
        prev_total_usd = total_usd;
        prev_total_input_tokens = total_input_tokens;
        prev_total_input_cost_usd = total_input_cost_usd;

        let entry = aggregated.entry(bucket_key).or_insert_with(|| {
            serde_json::json!({
                "timestamp": bucket_key.to_string(),
                "tokens_saved": 0,
                "compression_savings_usd_delta": 0.0,
                "total_tokens_saved": total_tokens_saved,
                "compression_savings_usd": round6(total_usd),
                "total_input_tokens_delta": 0,
                "total_input_tokens": total_input_tokens,
                "total_input_cost_usd_delta": 0.0,
                "total_input_cost_usd": round6(total_input_cost_usd),
            })
        });
        entry["tokens_saved"] =
            Value::from(entry["tokens_saved"].as_u64().unwrap_or(0) + delta_tokens);
        entry["compression_savings_usd_delta"] = Value::from(round6(
            entry["compression_savings_usd_delta"]
                .as_f64()
                .unwrap_or(0.0)
                + delta_usd,
        ));
        entry["total_input_tokens_delta"] = Value::from(
            entry["total_input_tokens_delta"].as_u64().unwrap_or(0) + delta_input_tokens,
        );
        entry["total_input_cost_usd_delta"] = Value::from(round6(
            entry["total_input_cost_usd_delta"].as_f64().unwrap_or(0.0) + delta_input_cost_usd,
        ));
        entry["total_tokens_saved"] = Value::from(total_tokens_saved);
        entry["compression_savings_usd"] = Value::from(round6(total_usd));
        entry["total_input_tokens"] = Value::from(total_input_tokens);
        entry["total_input_cost_usd"] = Value::from(round6(total_input_cost_usd));
    }

    aggregated.into_values().collect()
}

fn parse_timestamp_string(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    trimmed
        .parse::<u64>()
        .ok()
        .or_else(|| parse_iso_timestamp(trimmed))
}

fn bucket_start_timestamp(timestamp: u64, bucket: TimeBucket) -> u64 {
    match bucket {
        TimeBucket::Hour => (timestamp / 3600) * 3600,
        TimeBucket::Day => (timestamp / 86_400) * 86_400,
        TimeBucket::Week => {
            let days = (timestamp / 86_400) as i64;
            let days_from_monday = (days + 3).rem_euclid(7);
            let week_start_days = days - days_from_monday;
            if week_start_days <= 0 {
                0
            } else {
                (week_start_days as u64) * 86_400
            }
        }
        TimeBucket::Month => {
            let days = (timestamp / 86_400) as i64;
            let (year, month, _) = civil_from_days(days);
            let month_start_days = days_from_civil(year, month, 1);
            if month_start_days <= 0 {
                0
            } else {
                (month_start_days as u64) * 86_400
            }
        }
    }
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m as u32, d as u32)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = year - if month <= 2 { 1 } else { 0 };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i64;
    let day = day as i64;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn build_debug_memory_payload() -> Value {
    serde_json::json!({
        "process": {
            "rss_bytes": 0,
            "rss_mb": 0.0,
            "vms_bytes": 0,
            "vms_mb": 0.0,
            "percent": 0.0,
            "available_mb": 0.0,
            "total_mb": 0.0,
        },
        "components": {},
        "total_tracked_bytes": 0,
        "total_tracked_mb": 0.0,
        "target_budget_bytes": Value::Null,
        "target_budget_mb": Value::Null,
        "is_over_budget": false,
        "timestamp": unix_timestamp_seconds(),
    })
}

fn empty_display_session() -> Value {
    serde_json::json!({
        "requests": 0,
        "tokens_saved": 0,
        "compression_savings_usd": 0.0,
        "total_input_tokens": 0,
        "total_input_cost_usd": 0.0,
        "savings_percent": 0.0,
        "started_at": Value::Null,
        "last_activity_at": Value::Null,
    })
}

fn empty_persistent_savings() -> Value {
    serde_json::json!({
        "schema_version": 1,
        "storage_path": "",
        "lifetime": {
            "requests": 0,
            "tokens_saved": 0,
            "compression_savings_usd": 0.0,
            "total_input_tokens": 0,
            "total_input_cost_usd": 0.0,
        },
        "display_session": empty_display_session(),
        "display_session_policy": {
            "rollover_inactivity_minutes": 0,
        },
        "history_points": 0,
        "recent_history": [],
        "retention": {
            "max_history_points": 0,
            "max_history_age_days": 0,
            "max_response_history_points": 0,
        }
    })
}

fn unix_timestamp_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_timestamp_iso() -> String {
    timestamp_to_iso(unix_timestamp_seconds())
}

fn normalize_timestamp_string(value: &str) -> String {
    parse_timestamp_string(value)
        .map(timestamp_to_iso)
        .unwrap_or_else(|| value.trim().to_string())
}

fn timestamp_to_iso(timestamp: u64) -> String {
    let days = (timestamp / 86_400) as i64;
    let seconds_of_day = timestamp % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn parse_iso_timestamp(value: &str) -> Option<u64> {
    if value.len() != 20 || !value.ends_with('Z') {
        return None;
    }
    let year = value.get(0..4)?.parse::<i64>().ok()?;
    let month = value.get(5..7)?.parse::<u32>().ok()?;
    let day = value.get(8..10)?.parse::<u32>().ok()?;
    let hour = value.get(11..13)?.parse::<u64>().ok()?;
    let minute = value.get(14..16)?.parse::<u64>().ok()?;
    let second = value.get(17..19)?.parse::<u64>().ok()?;
    let days = days_from_civil(year, month, day);
    if days < 0 {
        return None;
    }
    Some((days as u64) * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn parse_ccr_tool_call(
    tool_call: &Value,
    provider: &str,
) -> Option<(String, Option<String>, String)> {
    const CCR_TOOL_NAME: &str = "headroom_retrieve";

    let (name, input_data, tool_call_id) = match provider {
        "anthropic" => (
            tool_call
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            tool_call
                .get("input")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
            tool_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "openai" => {
            let function = tool_call.get("function")?;
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|args| serde_json::from_str::<Value>(args).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            (
                function
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                arguments,
                tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        }
        "google" => {
            let function_call = tool_call.get("functionCall")?;
            (
                function_call
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                function_call
                    .get("args")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
                function_call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        }
        _ => (
            tool_call
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            tool_call
                .get("input")
                .or_else(|| tool_call.get("args"))
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
            tool_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
    };

    if name.as_deref() != Some(CCR_TOOL_NAME) {
        return None;
    }

    let hash_key = input_data.get("hash").and_then(Value::as_str)?;
    if hash_key.len() != 24
        || !hash_key
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return None;
    }
    let query = input_data
        .get("query")
        .and_then(Value::as_str)
        .map(str::to_string);
    Some((hash_key.to_string(), query, tool_call_id))
}

async fn execute_buffered_forward(
    state: &AppState,
    client_addr: SocketAddr,
    method: &Method,
    uri: &Uri,
    request_headers: &HeaderMap,
    body_bytes: Bytes,
    request_id: &str,
    upstream_base: &url::Url,
) -> Result<BufferedForwardResponse, ProxyError> {
    let start = Instant::now();
    let upstream_url = build_upstream_url(upstream_base, uri)?;
    let forwarded_host = request_headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut outgoing_headers = build_forward_request_headers(
        request_headers,
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        request_id,
    );
    if !state.config.rewrite_host {
        if let Some(h) = request_headers.get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;
    let upstream_resp = state
        .client
        .request(reqwest_method, upstream_url)
        .headers(outgoing_headers)
        .body(reqwest::Body::from(body_bytes))
        .send()
        .await?;
    let ttfb_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let headers = filter_response_headers(upstream_resp.headers());
    let body = upstream_resp.bytes().await?;
    Ok(BufferedForwardResponse {
        status,
        headers,
        body,
        ttfb_ms,
    })
}

async fn execute_streaming_forward(
    state: &AppState,
    client_addr: SocketAddr,
    method: &Method,
    uri: &Uri,
    request_headers: &HeaderMap,
    body_bytes: Bytes,
    request_id: &str,
    upstream_base: &url::Url,
) -> Result<StreamingForwardResponse, ProxyError> {
    let start = Instant::now();
    let upstream_url = build_upstream_url(upstream_base, uri)?;
    let forwarded_host = request_headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut outgoing_headers = build_forward_request_headers(
        request_headers,
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        request_id,
    );
    if !state.config.rewrite_host {
        if let Some(h) = request_headers.get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;
    let upstream_resp = state
        .client
        .request(reqwest_method, upstream_url)
        .headers(outgoing_headers)
        .body(reqwest::Body::from(body_bytes))
        .send()
        .await?;
    let ttfb_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let headers = filter_response_headers(upstream_resp.headers());
    Ok(StreamingForwardResponse {
        status,
        headers,
        body: Box::pin(upstream_resp.bytes_stream()),
        ttfb_ms,
    })
}

fn buffered_response_to_http(
    buffered: BufferedForwardResponse,
    request_id: &str,
) -> Result<Response<Body>, ProxyError> {
    let mut response = Response::builder().status(buffered.status);
    {
        let h = response.headers_mut().expect("builder has headers");
        h.extend(buffered.headers);
        if let Ok(v) = http::HeaderValue::from_str(request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
    }
    response
        .body(Body::from(buffered.body))
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))
}

fn cached_response_to_http(
    cached: CachedResponse,
    request_id: &str,
) -> Result<Response<Body>, ProxyError> {
    buffered_response_to_http(
        BufferedForwardResponse {
            status: cached.status,
            headers: cached.headers,
            body: cached.body,
            ttfb_ms: 0.0,
        },
        request_id,
    )
}

fn response_cache_is_enabled(config: &Config) -> bool {
    config.response_cache_enabled
        && config.response_cache_max_entries > 0
        && config.response_cache_ttl.as_secs() > 0
}

fn hash_response_cache_request(
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    body_bytes: &Bytes,
    upstream_base: &url::Url,
    provider: &str,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    method.as_str().hash(&mut hasher);
    uri.path().hash(&mut hasher);
    uri.query().unwrap_or_default().hash(&mut hasher);
    upstream_base.as_str().hash(&mut hasher);
    provider.hash(&mut hasher);
    body_bytes.hash(&mut hasher);
    for header in [
        "authorization",
        "x-api-key",
        "api-key",
        "chatgpt-account-id",
        "openai-organization",
        "anthropic-version",
    ] {
        header.hash(&mut hasher);
        if let Some(value) = headers.get(header) {
            value.as_bytes().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn streaming_response_to_http_with_shadow(
    streamed: StreamingForwardResponse,
    request_id: &str,
    state: AppState,
    provider: ShadowProvider,
    client_addr: SocketAddr,
    method: Method,
    uri: Uri,
    request_headers: HeaderMap,
    body_bytes: Bytes,
) -> Result<Response<Body>, ProxyError> {
    let primary_status = streamed.status;
    let shadow_request_id = request_id.to_string();
    let captured_body = Arc::new(Mutex::new(Vec::new()));
    let (primary_complete_tx, primary_complete_rx) = oneshot::channel::<bool>();
    let body = Body::from_stream(capture_stream_for_shadow(
        streamed.body,
        captured_body.clone(),
        shadow_request_id.clone(),
        primary_complete_tx,
    ));

    tokio::spawn(async move {
        let (shadow, primary_completed) = tokio::join!(
            execute_buffered_forward(
                &state,
                client_addr,
                &method,
                &uri,
                &request_headers,
                body_bytes,
                &shadow_request_id,
                &state.config.upstream,
            ),
            async { primary_complete_rx.await.unwrap_or(false) }
        );

        let primary_body = {
            let captured = captured_body.lock().expect("poisoned");
            Bytes::copy_from_slice(captured.as_slice())
        };
        let primary = BufferedForwardResponse {
            status: primary_status,
            headers: HeaderMap::new(),
            body: primary_body,
            ttfb_ms: 0.0,
        };
        let shadow_matched = primary_completed
            && match shadow {
                Ok(ref shadow_response) => responses_match(&primary, shadow_response),
                Err(_) => false,
            };

        match provider {
            ShadowProvider::OpenAi => state
                .metrics
                .record_openai_shadow_comparison(shadow_matched),
            ShadowProvider::Anthropic => state
                .metrics
                .record_anthropic_shadow_comparison(shadow_matched),
            ShadowProvider::Gemini => state
                .metrics
                .record_gemini_shadow_comparison(shadow_matched),
            ShadowProvider::GoogleCloudCodeStream => state
                .metrics
                .record_google_cloudcode_stream_shadow_comparison(shadow_matched),
        }
    });

    let mut response = Response::builder().status(streamed.status);
    {
        let h = response.headers_mut().expect("builder has headers");
        h.extend(streamed.headers);
        if let Ok(v) = http::HeaderValue::from_str(request_id) {
            h.insert(HeaderName::from_static("x-request-id"), v);
        }
    }
    response
        .body(body)
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))
}

fn capture_stream_for_shadow(
    stream: ResponseByteStream,
    captured_body: Arc<Mutex<Vec<u8>>>,
    request_id: String,
    complete_tx: oneshot::Sender<bool>,
) -> impl Stream<Item = Result<Bytes, reqwest::Error>> + Send {
    futures_util::stream::unfold(
        (stream, captured_body, request_id, Some(complete_tx)),
        |(mut stream, captured_body, request_id, mut complete_tx)| async move {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    captured_body
                        .lock()
                        .expect("poisoned")
                        .extend_from_slice(&bytes);
                    Some((Ok(bytes), (stream, captured_body, request_id, complete_tx)))
                }
                Some(Err(error)) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %error,
                        "upstream stream error mid-response"
                    );
                    if let Some(tx) = complete_tx.take() {
                        let _ = tx.send(false);
                    }
                    Some((Err(error), (stream, captured_body, request_id, complete_tx)))
                }
                None => {
                    if let Some(tx) = complete_tx.take() {
                        let _ = tx.send(true);
                    }
                    None
                }
            }
        },
    )
}

fn responses_match(primary: &BufferedForwardResponse, shadow: &BufferedForwardResponse) -> bool {
    if primary.status != shadow.status {
        return false;
    }

    if let (Ok(primary_json), Ok(shadow_json)) = (
        serde_json::from_slice::<Value>(&primary.body),
        serde_json::from_slice::<Value>(&shadow.body),
    ) {
        return primary_json == shadow_json;
    }

    if let (Some(primary_stream), Some(shadow_stream)) = (
        extract_semantic_sse_payload(&primary.body),
        extract_semantic_sse_payload(&shadow.body),
    ) {
        return primary_stream == shadow_stream;
    }

    primary.body == shadow.body
}

fn extract_semantic_sse_payload(body: &Bytes) -> Option<String> {
    let text = std::str::from_utf8(body).ok()?;
    if !(text.contains("data:") || text.contains("event:")) {
        return None;
    }

    let mut combined = String::new();
    for block in text.split("\n\n") {
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(value) = line.strip_prefix("data:") {
                data_lines.push(value.trim());
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let data = data_lines.join("\n");
        let trimmed = data.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            let mut extracted = String::new();
            append_shadow_text(&json, &mut extracted);
            if !extracted.is_empty() {
                combined.push_str(&extracted);
                continue;
            }
            if json == serde_json::json!({})
                || json.get("done").and_then(Value::as_bool) == Some(true)
            {
                continue;
            }
            if let Ok(normalized) = serde_json::to_string(&json) {
                combined.push_str(&normalized);
            }
            continue;
        }
        combined.push_str(trimmed);
    }

    Some(combined)
}

fn append_shadow_text(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            for item in items {
                append_shadow_text(item, out);
            }
        }
        Value::Object(map) => {
            for key in ["text", "content", "chunk"] {
                if let Some(text) = map.get(key).and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            for value in map.values() {
                if value.is_array() || value.is_object() {
                    append_shadow_text(value, out);
                }
            }
        }
        _ => {}
    }
}

fn gemini_model_from_path(path: &str) -> Option<String> {
    gemini_model_from_suffix(path, ":generateContent")
        .or_else(|| gemini_model_from_suffix(path, ":streamGenerateContent"))
}

fn gemini_model_from_suffix(path: &str, route_suffix: &str) -> Option<String> {
    let model_path = path.strip_prefix("/v1beta/models/")?;
    let model = model_path.strip_suffix(route_suffix)?;
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}

fn is_gemini_streaming_query(query: Option<&str>) -> bool {
    query
        .map(|value| {
            value
                .split('&')
                .any(|part| part.eq_ignore_ascii_case("alt=sse"))
        })
        .unwrap_or(false)
}

fn is_gemini_stream_generate_content_path(path: &str) -> bool {
    path.starts_with("/v1beta/models/") && path.ends_with(":streamGenerateContent")
}

fn parse_openai_chat_request(body: &Bytes) -> Result<(String, usize, bool), ProxyError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::InvalidRequest("missing string field `model`".to_string()))?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProxyError::InvalidRequest("missing array field `messages`".to_string()))?;
    let streaming = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok((model.to_string(), messages.len(), streaming))
}

struct OpenAiChatOptimization {
    primary_body_bytes: Bytes,
    compression_status: &'static str,
}

struct AnthropicMessagesOptimization {
    primary_body_bytes: Bytes,
    compression_status: &'static str,
}

struct OpenAiBatchCompressionStats {
    total_requests: usize,
    total_original_tokens: u64,
    total_compressed_tokens: u64,
    total_tokens_saved: u64,
    savings_percent: f64,
    errors: usize,
}

struct OpenAiBatchJsonlOptimization {
    content: String,
    compression_status: &'static str,
    stats: OpenAiBatchCompressionStats,
}

struct AnthropicBatchCreateOptimization {
    primary_body_bytes: Bytes,
    compression_status: &'static str,
    requests_count: usize,
}

fn optimize_openai_chat_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
) -> Result<OpenAiChatOptimization, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body_bytes)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if messages.is_empty() {
        return Ok(OpenAiChatOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
        });
    }

    let mut compression_body = object.clone();
    let mut compression_config = compression_body
        .get("config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
    compression_body.insert("config".to_string(), Value::Object(compression_config));

    let Ok(result) = build_compress_response(
        compression_cache,
        product_store,
        telemetry_store,
        &compression_body,
        &messages,
        model,
    ) else {
        return Ok(OpenAiChatOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "failed",
        });
    };
    let compressed_messages = result.body["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if compressed_messages == messages {
        return Ok(OpenAiChatOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "unchanged",
        });
    }

    object.insert("messages".to_string(), Value::Array(compressed_messages));
    let primary_body_bytes = serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    Ok(OpenAiChatOptimization {
        primary_body_bytes,
        compression_status: "compressed",
    })
}

fn optimize_anthropic_messages_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
) -> Result<AnthropicMessagesOptimization, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body_bytes)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if messages.is_empty() {
        return Ok(AnthropicMessagesOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
        });
    }

    let mut compression_body = object.clone();
    let mut compression_config = compression_body
        .get("config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
    compression_body.insert("config".to_string(), Value::Object(compression_config));

    let Ok(result) = build_compress_response(
        compression_cache,
        product_store,
        telemetry_store,
        &compression_body,
        &messages,
        model,
    ) else {
        return Ok(AnthropicMessagesOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "failed",
        });
    };
    let compressed_messages = result.body["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if compressed_messages == messages {
        return Ok(AnthropicMessagesOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "unchanged",
        });
    }

    object.insert("messages".to_string(), Value::Array(compressed_messages));
    let primary_body_bytes = serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    Ok(AnthropicMessagesOptimization {
        primary_body_bytes,
        compression_status: "compressed",
    })
}

fn optimize_anthropic_batch_create_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
) -> Result<AnthropicBatchCreateOptimization, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body_bytes)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let Some(requests) = object.get_mut("requests").and_then(Value::as_array_mut) else {
        return Ok(AnthropicBatchCreateOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
            requests_count: 0,
        });
    };
    let requests_count = requests.len();
    if requests.is_empty() {
        return Ok(AnthropicBatchCreateOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
            requests_count,
        });
    }

    let mut any_changed = false;
    let mut had_failure = false;
    for request in requests.iter_mut() {
        let Some(request_object) = request.as_object_mut() else {
            continue;
        };
        let Some(params) = request_object
            .get_mut("params")
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        let Some(messages) = params.get("messages").and_then(Value::as_array).cloned() else {
            continue;
        };
        if messages.is_empty() {
            continue;
        }

        let model = params
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let mut compression_body = params.clone();
        let mut compression_config = compression_body
            .get("config")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
        compression_body.insert("config".to_string(), Value::Object(compression_config));

        match build_compress_response(
            compression_cache,
            product_store,
            telemetry_store,
            &compression_body,
            &messages,
            model,
        ) {
            Ok(result) => {
                let compressed_messages = result.body["messages"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                if compressed_messages != messages {
                    params.insert("messages".to_string(), Value::Array(compressed_messages));
                    any_changed = true;
                }
            }
            Err(_) => {
                had_failure = true;
            }
        }
    }

    if !any_changed {
        return Ok(AnthropicBatchCreateOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: if had_failure { "failed" } else { "unchanged" },
            requests_count,
        });
    }

    let primary_body_bytes = serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    Ok(AnthropicBatchCreateOptimization {
        primary_body_bytes,
        compression_status: "compressed",
        requests_count,
    })
}

fn optimize_openai_batch_jsonl_content(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    content: &str,
) -> Result<OpenAiBatchJsonlOptimization, ProxyError> {
    let mut compressed_lines = Vec::new();
    let mut total_original_tokens = 0u64;
    let mut total_compressed_tokens = 0u64;
    let mut total_requests = 0usize;
    let mut errors = 0usize;
    let mut any_changed = false;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        total_requests += 1;
        let Ok(mut request_obj) = serde_json::from_str::<Value>(line) else {
            errors += 1;
            compressed_lines.push(line.to_string());
            continue;
        };
        let Some(body) = request_obj.get_mut("body").and_then(Value::as_object_mut) else {
            compressed_lines.push(line.to_string());
            continue;
        };
        let Some(messages) = body.get("messages").and_then(Value::as_array).cloned() else {
            compressed_lines.push(line.to_string());
            continue;
        };
        if messages.is_empty() {
            compressed_lines.push(line.to_string());
            continue;
        }

        let model = body.get("model").and_then(Value::as_str).unwrap_or("gpt-4");
        let mut compression_body = body.clone();
        let mut compression_config = compression_body
            .get("config")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
        compression_body.insert("config".to_string(), Value::Object(compression_config));

        match build_compress_response(
            compression_cache,
            product_store,
            telemetry_store,
            &compression_body,
            &messages,
            model,
        ) {
            Ok(result) => {
                total_original_tokens += result.body["tokens_before"].as_u64().unwrap_or(0);
                total_compressed_tokens += result.body["tokens_after"].as_u64().unwrap_or(0);
                let compressed_messages = result.body["messages"]
                    .as_array()
                    .cloned()
                    .unwrap_or(messages.clone());
                if compressed_messages != messages {
                    body.insert("messages".to_string(), Value::Array(compressed_messages));
                    any_changed = true;
                }
                compressed_lines.push(serde_json::to_string(&request_obj).map_err(|error| {
                    ProxyError::InvalidRequest(format!("invalid JSON body: {error}"))
                })?);
            }
            Err(_) => {
                errors += 1;
                compressed_lines.push(line.to_string());
            }
        }
    }

    let total_tokens_saved = total_original_tokens.saturating_sub(total_compressed_tokens);
    let savings_percent = if total_original_tokens > 0 {
        total_tokens_saved as f64 / total_original_tokens as f64 * 100.0
    } else {
        0.0
    };
    Ok(OpenAiBatchJsonlOptimization {
        content: compressed_lines.join("\n"),
        compression_status: if any_changed {
            "compressed"
        } else if errors > 0 {
            "failed"
        } else {
            "unchanged"
        },
        stats: OpenAiBatchCompressionStats {
            total_requests,
            total_original_tokens,
            total_compressed_tokens,
            total_tokens_saved,
            savings_percent,
            errors,
        },
    })
}

fn build_openai_batch_create_body(
    input_file_id: &str,
    endpoint: &str,
    completion_window: Value,
    metadata: Map<String, Value>,
    original_file_id: &str,
    stats: &OpenAiBatchCompressionStats,
) -> Value {
    let mut compression_metadata = metadata;
    compression_metadata.insert(
        "headroom_compressed".to_string(),
        Value::String("true".to_string()),
    );
    compression_metadata.insert(
        "headroom_original_file_id".to_string(),
        Value::String(original_file_id.to_string()),
    );
    compression_metadata.insert(
        "headroom_total_requests".to_string(),
        Value::String(stats.total_requests.to_string()),
    );
    compression_metadata.insert(
        "headroom_tokens_saved".to_string(),
        Value::String(stats.total_tokens_saved.to_string()),
    );
    compression_metadata.insert(
        "headroom_original_tokens".to_string(),
        Value::String(stats.total_original_tokens.to_string()),
    );
    compression_metadata.insert(
        "headroom_compressed_tokens".to_string(),
        Value::String(stats.total_compressed_tokens.to_string()),
    );
    compression_metadata.insert(
        "headroom_savings_percent".to_string(),
        Value::String(format!("{:.1}", stats.savings_percent)),
    );
    serde_json::json!({
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
        "metadata": compression_metadata,
    })
}

fn attach_openai_batch_response_headers(
    response: &mut Response<Body>,
    stats: &OpenAiBatchCompressionStats,
) -> Result<(), ProxyError> {
    let headers = response.headers_mut();
    headers.remove(http::header::CONTENT_LENGTH);
    headers.remove(http::header::CONTENT_ENCODING);
    headers.insert(
        HeaderName::from_static("x-headroom-tokens-saved"),
        HeaderValue::from_str(&stats.total_tokens_saved.to_string())
            .map_err(|error| ProxyError::InvalidHeader(error.to_string()))?,
    );
    headers.insert(
        HeaderName::from_static("x-headroom-savings-percent"),
        HeaderValue::from_str(&format!("{:.1}", stats.savings_percent))
            .map_err(|error| ProxyError::InvalidHeader(error.to_string()))?,
    );
    Ok(())
}

fn request_from_parts_head(parts: &http::request::Parts) -> Result<Request<Body>, ProxyError> {
    let mut request = Request::builder()
        .method(parts.method.clone())
        .uri(parts.uri.clone())
        .body(Body::empty())
        .map_err(|error| ProxyError::InvalidHeader(error.to_string()))?;
    *request.headers_mut() = parts.headers.clone();
    Ok(request)
}

async fn download_openai_batch_file(
    state: &AppState,
    client_addr: SocketAddr,
    request_headers: &HeaderMap,
    request_id: &str,
    openai_base_url: &url::Url,
    file_id: &str,
) -> Result<Option<String>, ProxyError> {
    let uri = Uri::try_from(format!("/v1/files/{file_id}/content"))
        .map_err(|error| ProxyError::InvalidUpstream(format!("invalid batch file URI: {error}")))?;
    let upstream_url = build_upstream_url(openai_base_url, &uri)?;
    let forwarded_host = request_headers
        .get(http::header::HOST)
        .and_then(|value| value.to_str().ok());
    let outgoing_headers = build_forward_request_headers(
        request_headers,
        client_addr.ip(),
        "http",
        forwarded_host,
        request_id,
    );
    let response = state
        .client
        .get(upstream_url)
        .headers(outgoing_headers)
        .send()
        .await?;
    if !response.status().is_success() {
        return Ok(None);
    }
    Ok(Some(response.text().await?))
}

async fn upload_openai_batch_file(
    state: &AppState,
    client_addr: SocketAddr,
    request_headers: &HeaderMap,
    request_id: &str,
    openai_base_url: &url::Url,
    original_file_id: &str,
    content: &str,
) -> Result<Option<String>, ProxyError> {
    let uri = Uri::from_static("/v1/files");
    let upstream_url = build_upstream_url(openai_base_url, &uri)?;
    let forwarded_host = request_headers
        .get(http::header::HOST)
        .and_then(|value| value.to_str().ok());
    let mut outgoing_headers = build_forward_request_headers(
        request_headers,
        client_addr.ip(),
        "http",
        forwarded_host,
        request_id,
    );
    outgoing_headers.remove(http::header::CONTENT_TYPE);

    let form = reqwest::multipart::Form::new()
        .text("purpose", "batch")
        .part(
            "file",
            reqwest::multipart::Part::bytes(content.as_bytes().to_vec())
                .file_name(format!("compressed_{original_file_id}.jsonl"))
                .mime_str("application/jsonl")
                .map_err(|error| ProxyError::InvalidHeader(error.to_string()))?,
        );
    let response = state
        .client
        .post(upstream_url)
        .headers(outgoing_headers)
        .multipart(form)
        .send()
        .await?;
    if !response.status().is_success() {
        return Ok(None);
    }
    let payload: Value = serde_json::from_str(&response.text().await?)
        .map_err(|error| ProxyError::InvalidRequest(format!("invalid JSON body: {error}")))?;
    Ok(payload
        .get("id")
        .and_then(Value::as_str)
        .map(|value| value.to_string()))
}

fn ensure_openai_stream_include_usage(body: &Bytes) -> Result<Bytes, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    match object.get_mut("stream_options") {
        Some(Value::Object(stream_options)) => {
            stream_options.insert("include_usage".to_string(), Value::Bool(true));
        }
        None => {
            let mut stream_options = Map::new();
            stream_options.insert("include_usage".to_string(), Value::Bool(true));
            object.insert("stream_options".to_string(), Value::Object(stream_options));
        }
        Some(_) => {}
    }
    serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))
}

fn validate_request_content_length(
    headers: &HeaderMap,
    max_body_bytes: u64,
) -> Result<(), ProxyError> {
    let Some(value) = headers.get(http::header::CONTENT_LENGTH) else {
        return Ok(());
    };
    let content_length = value
        .to_str()
        .map_err(|err| ProxyError::InvalidHeader(format!("invalid content-length header: {err}")))?
        .parse::<u64>()
        .map_err(|err| {
            ProxyError::InvalidHeader(format!("invalid content-length header: {err}"))
        })?;
    if content_length > max_body_bytes {
        return Err(ProxyError::RequestTooLarge {
            actual_bytes: content_length,
            max_bytes: max_body_bytes,
        });
    }
    Ok(())
}

fn validate_request_array_length(
    field: &'static str,
    actual_length: usize,
) -> Result<(), ProxyError> {
    if actual_length > MAX_REQUEST_ARRAY_LENGTH {
        return Err(ProxyError::RequestArrayTooLarge {
            field,
            actual_length,
            max_length: MAX_REQUEST_ARRAY_LENGTH,
        });
    }
    Ok(())
}

fn is_loopback_addr(addr: SocketAddr) -> bool {
    match addr.ip() {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| mapped.is_loopback())
        }
    }
}

fn build_runtime_debug_payload(state: &AppState) -> Value {
    serde_json::json!({
        "request_pipeline": {
            "plugins_enabled": state.runtime.is_enabled(),
            "plugin_count": state.runtime.plugin_count(),
        },
        "websocket_sessions": {
            "active_sessions": state.ws_sessions.active_count(),
            "active_relay_tasks": state.ws_sessions.active_relay_task_count(),
        },
    })
}

fn log_native_compress_request(
    state: &AppState,
    request_id: &str,
    model: &str,
    result: &Value,
    cache_hit: bool,
    optimization_latency_ms: f64,
) {
    let input_tokens_original = result["tokens_before"].as_u64().unwrap_or(0);
    let input_tokens_optimized = result["tokens_after"].as_u64().unwrap_or(0);
    let tokens_saved = result["tokens_saved"].as_u64().unwrap_or(0);
    let savings_percent = if input_tokens_original > 0 {
        (tokens_saved as f64 / input_tokens_original as f64) * 100.0
    } else {
        0.0
    };
    let transforms_applied = result["transforms_applied"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    state.request_logs.log(serde_json::json!({
        "request_id": request_id,
        "timestamp": current_timestamp_iso(),
        "provider": "headroom",
        "model": model,
        "input_tokens_original": input_tokens_original,
        "input_tokens_optimized": input_tokens_optimized,
        "output_tokens": Value::Null,
        "tokens_saved": tokens_saved,
        "savings_percent": savings_percent,
        "optimization_latency_ms": optimization_latency_ms,
        "total_latency_ms": Value::Null,
        "tags": {
            "route": "compress",
        },
        "cache_hit": cache_hit,
        "transforms_applied": transforms_applied,
        "request_messages": Value::Null,
        "response_content": Value::Null,
        "error": Value::Null,
        "turn_id": Value::Null,
    }));
}

fn log_native_provider_request(
    state: &AppState,
    request_id: &str,
    provider: &str,
    model: &str,
    route_mode: &str,
    body_bytes: &Bytes,
    cache_hit: bool,
    optimization_latency_ms: f64,
    ttfb_ms: f64,
    total_latency_ms: f64,
) {
    let input_tokens = request_log_input_tokens(provider, model, body_bytes);
    state.request_logs.log(serde_json::json!({
        "request_id": request_id,
        "timestamp": current_timestamp_iso(),
        "provider": provider,
        "model": model,
        "input_tokens_original": input_tokens,
        "input_tokens_optimized": input_tokens,
        "output_tokens": Value::Null,
        "tokens_saved": 0,
        "savings_percent": 0.0,
        "optimization_latency_ms": optimization_latency_ms,
        "ttfb_ms": ttfb_ms,
        "total_latency_ms": total_latency_ms,
        "tags": {
            "route_mode": route_mode,
        },
        "cache_hit": cache_hit,
        "transforms_applied": [],
        "request_messages": Value::Null,
        "response_content": Value::Null,
        "error": Value::Null,
        "turn_id": Value::Null,
    }));
}

fn request_log_input_tokens(provider: &str, model: &str, body_bytes: &Bytes) -> u64 {
    let Ok(payload) = serde_json::from_slice::<Value>(body_bytes) else {
        return 0;
    };
    let Some(object) = payload.as_object() else {
        return 0;
    };
    let Some(messages) = object.get("messages").and_then(Value::as_array) else {
        let tokenizer = get_tokenizer(model);
        return match provider {
            "gemini" => object
                .get("contents")
                .and_then(Value::as_array)
                .map(|contents| {
                    contents
                        .iter()
                        .map(|content| count_content_tokens(content, tokenizer.as_ref()) as u64)
                        .sum()
                })
                .unwrap_or(0),
            "cloudcode" => object
                .get("request")
                .and_then(Value::as_object)
                .and_then(|request| request.get("contents"))
                .and_then(Value::as_array)
                .map(|contents| {
                    contents
                        .iter()
                        .map(|content| count_content_tokens(content, tokenizer.as_ref()) as u64)
                        .sum()
                })
                .unwrap_or(0),
            _ => 0,
        };
    };
    let tokenizer = get_tokenizer(model);
    match provider {
        "openai" | "anthropic" => count_message_tokens(messages, tokenizer.as_ref()) as u64,
        _ => 0,
    }
}

fn build_compress_response(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body: &serde_json::Map<String, Value>,
    messages: &[Value],
    model: &str,
) -> Result<CompressBuildResult, String> {
    let options = parse_compress_options(body);
    let query = extract_user_query(messages);
    let tokenizer = get_tokenizer(model);
    let tokens_before = count_message_tokens(messages, tokenizer.as_ref());
    let mut transforms_applied = Vec::new();
    let mut transform_counts = BTreeMap::new();
    let mut compressed_messages = Vec::with_capacity(messages.len());
    let mut ccr_hashes = Vec::new();
    let mut cache_hit = false;
    for (index, message) in messages.iter().enumerate() {
        let (compressed, message_transforms, message_hashes) = compress_message_value(
            message,
            index,
            messages.len(),
            compression_cache,
            &product_store,
            telemetry_store,
            &options,
            &query,
            tokenizer.as_ref(),
        );
        for transform in message_transforms {
            if transform == "router:cache_hit" {
                cache_hit = true;
                continue;
            }
            *transform_counts.entry(transform.clone()).or_insert(0usize) += 1;
            transforms_applied.push(transform);
        }
        for hash in message_hashes {
            if !ccr_hashes.contains(&hash) {
                ccr_hashes.push(hash);
            }
        }
        compressed_messages.push(compressed);
    }

    if let Some(model_limit) = compress_model_limit(model, &options) {
        let (windowed_messages, window_transforms) =
            apply_token_budget_window(compressed_messages, tokenizer.as_ref(), model_limit);
        compressed_messages = windowed_messages;
        for transform in window_transforms {
            *transform_counts.entry(transform.clone()).or_insert(0usize) += 1;
            transforms_applied.push(transform);
        }
    }

    let tokens_after = count_message_tokens(&compressed_messages, tokenizer.as_ref());
    let tokens_saved = tokens_before.saturating_sub(tokens_after);
    let compression_ratio = if tokens_before > 0 {
        tokens_after as f64 / tokens_before as f64
    } else {
        1.0
    };
    if transforms_applied.is_empty() {
        transforms_applied.push("router:noop".to_string());
        transform_counts.insert("router:noop".to_string(), 1);
    }
    let transforms_summary = build_transform_summary(&transform_counts);
    Ok(CompressBuildResult {
        body: serde_json::json!({
            "messages": compressed_messages,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_saved,
            "compression_ratio": compression_ratio,
            "transforms_applied": transforms_applied,
            "transforms_summary": transforms_summary,
            "ccr_hashes": ccr_hashes,
        }),
        cache_hit,
    })
}

struct CompressBuildResult {
    body: Value,
    cache_hit: bool,
}

#[derive(Debug, Clone, Copy)]
struct CompressOptions {
    compress_user_messages: bool,
    compress_system_messages: bool,
    protect_recent: usize,
    min_tokens_to_compress: usize,
    target_ratio: Option<f64>,
    protect_analysis_context: bool,
    token_budget: Option<usize>,
}

fn parse_compress_options(body: &serde_json::Map<String, Value>) -> CompressOptions {
    let config = body.get("config").and_then(Value::as_object);
    CompressOptions {
        compress_user_messages: config
            .and_then(|value| value.get("compress_user_messages"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        compress_system_messages: true,
        protect_recent: config
            .and_then(|value| value.get("protect_recent"))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(4),
        min_tokens_to_compress: 50,
        target_ratio: config
            .and_then(|value| value.get("target_ratio"))
            .and_then(Value::as_f64)
            .filter(|value| *value > 0.0 && *value < 1.0),
        protect_analysis_context: config
            .and_then(|value| value.get("protect_analysis_context"))
            .and_then(Value::as_bool)
            .unwrap_or(true),
        token_budget: body
            .get("token_budget")
            .and_then(Value::as_u64)
            .map(|value| value as usize),
    }
}

fn compress_model_limit(model: &str, options: &CompressOptions) -> Option<usize> {
    if let Some(token_budget) = options.token_budget {
        return Some(token_budget);
    }
    infer_openai_context_limit(model)
}

fn infer_openai_context_limit(model: &str) -> Option<usize> {
    let model = model.to_ascii_lowercase();
    let inferred = if model == "gpt-4-32k" || model.starts_with("gpt-4-32k") {
        32_768
    } else if model == "o1" {
        200_000
    } else if model == "gpt-4" || model.starts_with("gpt-4-0613") || model.starts_with("gpt-4-0314")
    {
        8_192
    } else if model.starts_with("gpt-4o") || model.starts_with("gpt-4-turbo") {
        128_000
    } else if model.starts_with("gpt-3.5") {
        16_385
    } else if model.starts_with("o1-preview") || model.starts_with("o1-mini") {
        128_000
    } else if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
        200_000
    } else {
        128_000
    };
    Some(inferred)
}

fn compress_message_value(
    message: &Value,
    index: usize,
    total_messages: usize,
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    options: &CompressOptions,
    query: &str,
    tokenizer: &dyn Tokenizer,
) -> (Value, Vec<String>, Vec<String>) {
    let Some(object) = message.as_object() else {
        return (message.clone(), Vec::new(), Vec::new());
    };
    let Some(content) = object.get("content") else {
        return (message.clone(), Vec::new(), Vec::new());
    };
    if let Some(marker) =
        compression_skip_marker(object, content, index, total_messages, options, query)
    {
        return (message.clone(), vec![marker.to_string()], Vec::new());
    }
    if count_content_tokens(content, tokenizer) < options.min_tokens_to_compress {
        return (message.clone(), Vec::new(), Vec::new());
    }

    let (compressed_content, transforms, ccr_hashes) = compress_content_value(
        content,
        compression_cache,
        product_store,
        telemetry_store,
        query,
        options,
    );
    if transforms.is_empty() {
        return (message.clone(), Vec::new(), Vec::new());
    }

    let mut updated = object.clone();
    updated.insert("content".to_string(), compressed_content);
    (Value::Object(updated), transforms, ccr_hashes)
}

fn compression_skip_marker(
    message: &serde_json::Map<String, Value>,
    content: &Value,
    index: usize,
    total_messages: usize,
    options: &CompressOptions,
    query: &str,
) -> Option<&'static str> {
    match message.get("role").and_then(Value::as_str) {
        Some("user") if !options.compress_user_messages => Some("router:protected:user_message"),
        Some("system") if !options.compress_system_messages => {
            Some("router:protected:system_message")
        }
        _ if options.protect_recent > 0
            && index + options.protect_recent >= total_messages
            && content_contains_source_code(content) =>
        {
            Some("router:protected:recent_code")
        }
        _ if options.protect_analysis_context
            && detect_analysis_intent(query)
            && content_contains_source_code(content) =>
        {
            Some("router:protected:analysis_context")
        }
        _ => None,
    }
}

fn build_transform_summary(transform_counts: &BTreeMap<String, usize>) -> Value {
    let mut summary = Map::new();
    for (transform, count) in transform_counts {
        summary.insert(transform.clone(), Value::from(*count as u64));
    }
    Value::Object(summary)
}

fn count_logical_transforms(items: &[Value]) -> usize {
    items.iter().filter_map(Value::as_str).count()
}

fn compress_content_value(
    content: &Value,
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
) -> (Value, Vec<String>, Vec<String>) {
    match content {
        Value::String(text) => route_string_content(
            text,
            compression_cache,
            product_store,
            telemetry_store,
            query,
            options,
        )
        .map_or_else(
            || (content.clone(), Vec::new(), Vec::new()),
            |result| result,
        ),
        Value::Array(parts) => {
            let mut updated = Vec::with_capacity(parts.len());
            let mut changed = false;
            let mut ccr_hashes = Vec::new();
            let mut transforms = Vec::new();
            for part in parts {
                if let Some((updated_part, part_transforms, hashes)) = compress_content_part(
                    part,
                    compression_cache,
                    product_store,
                    telemetry_store,
                    query,
                    options,
                ) {
                    updated.push(updated_part);
                    changed = true;
                    for transform in part_transforms {
                        if !transforms.contains(&transform) {
                            transforms.push(transform);
                        }
                    }
                    for hash in hashes {
                        if !ccr_hashes.contains(&hash) {
                            ccr_hashes.push(hash);
                        }
                    }
                } else {
                    updated.push(part.clone());
                }
            }
            if changed {
                (Value::Array(updated), transforms, ccr_hashes)
            } else {
                (content.clone(), Vec::new(), Vec::new())
            }
        }
        _ => (content.clone(), Vec::new(), Vec::new()),
    }
}

fn compress_content_part(
    part: &Value,
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let object = part.as_object()?;
    let part_type = object.get("type").and_then(Value::as_str).unwrap_or("");
    if part_type != "text" && part_type != "input_text" && part_type != "output_text" {
        return None;
    }
    let text = object.get("text").and_then(Value::as_str)?;
    let (compressed, transforms, ccr_hashes) = route_string_content(
        text,
        compression_cache,
        product_store,
        telemetry_store,
        query,
        options,
    )?;

    let mut updated = object.clone();
    updated.insert("text".to_string(), compressed);
    Some((Value::Object(updated), transforms, ccr_hashes))
}

fn route_string_content(
    text: &str,
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let cache_key = hash_compression_content(text);
    let min_ratio = compression_acceptance_ratio(options);
    {
        let mut cache = compression_cache
            .lock()
            .expect("compression cache poisoned");
        if cache.is_skipped(cache_key) {
            return None;
        }
        if let Some(cached) = cache.get(cache_key) {
            if cached.compression_ratio < min_ratio {
                let ccr_hashes = extract_ccr_hashes(&cached.compressed);
                return Some((
                    Value::String(cached.compressed),
                    cached_routing_transforms(&cached.strategy, cached.compression_ratio),
                    ccr_hashes,
                ));
            }
            cache.move_to_skip(cache_key);
            return None;
        }
    }

    if is_mixed_content(text) {
        if let Some(result) =
            route_mixed_string_content(text, product_store, telemetry_store, query, options)
        {
            return cache_compression_result(compression_cache, cache_key, min_ratio, result);
        }
    }
    let result = route_pure_string_content(text, product_store, telemetry_store, query, options);
    if result.is_none() {
        compression_cache
            .lock()
            .expect("compression cache poisoned")
            .mark_skip(cache_key);
        return None;
    }
    cache_compression_result(compression_cache, cache_key, min_ratio, result?)
}

fn route_pure_string_content(
    text: &str,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    match detect_text_content_kind(text) {
        CompressContentKind::AlreadyCompressed | CompressContentKind::SourceCode => None,
        CompressContentKind::Html => {
            compress_with_html_strategy(text, product_store, telemetry_store, query)
        }
        CompressContentKind::Diff => {
            let compressor = build_diff_compressor(options);
            let result = compressor.compress(text, query);
            if result.compressed != text {
                let ccr_hashes = result.cache_key.into_iter().collect::<Vec<_>>();
                let ratio =
                    result.compressed_line_count as f64 / result.original_line_count.max(1) as f64;
                product_store.record_compression(
                    ccr_hashes.first().map(String::as_str),
                    "diff_compressor",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    Some(query),
                );
                telemetry_store.record_compression(
                    "diff_compressor",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    (text.len() / 4).max(1) as i64,
                    (result.compressed.len() / 4).max(1) as i64,
                );
                Some((
                    Value::String(result.compressed),
                    routing_transforms("diff_compressor", ratio),
                    ccr_hashes,
                ))
            } else {
                None
            }
        }
        CompressContentKind::SearchResults => {
            let compressor = build_search_compressor();
            let (result, _stats) = compressor.compress_with_store(
                text,
                query,
                compression_bias(options),
                Some(product_store),
            );
            if result.compressed != text {
                let ccr_hashes = result.cache_key.into_iter().collect::<Vec<_>>();
                product_store.record_compression(
                    ccr_hashes.first().map(String::as_str),
                    "search",
                    result.original_match_count as u64,
                    result.compressed_match_count as u64,
                    Some(query),
                );
                telemetry_store.record_compression(
                    "search",
                    result.original_match_count as u64,
                    result.compressed_match_count as u64,
                    (result.original.len() / 4).max(1) as i64,
                    (result.compressed.len() / 4).max(1) as i64,
                );
                Some((
                    Value::String(result.compressed),
                    routing_transforms("search", result.compression_ratio),
                    ccr_hashes,
                ))
            } else {
                None
            }
        }
        CompressContentKind::BuildOutput => {
            let compressor = build_log_compressor();
            let (result, _stats) = compressor.compress_with_store(
                text,
                compression_bias(options),
                Some(product_store),
            );
            if result.compressed != text {
                let ccr_hashes = result.cache_key.into_iter().collect::<Vec<_>>();
                product_store.record_compression(
                    ccr_hashes.first().map(String::as_str),
                    "log",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    Some(query),
                );
                telemetry_store.record_compression(
                    "log",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    (result.original.len() / 4).max(1) as i64,
                    (result.compressed.len() / 4).max(1) as i64,
                );
                Some((
                    Value::String(result.compressed),
                    routing_transforms("log", result.compression_ratio),
                    ccr_hashes,
                ))
            } else {
                None
            }
        }
        CompressContentKind::StructuredJson => compress_with_smart_crusher_strategy(
            text,
            product_store,
            telemetry_store,
            query,
            options,
            "smart_crusher",
        ),
        CompressContentKind::PlainText => {
            let compressor = build_text_compressor(text, options);
            let result = compressor.compress(text, query);
            if result.compressed != text {
                let ccr_hashes = result.cache_key.into_iter().collect::<Vec<_>>();
                product_store.record_compression(
                    ccr_hashes.first().map(String::as_str),
                    "text",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    Some(query),
                );
                telemetry_store.record_compression(
                    "text",
                    result.original_line_count as u64,
                    result.compressed_line_count as u64,
                    (result.original.len() / 4).max(1) as i64,
                    (result.compressed.len() / 4).max(1) as i64,
                );
                Some((
                    Value::String(result.compressed),
                    routing_transforms("text", result.compression_ratio),
                    ccr_hashes,
                ))
            } else {
                None
            }
        }
    }
}

fn route_mixed_string_content(
    text: &str,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let sections = split_mixed_content_sections(text);
    if sections.len() < 2 {
        return None;
    }

    let mut changed = false;
    let mut hashes = Vec::new();
    let mut rendered = Vec::with_capacity(sections.len());
    for section in sections {
        let original = match &section {
            MixedContentSection::CodeFence { language, content } => {
                format!("```{language}\n{content}\n```")
            }
            MixedContentSection::CodeBlock(content) => content.clone(),
            MixedContentSection::Raw(content) => content.clone(),
        };

        let (rendered_section, section_hashes, section_changed) = match section {
            MixedContentSection::CodeFence { language, content } => {
                let wrapped = format!("```{language}\n{content}\n```");
                (wrapped, Vec::new(), false)
            }
            MixedContentSection::CodeBlock(content) => (content, Vec::new(), false),
            MixedContentSection::Raw(content) => {
                if let Some((Value::String(compressed), _, ccr_hashes)) = route_pure_string_content(
                    &content,
                    product_store,
                    telemetry_store,
                    query,
                    options,
                ) {
                    let did_change = compressed != content;
                    (compressed, ccr_hashes, did_change)
                } else {
                    (content.clone(), Vec::new(), false)
                }
            }
        };

        if section_changed || rendered_section != original {
            changed = true;
        }
        for hash in section_hashes {
            if !hashes.contains(&hash) {
                hashes.push(hash);
            }
        }
        rendered.push(rendered_section);
    }

    if !changed {
        return None;
    }

    Some((
        Value::String(rendered.join("\n\n")),
        routing_transforms(
            "mixed",
            compute_compression_ratio(&rendered.join("\n\n"), text),
        ),
        hashes,
    ))
}

fn cache_compression_result(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    cache_key: u64,
    min_ratio: f64,
    result: (Value, Vec<String>, Vec<String>),
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let (value, transforms, hashes) = result;
    let Value::String(compressed) = &value else {
        return Some((value, transforms, hashes));
    };
    let Some((strategy, ratio)) = transforms
        .iter()
        .find_map(|transform| parse_router_transform(transform))
    else {
        return Some((value, transforms, hashes));
    };
    let mut cache = compression_cache
        .lock()
        .expect("compression cache poisoned");
    if ratio < min_ratio {
        cache.put(cache_key, compressed.clone(), ratio, strategy);
    } else if transforms.is_empty() {
        cache.mark_skip(cache_key);
    }
    Some((value, transforms, hashes))
}

fn compress_with_smart_crusher_strategy(
    text: &str,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
    options: &CompressOptions,
    strategy: &str,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let crusher = build_smart_crusher_for_text(text, options);
    let (compressed, modified, _) = crusher.smart_crush_content(text, query, 1.0);
    if modified {
        let ccr_hashes = extract_ccr_hashes(&compressed);
        let ratio = compute_compression_ratio(&compressed, text);
        product_store.record_compression(
            ccr_hashes.first().map(String::as_str),
            strategy,
            count_compression_items(text),
            count_compression_items(&compressed),
            Some(query),
        );
        telemetry_store.record_compression(
            strategy,
            count_compression_items(text),
            count_compression_items(&compressed),
            (text.len() / 4).max(1) as i64,
            (compressed.len() / 4).max(1) as i64,
        );
        Some((
            Value::String(compressed),
            routing_transforms(strategy, ratio),
            ccr_hashes,
        ))
    } else {
        None
    }
}

fn compress_with_html_strategy(
    text: &str,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    query: &str,
) -> Option<(Value, Vec<String>, Vec<String>)> {
    let compressed = extract_html_text(text);
    if compressed.is_empty() || compressed == text {
        return None;
    }
    let ratio = compute_compression_ratio(&compressed, text);
    if ratio >= 1.0 {
        return None;
    }
    product_store.record_compression(
        None,
        "html",
        count_compression_items(text),
        count_compression_items(&compressed),
        Some(query),
    );
    telemetry_store.record_compression(
        "html",
        count_compression_items(text),
        count_compression_items(&compressed),
        (text.len() / 4).max(1) as i64,
        (compressed.len() / 4).max(1) as i64,
    );
    Some((
        Value::String(compressed),
        routing_transforms("html", ratio),
        Vec::new(),
    ))
}

fn count_compression_items(text: &str) -> u64 {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| match value {
            Value::Array(items) => Some(items.len() as u64),
            _ => None,
        })
        .unwrap_or_else(|| text.lines().count().max(1) as u64)
}

fn routing_transforms(strategy: &str, ratio: f64) -> Vec<String> {
    vec![format!("router:{strategy}:{ratio:.2}")]
}

fn cached_routing_transforms(strategy: &str, ratio: f64) -> Vec<String> {
    let mut transforms = routing_transforms(strategy, ratio);
    transforms.push("router:cache_hit".to_string());
    transforms
}

fn compute_compression_ratio(compressed: &str, original: &str) -> f64 {
    compressed.len() as f64 / original.len().max(1) as f64
}

fn compression_acceptance_ratio(options: &CompressOptions) -> f64 {
    options.target_ratio.unwrap_or(1.0)
}

fn parse_router_ratio(transform: &str, strategy: &str) -> Option<f64> {
    transform
        .strip_prefix(&format!("router:{strategy}:"))
        .and_then(|value| value.parse::<f64>().ok())
}

fn parse_router_transform(transform: &str) -> Option<(String, f64)> {
    let remainder = transform.strip_prefix("router:")?;
    let (strategy, ratio) = remainder.rsplit_once(':')?;
    let ratio = ratio.parse::<f64>().ok()?;
    Some((strategy.to_string(), ratio))
}

fn hash_compression_content(content: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

fn build_smart_crusher_for_text(text: &str, options: &CompressOptions) -> SmartCrusher {
    let mut config = headroom_core::transforms::smart_crusher::SmartCrusherConfig::default();
    if let Some(target_ratio) = options.target_ratio {
        if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(text) {
            let keep_count = ((items.len() as f64) * target_ratio).ceil() as usize;
            config.max_items_after_crush = keep_count.max(1);
        }
    }
    SmartCrusher::new(config)
}

fn build_diff_compressor(options: &CompressOptions) -> DiffCompressor {
    let mut config = DiffCompressorConfig::default();
    if let Some(target_ratio) = options.target_ratio {
        config.min_compression_ratio_for_ccr = target_ratio.clamp(0.05, 0.95);
    }
    DiffCompressor::new(config)
}

fn build_search_compressor() -> SearchCompressor {
    SearchCompressor::new(SearchCompressorConfig::default())
}

fn build_log_compressor() -> LogCompressor {
    LogCompressor::new(LogCompressorConfig::default())
}

fn build_text_compressor(text: &str, options: &CompressOptions) -> TextCompressor {
    let mut config = TextCompressorConfig::default();
    if let Some(target_ratio) = options.target_ratio {
        let total_lines = text.lines().count().max(1);
        let desired_lines = ((total_lines as f64) * target_ratio.clamp(0.05, 0.95)).ceil() as usize;
        let floor = config.keep_first_lines + config.keep_last_lines;
        config.max_total_lines = desired_lines.max(floor).max(10);
        let middle_budget = config
            .max_total_lines
            .saturating_sub(config.keep_first_lines + config.keep_last_lines)
            .max(1);
        let middle_lines =
            total_lines.saturating_sub(config.keep_first_lines + config.keep_last_lines);
        config.sample_every_n_lines = (middle_lines / middle_budget).max(1);
    }
    TextCompressor::new(config)
}

fn extract_html_text(html: &str) -> String {
    let without_comments = remove_html_comments(html);
    let mut cleaned = without_comments;
    for tag in [
        "script", "style", "nav", "header", "footer", "aside", "noscript", "svg",
    ] {
        cleaned = remove_html_element_blocks(&cleaned, tag);
    }

    let title = extract_html_tag_text(&cleaned, "title");
    let fragment = extract_html_body_fragment(&cleaned);
    let body_text = normalize_extracted_html_text(&strip_html_tags(&fragment));

    match title {
        Some(title) if !title.is_empty() && !body_text.is_empty() => {
            let title_lower = title.to_ascii_lowercase();
            let body_lower = body_text.to_ascii_lowercase();
            if body_lower.contains(&title_lower) {
                body_text
            } else {
                format!("{title}\n\n{body_text}")
            }
        }
        Some(title) if !title.is_empty() => title,
        _ => body_text,
    }
}

fn remove_html_comments(html: &str) -> String {
    let mut remaining = html.to_string();
    loop {
        let Some(start) = remaining.find("<!--") else {
            break;
        };
        let Some(end_rel) = remaining[start + 4..].find("-->") else {
            remaining.truncate(start);
            break;
        };
        let end = start + 4 + end_rel + 3;
        remaining.replace_range(start..end, " ");
    }
    remaining
}

fn remove_html_element_blocks(html: &str, tag: &str) -> String {
    let mut remaining = html.to_string();
    let open_tag = format!("<{tag}");
    let close_tag = format!("</{tag}>");
    loop {
        let lower = remaining.to_ascii_lowercase();
        let Some(start) = lower.find(&open_tag) else {
            break;
        };
        let Some(end_rel) = lower[start..].find(&close_tag) else {
            remaining.truncate(start);
            break;
        };
        let end = start + end_rel + close_tag.len();
        remaining.replace_range(start..end, " ");
    }
    remaining
}

fn extract_html_tag_text(html: &str, tag: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find(&format!("<{tag}"))?;
    let open_end = html[start..].find('>')? + start + 1;
    let end = lower[open_end..].find(&format!("</{tag}>"))? + open_end;
    let extracted = normalize_extracted_html_text(&strip_html_tags(&html[open_end..end]));
    if extracted.is_empty() {
        None
    } else {
        Some(extracted)
    }
}

fn extract_html_body_fragment(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let Some(start) = lower.find("<body") else {
        return html.to_string();
    };
    let Some(open_end_rel) = html[start..].find('>') else {
        return html.to_string();
    };
    let body_start = start + open_end_rel + 1;
    let Some(close_rel) = lower[body_start..].find("</body>") else {
        return html[body_start..].to_string();
    };
    html[body_start..body_start + close_rel].to_string()
}

fn strip_html_tags(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag = String::new();

    for ch in html.chars() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let tag_name = parse_html_tag_name(&tag);
                if is_html_block_break_tag(&tag_name) {
                    text.push('\n');
                } else if !matches!(text.chars().last(), None | Some(' ') | Some('\n')) {
                    text.push(' ');
                }
                tag.clear();
            } else {
                tag.push(ch);
            }
            continue;
        }

        if ch == '<' {
            in_tag = true;
            continue;
        }

        text.push(ch);
    }

    decode_html_entities(&text)
}

fn parse_html_tag_name(tag: &str) -> String {
    tag.trim()
        .trim_start_matches('/')
        .trim_start_matches('!')
        .split(|ch: char| ch.is_whitespace() || ch == '/')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn is_html_block_break_tag(tag: &str) -> bool {
    matches!(
        tag,
        "html"
            | "head"
            | "body"
            | "title"
            | "main"
            | "section"
            | "article"
            | "div"
            | "p"
            | "br"
            | "li"
            | "ul"
            | "ol"
            | "table"
            | "thead"
            | "tbody"
            | "tr"
            | "td"
            | "th"
            | "pre"
            | "blockquote"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
    )
}

fn decode_html_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn normalize_extracted_html_text(text: &str) -> String {
    let mut normalized_lines = Vec::new();
    let mut previous_blank = false;

    for line in text.lines() {
        let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            if !previous_blank && !normalized_lines.is_empty() {
                normalized_lines.push(String::new());
            }
            previous_blank = true;
            continue;
        }
        normalized_lines.push(collapsed);
        previous_blank = false;
    }

    normalized_lines.join("\n").trim().to_string()
}

fn compression_bias(options: &CompressOptions) -> f64 {
    options
        .target_ratio
        .map(|ratio| (ratio / 0.35).clamp(0.5, 2.0))
        .unwrap_or(1.0)
}

fn extract_ccr_hashes(text: &str) -> Vec<String> {
    let mut hashes = Vec::new();
    let mut start = 0;
    while let Some(relative) = text[start..].find("<<ccr:") {
        let hash_start = start + relative + "<<ccr:".len();
        let hash: String = text[hash_start..]
            .chars()
            .take_while(|ch| ch.is_ascii_hexdigit())
            .collect();
        let hash_len = hash.len();
        if !hash.is_empty() && !hashes.contains(&hash) {
            hashes.push(hash);
        }
        start = hash_start.saturating_add(hash_len);
    }
    hashes
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompressContentKind {
    AlreadyCompressed,
    Diff,
    Html,
    SearchResults,
    BuildOutput,
    StructuredJson,
    SourceCode,
    PlainText,
}

#[derive(Clone, Debug, PartialEq)]
struct ContentDetection {
    kind: CompressContentKind,
    confidence: f64,
    metadata: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MixedContentSection {
    CodeFence { language: String, content: String },
    CodeBlock(String),
    Raw(String),
}

fn detect_text_content_kind(text: &str) -> CompressContentKind {
    detect_text_content(text).kind
}

fn detect_text_content(text: &str) -> ContentDetection {
    if text.contains("Retrieve more: hash=") || text.contains("Retrieve original: hash=") {
        return ContentDetection {
            kind: CompressContentKind::AlreadyCompressed,
            confidence: 1.0,
            metadata: Map::new(),
        };
    }
    if let Some(result) = try_detect_structured_json(text) {
        return result;
    }
    if let Some(result) = try_detect_diff(text).filter(|result| result.confidence >= 0.7) {
        return result;
    }
    if let Some(result) = try_detect_html(text).filter(|result| result.confidence >= 0.7) {
        return result;
    }
    if let Some(result) = try_detect_search_results(text).filter(|result| result.confidence >= 0.6)
    {
        return result;
    }
    if let Some(result) = try_detect_build_output(text).filter(|result| result.confidence >= 0.5) {
        return result;
    }
    if let Some(result) = try_detect_source_code(text).filter(|result| result.confidence >= 0.5) {
        return result;
    }
    ContentDetection {
        kind: CompressContentKind::PlainText,
        confidence: if text.trim().is_empty() { 0.0 } else { 0.5 },
        metadata: Map::new(),
    }
}

fn try_detect_structured_json(text: &str) -> Option<ContentDetection> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('[') {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(trimmed).ok()?;
    match parsed {
        Value::Array(items) => {
            let is_dict_array = items.iter().all(Value::is_object);
            let mut metadata = Map::new();
            metadata.insert("item_count".to_string(), Value::from(items.len()));
            metadata.insert("is_dict_array".to_string(), Value::Bool(is_dict_array));
            Some(ContentDetection {
                kind: CompressContentKind::StructuredJson,
                confidence: if is_dict_array { 1.0 } else { 0.8 },
                metadata,
            })
        }
        _ => None,
    }
}

fn is_mixed_content(text: &str) -> bool {
    let has_code_fence = text.lines().any(|line| code_fence_language(line).is_some());
    let has_source_code_block = contains_unfenced_source_code_block(text);
    let has_json_block = text
        .lines()
        .any(|line| line.trim_start().starts_with('[') || line.trim_start().starts_with('{'));
    let has_search_results = text.lines().any(|line| is_search_result_line(line.trim()));
    let prose_hits = text
        .split(|ch: char| ch == '\n' || ch == '.' || ch == '!' || ch == '?')
        .filter(|segment| {
            let words = segment.split_whitespace().collect::<Vec<_>>();
            words.len() >= 3
                && words
                    .first()
                    .and_then(|word| word.chars().next())
                    .is_some_and(|ch| ch.is_ascii_uppercase())
        })
        .count();
    let indicators = [
        has_code_fence,
        has_source_code_block,
        has_json_block,
        has_search_results,
        prose_hits > 5,
    ];
    indicators.into_iter().filter(|present| *present).count() >= 2
}

fn contains_unfenced_source_code_block(text: &str) -> bool {
    let lines = text.lines().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < lines.len() {
        if let Some((_, _end_index)) = extract_source_code_block(&lines, index) {
            return true;
        }
        index += 1;
    }
    false
}

fn split_mixed_content_sections(text: &str) -> Vec<MixedContentSection> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut sections = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let line = lines[index];
        if let Some(language) = code_fence_language(line) {
            let start = index + 1;
            index += 1;
            while index < lines.len() && !lines[index].trim_start().starts_with("```") {
                index += 1;
            }
            let content = lines[start..index].join("\n");
            sections.push(MixedContentSection::CodeFence { language, content });
            if index < lines.len() {
                index += 1;
            }
            continue;
        }

        if let Some((json_block, end_index)) = extract_json_block(&lines, index) {
            sections.push(MixedContentSection::Raw(json_block));
            index = end_index + 1;
            continue;
        }

        if let Some((code_block, end_index)) = extract_source_code_block(&lines, index) {
            sections.push(MixedContentSection::CodeBlock(code_block));
            index = end_index + 1;
            continue;
        }

        if is_search_result_line(line.trim()) {
            let start = index;
            index += 1;
            while index < lines.len() && is_search_result_line(lines[index].trim()) {
                index += 1;
            }
            sections.push(MixedContentSection::Raw(lines[start..index].join("\n")));
            continue;
        }

        let start = index;
        index += 1;
        while index < lines.len()
            && code_fence_language(lines[index]).is_none()
            && extract_json_block(&lines, index).is_none()
            && extract_source_code_block(&lines, index).is_none()
            && !is_search_result_line(lines[index].trim())
        {
            index += 1;
        }
        let raw = lines[start..index].join("\n");
        if !raw.trim().is_empty() {
            sections.push(MixedContentSection::Raw(raw));
        }
    }
    sections
}

fn extract_source_code_block(lines: &[&str], start: usize) -> Option<(String, usize)> {
    let first = lines.get(start)?.trim();
    if first.is_empty() || code_fence_language(first).is_some() || is_search_result_line(first) {
        return None;
    }

    let mut candidate_lines = Vec::new();
    let mut matched_end = None;
    for end in start..lines.len().min(start + 80) {
        let line = lines[end];
        if end > start && line.trim().is_empty() {
            break;
        }
        candidate_lines.push(line);
        if candidate_lines.len() < 3 {
            continue;
        }
        let candidate = candidate_lines.join("\n");
        if matches!(
            detect_text_content_kind(&candidate),
            CompressContentKind::SourceCode
        ) {
            matched_end = Some(end);
        }
    }

    matched_end.map(|end| (lines[start..=end].join("\n"), end))
}

fn extract_json_block(lines: &[&str], start: usize) -> Option<(String, usize)> {
    let first = lines.get(start)?.trim_start();
    if !first.starts_with('[') && !first.starts_with('{') {
        return None;
    }

    let mut candidate = String::new();
    for end in start..lines.len().min(start + 200) {
        if end > start {
            candidate.push('\n');
        }
        candidate.push_str(lines[end]);
        let trimmed = candidate.trim();
        if matches!(
            serde_json::from_str::<Value>(trimmed),
            Ok(Value::Array(_) | Value::Object(_))
        ) {
            return Some((trimmed.to_string(), end));
        }
        if end > start && lines[end].trim().is_empty() {
            break;
        }
    }
    None
}

fn code_fence_language(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("```")?;
    Some(if rest.trim().is_empty() {
        "unknown".to_string()
    } else {
        rest.trim().to_string()
    })
}

fn content_contains_already_compressed_marker(content: &Value) -> bool {
    content_text_segments(content).into_iter().any(|text| {
        matches!(
            detect_text_content_kind(text),
            CompressContentKind::AlreadyCompressed
        )
    })
}

fn content_contains_source_code(content: &Value) -> bool {
    content_text_segments(content).into_iter().any(|text| {
        matches!(
            detect_text_content_kind(text),
            CompressContentKind::SourceCode
        )
    })
}

fn content_contains_html(content: &Value) -> bool {
    content_text_segments(content)
        .into_iter()
        .any(|text| matches!(detect_text_content_kind(text), CompressContentKind::Html))
}

fn content_text_segments(content: &Value) -> Vec<&str> {
    match content {
        Value::String(text) => vec![text.as_str()],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                let object = part.as_object()?;
                let part_type = object.get("type").and_then(Value::as_str).unwrap_or("");
                if part_type != "text" && part_type != "input_text" && part_type != "output_text" {
                    return None;
                }
                object.get("text").and_then(Value::as_str)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn try_detect_diff(text: &str) -> Option<ContentDetection> {
    let mut header_matches = 0;
    let mut change_matches = 0;
    for line in text.lines().take(500) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("diff --git")
            || trimmed.starts_with("diff --combined ")
            || trimmed.starts_with("diff --cc ")
            || trimmed.starts_with("--- a/")
            || trimmed.starts_with("@@ ")
            || trimmed.starts_with("@@@")
        {
            header_matches += 1;
        }
        if (line.starts_with('+') || line.starts_with('-'))
            && !line.starts_with("+++")
            && !line.starts_with("---")
        {
            change_matches += 1;
        }
    }
    if header_matches == 0 {
        return None;
    }
    let confidence =
        (0.5 + (header_matches as f64 * 0.2) + (change_matches as f64 * 0.05)).min(1.0);
    let mut metadata = Map::new();
    metadata.insert("header_matches".to_string(), Value::from(header_matches));
    metadata.insert("change_lines".to_string(), Value::from(change_matches));
    Some(ContentDetection {
        kind: CompressContentKind::Diff,
        confidence,
        metadata,
    })
}

fn try_detect_source_code(text: &str) -> Option<ContentDetection> {
    let mut python_matches = 0usize;
    let mut javascript_matches = 0usize;
    let mut typescript_matches = 0usize;
    let mut go_matches = 0usize;
    let mut rust_matches = 0usize;
    let mut java_matches = 0usize;
    let mut non_empty_lines = 0usize;
    for line in text.lines().take(100) {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        non_empty_lines += 1;

        if trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("async def ")
            || trimmed.starts_with("if __name__ ==")
            || trimmed.starts_with("\"\"\"")
        {
            python_matches += 1;
        }
        if trimmed.starts_with("function ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("var ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("export ")
            || trimmed.starts_with("async function")
            || trimmed.starts_with("module.exports")
        {
            javascript_matches += 1;
        }
        if trimmed.starts_with("interface ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("namespace ")
            || trimmed.contains(": string")
            || trimmed.contains(": number")
            || trimmed.contains(": boolean")
            || trimmed.contains(": any")
            || trimmed.contains(": void")
        {
            typescript_matches += 1;
        }
        if trimmed.starts_with("func ")
            || trimmed.starts_with("type ")
            || trimmed.starts_with("package ")
            || trimmed == "import"
            || trimmed.starts_with("import (")
        {
            go_matches += 1;
        }
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("mod ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("pub ")
            || trimmed.starts_with("#[")
        {
            rust_matches += 1;
        }
        if trimmed.starts_with("public class ")
            || trimmed.starts_with("private class ")
            || trimmed.starts_with("protected class ")
            || trimmed.starts_with("public interface ")
            || trimmed.starts_with("private interface ")
            || trimmed.starts_with("protected interface ")
            || trimmed.starts_with("public enum ")
            || trimmed.starts_with("private enum ")
            || trimmed.starts_with("protected enum ")
            || trimmed.starts_with("package ")
            || trimmed.starts_with('@')
        {
            java_matches += 1;
        }
    }

    let language_scores = [
        ("python", python_matches),
        ("javascript", javascript_matches),
        ("typescript", typescript_matches),
        ("go", go_matches),
        ("rust", rust_matches),
        ("java", java_matches),
    ];
    let (language, best_score) = language_scores
        .into_iter()
        .max_by_key(|(_, score)| *score)
        .unwrap_or(("python", 0));
    if best_score < 3 {
        return None;
    }
    let ratio = best_score as f64 / non_empty_lines.max(1) as f64;
    let confidence = (0.4 + (ratio * 0.4) + (best_score as f64 * 0.02)).min(1.0);
    let mut metadata = Map::new();
    metadata.insert("language".to_string(), Value::String(language.to_string()));
    metadata.insert("pattern_matches".to_string(), Value::from(best_score));
    Some(ContentDetection {
        kind: CompressContentKind::SourceCode,
        confidence,
        metadata,
    })
}

fn try_detect_html(text: &str) -> Option<ContentDetection> {
    let sample = text
        .chars()
        .take(3000)
        .collect::<String>()
        .to_ascii_lowercase();
    let has_doctype = sample.contains("<!doctype html");
    let has_html_tag = sample.contains("<html");
    let has_head = sample.contains("<head");
    let has_body = sample.contains("<body");
    let structural_tags = [
        "<div", "<span", "<script", "<style", "<link", "<meta", "<nav", "<header", "<footer",
        "<aside", "<article", "<section", "<main",
    ];
    let structural_matches = structural_tags
        .iter()
        .filter(|tag| sample.contains(**tag))
        .count();
    if !has_doctype && !has_html_tag && structural_matches < 3 {
        return None;
    }
    let mut confidence = 0.0;
    if has_doctype {
        confidence += 0.5;
    }
    if has_html_tag {
        confidence += 0.3;
    }
    if has_head {
        confidence += 0.1;
    }
    if has_body {
        confidence += 0.1;
    }
    confidence += (structural_matches as f64 * 0.03).min(0.3);
    confidence = confidence.min(1.0);
    if confidence < 0.5 || !sample.contains("</") {
        return None;
    }
    let mut metadata = Map::new();
    metadata.insert("has_doctype".to_string(), Value::Bool(has_doctype));
    metadata.insert("has_html_tag".to_string(), Value::Bool(has_html_tag));
    metadata.insert(
        "structural_tags".to_string(),
        Value::from(structural_matches),
    );
    Some(ContentDetection {
        kind: CompressContentKind::Html,
        confidence,
        metadata,
    })
}

fn try_detect_search_results(text: &str) -> Option<ContentDetection> {
    let mut non_empty_lines = 0usize;
    let mut matching_lines = 0usize;
    for line in text.lines().take(100) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        non_empty_lines += 1;
        if is_search_result_line(trimmed) {
            matching_lines += 1;
        }
    }
    if matching_lines == 0 || non_empty_lines == 0 {
        return None;
    }
    let ratio = matching_lines as f64 / non_empty_lines as f64;
    if ratio < 0.3 {
        return None;
    }
    let mut metadata = Map::new();
    metadata.insert("matching_lines".to_string(), Value::from(matching_lines));
    metadata.insert("total_lines".to_string(), Value::from(non_empty_lines));
    Some(ContentDetection {
        kind: CompressContentKind::SearchResults,
        confidence: (0.4 + (ratio * 0.6)).min(1.0),
        metadata,
    })
}

fn is_search_result_line(line: &str) -> bool {
    let Some((prefix, rest)) = line.split_once(':') else {
        return false;
    };
    if prefix.is_empty() || prefix.contains(char::is_whitespace) {
        return false;
    }
    let Some((line_number, _)) = rest.split_once(':') else {
        return false;
    };
    !line_number.is_empty() && line_number.chars().all(|ch| ch.is_ascii_digit())
}

fn try_detect_build_output(text: &str) -> Option<ContentDetection> {
    let mut non_empty_lines = 0usize;
    let mut matching_lines = 0usize;
    let mut error_matches = 0usize;
    for line in text.lines().take(200) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        non_empty_lines += 1;
        if let Some(classification) = classify_log_line(trimmed) {
            matching_lines += 1;
            if classification.counts_as_error_match {
                error_matches += 1;
            }
        }
    }
    if matching_lines == 0 || non_empty_lines == 0 {
        return None;
    }
    let ratio = matching_lines as f64 / non_empty_lines as f64;
    if ratio < 0.1 {
        return None;
    }
    let mut metadata = Map::new();
    metadata.insert("pattern_matches".to_string(), Value::from(matching_lines));
    metadata.insert("error_matches".to_string(), Value::from(error_matches));
    metadata.insert("total_lines".to_string(), Value::from(non_empty_lines));
    Some(ContentDetection {
        kind: CompressContentKind::BuildOutput,
        confidence: (0.3 + (ratio * 0.5) + (error_matches as f64 * 0.05)).min(1.0),
        metadata,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LogLineClassification {
    counts_as_error_match: bool,
}

fn classify_log_line(line: &str) -> Option<LogLineClassification> {
    let lower = line.to_ascii_lowercase();
    let upper = line.to_ascii_uppercase();
    if upper.contains("ERROR")
        || upper.contains("FAIL")
        || upper.contains("FAILED")
        || upper.contains("FATAL")
        || upper.contains("CRITICAL")
    {
        return Some(LogLineClassification {
            counts_as_error_match: true,
        });
    }
    if upper.contains("WARN") || upper.contains("WARNING") {
        return Some(LogLineClassification {
            counts_as_error_match: true,
        });
    }
    if upper.contains("INFO")
        || upper.contains("DEBUG")
        || upper.contains("TRACE")
        || starts_with_date(line)
        || starts_with_time_block(line)
        || line.starts_with("===")
        || line.starts_with("---")
        || upper.starts_with("PASSED")
        || upper.starts_with("SKIPPED")
        || lower.starts_with("npm err!")
        || lower.starts_with("yarn error")
        || lower.starts_with("cargo error")
        || line.contains("Traceback (most recent call last)")
        || (line.starts_with("at ") && line.contains('('))
    {
        return Some(LogLineClassification {
            counts_as_error_match: false,
        });
    }
    None
}

fn starts_with_date(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= 10
        && bytes[0..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit)
}

fn starts_with_time_block(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= 10
        && bytes[0] == b'['
        && bytes[1..3].iter().all(u8::is_ascii_digit)
        && bytes[3] == b':'
        && bytes[4..6].iter().all(u8::is_ascii_digit)
        && bytes[6] == b':'
        && bytes[7..9].iter().all(u8::is_ascii_digit)
        && bytes[9] == b']'
}

fn detect_analysis_intent(query: &str) -> bool {
    if query.is_empty() {
        return false;
    }
    let query = query.to_ascii_lowercase();
    [
        "analyze",
        "analyse",
        "review",
        "audit",
        "inspect",
        "security",
        "vulnerability",
        "bug",
        "issue",
        "problem",
        "explain",
        "understand",
        "how does",
        "what does",
        "debug",
        "fix",
        "error",
        "wrong",
        "broken",
        "refactor",
        "improve",
        "optimize",
        "clean up",
    ]
    .iter()
    .any(|keyword| query.contains(keyword))
}

fn extract_user_query(messages: &[Value]) -> String {
    for message in messages.iter().rev() {
        let Some(object) = message.as_object() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        match object.get("content") {
            Some(Value::String(text)) if !text.trim().is_empty() => {
                return text.trim().to_string();
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    let Some(block) = part.as_object() else {
                        continue;
                    };
                    let part_type = block.get("type").and_then(Value::as_str).unwrap_or("");
                    if part_type != "text"
                        && part_type != "input_text"
                        && part_type != "output_text"
                    {
                        continue;
                    }
                    let Some(text) = block.get("text").and_then(Value::as_str) else {
                        continue;
                    };
                    if !text.trim().is_empty() {
                        return text.trim().to_string();
                    }
                }
            }
            _ => {}
        }
    }
    String::new()
}

fn count_message_tokens(messages: &[Value], tokenizer: &dyn Tokenizer) -> usize {
    messages
        .iter()
        .map(|message| {
            message
                .as_object()
                .and_then(|object| object.get("content"))
                .map(|content| tokenizer.count_text(&render_content_for_token_count(content)))
                .unwrap_or(0)
        })
        .sum()
}

fn apply_token_budget_window(
    mut messages: Vec<Value>,
    tokenizer: &dyn Tokenizer,
    model_limit: usize,
) -> (Vec<Value>, Vec<String>) {
    let available = model_limit.saturating_sub(4000);
    let current_tokens = count_message_tokens(&messages, tokenizer);
    if current_tokens <= available {
        return (messages, Vec::new());
    }

    let protected = protected_window_indices(&messages);
    let drop_candidates = build_window_drop_candidates(&messages, &protected);
    let mut dropped = 0usize;
    let mut current_tokens = current_tokens;
    let mut indices_to_drop = HashSet::new();
    for candidate in drop_candidates {
        if current_tokens <= available {
            break;
        }
        if candidate
            .iter()
            .any(|index| protected.contains(index) || indices_to_drop.contains(index))
        {
            continue;
        }
        let tokens_saved = candidate
            .iter()
            .filter_map(|index| messages.get(*index))
            .map(|message| count_message_tokens(std::slice::from_ref(message), tokenizer))
            .sum::<usize>();
        indices_to_drop.extend(candidate);
        current_tokens = current_tokens.saturating_sub(tokens_saved);
        dropped += 1;
    }

    let mut indices_to_drop = indices_to_drop.into_iter().collect::<Vec<_>>();
    indices_to_drop.sort_unstable_by(|a, b| b.cmp(a));
    for index in indices_to_drop {
        if index < messages.len() {
            messages.remove(index);
        }
    }

    if dropped == 0 {
        return (messages, Vec::new());
    }

    let insert_index = messages
        .iter()
        .position(|message| message.get("role").and_then(Value::as_str) != Some("system"))
        .unwrap_or(messages.len());
    let uses_block_format = messages.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("user")
            && message.get("content").map(Value::is_array).unwrap_or(false)
    });
    let marker = create_dropped_context_marker("token_cap", dropped);
    let marker_content = if uses_block_format {
        Value::Array(vec![serde_json::json!({
            "type": "text",
            "text": marker,
        })])
    } else {
        Value::String(marker)
    };
    messages.insert(
        insert_index,
        serde_json::json!({
            "role": "user",
            "content": marker_content,
        }),
    );

    (messages, vec![format!("window_cap:{dropped}")])
}

fn protected_window_indices(messages: &[Value]) -> HashSet<usize> {
    let mut protected = HashSet::new();
    for (index, message) in messages.iter().enumerate() {
        if message.get("role").and_then(Value::as_str) == Some("system") {
            protected.insert(index);
        }
    }

    let mut protected_user_turns = 0usize;
    let mut index = messages.len();
    while index > 0 && protected_user_turns < 2 {
        index -= 1;
        protected.insert(index);
        if messages[index].get("role").and_then(Value::as_str) == Some("user") {
            protected_user_turns += 1;
        }
    }

    let initially_protected = protected.iter().copied().collect::<Vec<_>>();
    for index in initially_protected {
        let Some(message) = messages.get(index) else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }

        let tool_call_ids = message_tool_call_ids(message);
        if tool_call_ids.is_empty() {
            continue;
        }

        for (other_index, other_message) in messages.iter().enumerate() {
            let role = other_message.get("role").and_then(Value::as_str);
            if role == Some("tool")
                && other_message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(|id| tool_call_ids.contains(id))
                    .unwrap_or(false)
            {
                protected.insert(other_index);
            }
            if role == Some("user") && user_message_has_tool_result(other_message, &tool_call_ids) {
                protected.insert(other_index);
            }
        }
    }
    protected
}

fn build_window_drop_candidates(messages: &[Value], protected: &HashSet<usize>) -> Vec<Vec<usize>> {
    let tool_units = find_window_tool_units(messages);
    let mut candidates = Vec::new();
    let mut tool_unit_indices = HashSet::new();

    for (assistant_index, response_indices) in &tool_units {
        tool_unit_indices.insert(*assistant_index);
        tool_unit_indices.extend(response_indices.iter().copied());
    }

    for (assistant_index, response_indices) in tool_units {
        if protected.contains(&assistant_index) {
            continue;
        }
        let mut indices = Vec::with_capacity(response_indices.len() + 1);
        indices.push(assistant_index);
        indices.extend(response_indices);
        candidates.push(indices);
    }

    let mut index = 0usize;
    while index < messages.len() {
        if protected.contains(&index) || tool_unit_indices.contains(&index) {
            index += 1;
            continue;
        }
        let role = messages[index].get("role").and_then(Value::as_str);
        if matches!(role, Some("user" | "assistant")) {
            if role == Some("user")
                && index + 1 < messages.len()
                && !tool_unit_indices.contains(&(index + 1))
                && messages[index + 1].get("role").and_then(Value::as_str) == Some("assistant")
            {
                candidates.push(vec![index, index + 1]);
                index += 2;
                continue;
            }
            candidates.push(vec![index]);
        }
        index += 1;
    }

    candidates
}

fn find_window_tool_units(messages: &[Value]) -> Vec<(usize, Vec<usize>)> {
    let mut units = Vec::new();
    for (assistant_index, message) in messages.iter().enumerate() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let tool_call_ids = message_tool_call_ids(message);
        if tool_call_ids.is_empty() {
            continue;
        }
        let mut response_indices = Vec::new();
        for (response_index, response) in messages.iter().enumerate().skip(assistant_index + 1) {
            let role = response.get("role").and_then(Value::as_str);
            if role == Some("tool")
                && response
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(|id| tool_call_ids.contains(id))
                    .unwrap_or(false)
            {
                response_indices.push(response_index);
            }
            if role == Some("user") && user_message_has_tool_result(response, &tool_call_ids) {
                response_indices.push(response_index);
            }
        }
        units.push((assistant_index, response_indices));
    }
    units
}

fn message_tool_call_ids(message: &Value) -> HashSet<String> {
    let mut ids = HashSet::new();
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
                ids.insert(id.to_string());
            }
        }
    }
    if let Some(content) = message.get("content").and_then(Value::as_array) {
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                if let Some(id) = block.get("id").and_then(Value::as_str) {
                    ids.insert(id.to_string());
                }
            }
        }
    }
    ids
}

fn user_message_has_tool_result(message: &Value, tool_call_ids: &HashSet<String>) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .map(|content| {
            content.iter().any(|block| {
                block.get("type").and_then(Value::as_str) == Some("tool_result")
                    && block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .map(|id| tool_call_ids.contains(id))
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn create_dropped_context_marker(reason: &str, count: usize) -> String {
    format!("<headroom:dropped_context reason=\"{reason}\" count=\"{count}\">")
}

fn count_content_tokens(content: &Value, tokenizer: &dyn Tokenizer) -> usize {
    tokenizer.count_text(&render_content_for_token_count(content))
}

fn render_content_for_token_count(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn parse_gemini_generate_content_request(body: &Bytes) -> Result<(usize, bool), ProxyError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let contents_count = match object.get("contents") {
        Some(Value::Array(contents)) => contents.len(),
        Some(_) => {
            return Err(ProxyError::InvalidRequest(
                "field `contents` must be an array".to_string(),
            ))
        }
        None => 0,
    };
    let system_instruction_present = validate_gemini_system_instruction_field(object)?;
    Ok((contents_count, system_instruction_present))
}

struct GeminiRequestOptimization {
    primary_body_bytes: Bytes,
    compression_status: &'static str,
}

struct GoogleCloudCodeStreamOptimization {
    primary_body_bytes: Bytes,
    compression_status: &'static str,
}

fn optimize_gemini_generate_content_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
) -> Result<GeminiRequestOptimization, ProxyError> {
    optimize_gemini_request_body(
        compression_cache,
        product_store,
        telemetry_store,
        body_bytes,
        model,
    )
}

fn optimize_gemini_count_tokens_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
) -> Result<GeminiRequestOptimization, ProxyError> {
    optimize_gemini_request_body(
        compression_cache,
        product_store,
        telemetry_store,
        body_bytes,
        model,
    )
}

fn optimize_gemini_request_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
) -> Result<GeminiRequestOptimization, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body_bytes)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let contents = object
        .get("contents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (messages, preserved_indices) =
        gemini_contents_to_messages(&contents, object.get("systemInstruction"));
    if contents.is_empty() || messages.is_empty() || preserved_indices.len() == contents.len() {
        return Ok(GeminiRequestOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
        });
    }

    let mut compression_body = object.clone();
    let mut compression_config = compression_body
        .get("config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
    compression_body.insert("config".to_string(), Value::Object(compression_config));

    let Ok(result) = build_compress_response(
        compression_cache,
        product_store,
        telemetry_store,
        &compression_body,
        &messages,
        model,
    ) else {
        return Ok(GeminiRequestOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "failed",
        });
    };
    let compressed_messages = result.body["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if compressed_messages == messages {
        return Ok(GeminiRequestOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "unchanged",
        });
    }

    let (mut optimized_contents, optimized_system_instruction) =
        messages_to_gemini_contents(&compressed_messages);
    for (index, original_content) in contents.iter().enumerate() {
        if preserved_indices.contains(&index) && index < optimized_contents.len() {
            optimized_contents[index] = original_content.clone();
        }
    }
    object.insert("contents".to_string(), Value::Array(optimized_contents));
    match optimized_system_instruction {
        Some(system_instruction) => {
            object.insert("systemInstruction".to_string(), system_instruction);
        }
        None => {
            object.remove("systemInstruction");
        }
    }

    let primary_body_bytes = serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    Ok(GeminiRequestOptimization {
        primary_body_bytes,
        compression_status: "compressed",
    })
}

fn optimize_google_cloudcode_stream_body(
    compression_cache: &Arc<Mutex<CompressionCache>>,
    product_store: &ProductStore,
    telemetry_store: &TelemetryStore,
    body_bytes: &Bytes,
    model: &str,
    antigravity: bool,
) -> Result<GoogleCloudCodeStreamOptimization, ProxyError> {
    let mut payload: Value = serde_json::from_slice(body_bytes)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object_mut().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let request = object
        .get_mut("request")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| ProxyError::InvalidRequest("missing object field `request`".to_string()))?;
    let contents = request
        .get("contents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let system_instruction = if antigravity {
        None
    } else {
        request.get("systemInstruction")
    };
    let (messages, preserved_indices) = gemini_contents_to_messages(&contents, system_instruction);
    if contents.is_empty() || messages.is_empty() || preserved_indices.len() == contents.len() {
        return Ok(GoogleCloudCodeStreamOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "bypassed",
        });
    }

    let mut compression_body = request.clone();
    let mut compression_config = compression_body
        .get("config")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    compression_config.insert("compress_user_messages".to_string(), Value::Bool(true));
    compression_body.insert("config".to_string(), Value::Object(compression_config));

    let Ok(result) = build_compress_response(
        compression_cache,
        product_store,
        telemetry_store,
        &compression_body,
        &messages,
        model,
    ) else {
        return Ok(GoogleCloudCodeStreamOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "failed",
        });
    };
    let compressed_messages = result.body["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if compressed_messages == messages {
        return Ok(GoogleCloudCodeStreamOptimization {
            primary_body_bytes: body_bytes.clone(),
            compression_status: "unchanged",
        });
    }

    let (mut optimized_contents, optimized_system_instruction) =
        messages_to_gemini_contents(&compressed_messages);
    for (index, original_content) in contents.iter().enumerate() {
        if preserved_indices.contains(&index) && index < optimized_contents.len() {
            optimized_contents[index] = original_content.clone();
        }
    }
    request.insert("contents".to_string(), Value::Array(optimized_contents));
    if !antigravity {
        match optimized_system_instruction {
            Some(system_instruction) => {
                request.insert("systemInstruction".to_string(), system_instruction);
            }
            None => {
                request.remove("systemInstruction");
            }
        }
    }

    let primary_body_bytes = serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    Ok(GoogleCloudCodeStreamOptimization {
        primary_body_bytes,
        compression_status: "compressed",
    })
}

fn gemini_contents_to_messages(
    contents: &[Value],
    system_instruction: Option<&Value>,
) -> (Vec<Value>, Vec<usize>) {
    let mut messages = Vec::new();
    let mut preserved_indices = Vec::new();

    if let Some(system_instruction) = system_instruction.and_then(Value::as_object) {
        let text_parts = system_instruction
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !text_parts.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": text_parts.join("\n"),
            }));
        }
    }

    for (index, content) in contents.iter().enumerate() {
        if gemini_content_has_non_text_parts(content) {
            preserved_indices.push(index);
        }
        let Some(content_object) = content.as_object() else {
            continue;
        };
        let role = match content_object.get("role").and_then(Value::as_str) {
            Some("model") => "assistant",
            Some(role) if !role.is_empty() => role,
            _ => "user",
        };
        let text_parts = content_object
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !text_parts.is_empty() {
            messages.push(serde_json::json!({
                "role": role,
                "content": text_parts.join("\n"),
            }));
        }
    }

    (messages, preserved_indices)
}

fn messages_to_gemini_contents(messages: &[Value]) -> (Vec<Value>, Option<Value>) {
    let mut contents = Vec::new();
    let mut system_instruction = None;

    for message in messages {
        let Some(message_object) = message.as_object() else {
            continue;
        };
        let role = message_object
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message_object
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if role == "system" {
            system_instruction = Some(serde_json::json!({
                "parts": [{"text": content}],
            }));
            continue;
        }
        let gemini_role = if role == "assistant" { "model" } else { "user" };
        contents.push(serde_json::json!({
            "role": gemini_role,
            "parts": [{"text": content}],
        }));
    }

    (contents, system_instruction)
}

fn gemini_content_has_non_text_parts(content: &Value) -> bool {
    content
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts.iter().any(|part| {
                part.get("inlineData").is_some()
                    || part.get("fileData").is_some()
                    || part.get("functionCall").is_some()
                    || part.get("functionResponse").is_some()
            })
        })
        .unwrap_or(false)
}

fn parse_google_cloudcode_stream_request(
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<(String, usize, bool), ProxyError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::InvalidRequest("missing string field `model`".to_string()))?;
    let request = object
        .get("request")
        .and_then(Value::as_object)
        .ok_or_else(|| ProxyError::InvalidRequest("missing object field `request`".to_string()))?;
    let contents = request
        .get("contents")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProxyError::InvalidRequest("missing array field `request.contents`".to_string())
        })?;
    let header_user_agent = headers
        .get(http::header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let body_user_agent = object
        .get("userAgent")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let antigravity = object.get("requestType").and_then(Value::as_str) == Some("agent")
        || body_user_agent == "antigravity"
        || header_user_agent.starts_with("antigravity/");
    Ok((model.to_string(), contents.len(), antigravity))
}

fn parse_anthropic_messages_request(
    body: &Bytes,
) -> Result<(String, usize, u64, bool), ProxyError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let model = match object.get("model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => {
            return Err(ProxyError::InvalidRequest(
                "field `model` must be a string".to_string(),
            ))
        }
        None => "unknown".to_string(),
    };
    let max_tokens = match object.get("max_tokens") {
        Some(Value::Number(value)) => value.as_u64().ok_or_else(|| {
            ProxyError::InvalidRequest("field `max_tokens` must be an integer".to_string())
        })?,
        Some(_) => {
            return Err(ProxyError::InvalidRequest(
                "field `max_tokens` must be an integer".to_string(),
            ))
        }
        None => 0,
    };
    let messages_count = match object.get("messages") {
        Some(Value::Array(messages)) => messages.len(),
        Some(_) => {
            return Err(ProxyError::InvalidRequest(
                "field `messages` must be an array".to_string(),
            ))
        }
        None => 0,
    };
    let streaming = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok((model, messages_count, max_tokens, streaming))
}

fn parse_anthropic_count_tokens_request(body: &Bytes) -> Result<(String, usize, bool), ProxyError> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid JSON body: {e}")))?;
    let object = payload.as_object().ok_or_else(|| {
        ProxyError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::InvalidRequest("missing string field `model`".to_string()))?;
    let messages = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProxyError::InvalidRequest("missing array field `messages`".to_string()))?;
    let system_present = object.contains_key("system");
    Ok((model.to_string(), messages.len(), system_present))
}

fn validate_gemini_system_instruction_field(
    object: &Map<String, Value>,
) -> Result<bool, ProxyError> {
    match object.get("systemInstruction") {
        Some(Value::Object(_)) => Ok(true),
        Some(_) => Err(ProxyError::InvalidRequest(
            "field `systemInstruction` must be an object".to_string(),
        )),
        None => Ok(false),
    }
}

fn ensure_request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

fn provider_from_path(path: &str) -> &'static str {
    if path.starts_with("/v1/chat/completions")
        || path.starts_with("/v1/batches")
        || path.starts_with("/v1/responses")
        || path.starts_with("/v1/codex/responses")
        || path.starts_with("/backend-api/responses")
        || path.starts_with("/backend-api/codex/responses")
        || path.starts_with("/v1/embeddings")
        || path.starts_with("/v1/moderations")
        || path.starts_with("/v1/images/")
        || path.starts_with("/v1/audio/")
    {
        "openai"
    } else if path.starts_with("/serving-endpoints/") && path.ends_with("/invocations") {
        "databricks"
    } else if path.starts_with("/v1/messages") {
        "anthropic"
    } else if path == "/v1internal:streamGenerateContent"
        || path == "/v1/v1internal:streamGenerateContent"
    {
        "cloudcode"
    } else if path.starts_with("/v1beta/models") {
        "gemini"
    } else {
        "passthrough"
    }
}

fn model_metadata_provider(headers: &HeaderMap) -> &'static str {
    let authorization = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();

    if headers.contains_key("x-api-key")
        || headers.contains_key("anthropic-version")
        || authorization.starts_with("Bearer sk-ant-")
    {
        "anthropic"
    } else {
        "openai"
    }
}

/// Test-only helper: drain a body to bytes (uses BodyExt).
#[cfg(test)]
pub async fn body_to_bytes(body: Body) -> Result<Bytes, axum::Error> {
    use axum::Error;
    body.collect()
        .await
        .map(|c| c.to_bytes())
        .map_err(Error::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use headroom_runtime::MetadataValue;

    #[test]
    fn url_build_basic() {
        let base: url::Url = "http://up:8080".parse().unwrap();
        let uri: Uri = "/v1/messages?stream=true".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/v1/messages?stream=true");
    }

    #[test]
    fn url_build_with_base_path() {
        let base: url::Url = "http://up:8080/api".parse().unwrap();
        let uri: Uri = "/v1/messages".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/api/v1/messages");
    }

    #[test]
    fn url_build_root() {
        let base: url::Url = "http://up:8080/".parse().unwrap();
        let uri: Uri = "/".parse().unwrap();
        let out = build_upstream_url(&base, &uri).unwrap();
        assert_eq!(out.as_str(), "http://up:8080/");
    }

    #[test]
    fn provider_from_path_infers_known_routes() {
        assert_eq!(provider_from_path("/v1/chat/completions"), "openai");
        assert_eq!(provider_from_path("/v1/batches"), "openai");
        assert_eq!(provider_from_path("/backend-api/codex/responses"), "openai");
        assert_eq!(provider_from_path("/v1/embeddings"), "openai");
        assert_eq!(
            provider_from_path("/serving-endpoints/demo/invocations"),
            "databricks"
        );
        assert_eq!(provider_from_path("/v1/messages"), "anthropic");
        assert_eq!(
            provider_from_path("/v1internal:streamGenerateContent"),
            "cloudcode"
        );
        assert_eq!(provider_from_path("/v1beta/models/gemini"), "gemini");
        assert_eq!(provider_from_path("/internal"), "passthrough");
    }

    #[test]
    fn databricks_invocations_route_extracts_model() {
        let (model, metadata) = databricks_invocations_route(
            &Method::POST,
            "/serving-endpoints/databricks-meta-llama/invocations",
        )
        .unwrap();

        assert_eq!(model, "databricks-meta-llama");
        assert_eq!(
            metadata.get("endpoint"),
            Some(&MetadataValue::String(
                "serving-endpoints/invocations".to_string()
            ))
        );
    }

    #[test]
    fn model_metadata_provider_matches_python_auth_semantics() {
        let mut headers = HeaderMap::new();
        assert_eq!(model_metadata_provider(&headers), "openai");

        headers.insert("x-api-key", HeaderValue::from_static("anthropic-key"));
        assert_eq!(model_metadata_provider(&headers), "anthropic");

        let mut bearer_headers = HeaderMap::new();
        bearer_headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-ant-test"),
        );
        assert_eq!(model_metadata_provider(&bearer_headers), "anthropic");
    }

    #[test]
    fn canonical_openai_responses_path_normalizes_aliases() {
        assert_eq!(
            canonical_openai_responses_path("/backend-api/codex/responses/compact"),
            Some("/v1/responses/compact".to_string())
        );
        assert_eq!(
            canonical_openai_responses_path("/v1/codex/responses"),
            Some("/v1/responses".to_string())
        );
        assert_eq!(
            canonical_openai_responses_path("/v1/responses/output_items"),
            Some("/v1/responses/output_items".to_string())
        );
        assert_eq!(canonical_openai_responses_path("/backend-api/other"), None);
    }

    #[test]
    fn prepare_openai_responses_request_routes_chatgpt_auth_to_chatgpt_backend() {
        let payload = r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_jwt"}}"#;
        let encoded = URL_SAFE
            .encode(payload.as_bytes())
            .trim_end_matches('=')
            .to_string();
        let token = format!("aaa.{encoded}.bbb");
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/codex/responses/compact?trace=jwt")
            .header(http::header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();

        let (req, metadata, upstream_base) = prepare_openai_responses_request(
            req,
            &url::Url::parse("https://api.openai.example/v1").unwrap(),
        )
        .unwrap();

        assert_eq!(req.uri().path(), "/backend-api/codex/responses/compact");
        assert_eq!(req.uri().query(), Some("trace=jwt"));
        assert_eq!(req.headers().get("chatgpt-account-id").unwrap(), "acct_jwt");
        assert_eq!(upstream_base.as_str(), "https://chatgpt.com/");
        assert_eq!(
            metadata.get("canonical_path"),
            Some(&MetadataValue::String("/v1/responses/compact".to_string()))
        );
        assert_eq!(
            metadata.get("chatgpt_auth"),
            Some(&MetadataValue::Bool(true))
        );
        assert_eq!(
            metadata.get("target_path"),
            Some(&MetadataValue::String(
                "/backend-api/codex/responses/compact".to_string()
            ))
        );
    }

    #[test]
    fn apply_openai_responses_websocket_defaults_injects_missing_auth_and_beta() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::UPGRADE, HeaderValue::from_static("websocket"));
        headers.insert(
            http::header::CONNECTION,
            HeaderValue::from_static("Upgrade"),
        );

        apply_openai_responses_websocket_defaults(&mut headers, Some("sk-test")).unwrap();

        assert_eq!(
            headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer sk-test"
        );
        assert_eq!(
            headers.get("openai-beta").unwrap(),
            "responses_websockets=2026-02-06"
        );
    }

    #[test]
    fn apply_openai_responses_websocket_defaults_preserves_existing_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::UPGRADE, HeaderValue::from_static("websocket"));
        headers.insert(
            http::header::CONNECTION,
            HeaderValue::from_static("Upgrade"),
        );
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer client-token"),
        );
        headers.insert(
            HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("responses_websockets=custom-version"),
        );

        apply_openai_responses_websocket_defaults(&mut headers, Some("sk-test")).unwrap();

        assert_eq!(
            headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer client-token"
        );
        assert_eq!(
            headers.get("openai-beta").unwrap(),
            "responses_websockets=custom-version"
        );
    }

    #[test]
    fn resolve_chatgpt_account_id_prefers_header_or_jwt() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("chatgpt-account-id"),
            HeaderValue::from_static("acct_header"),
        );
        assert_eq!(
            resolve_chatgpt_account_id(&headers),
            Some("acct_header".to_string())
        );

        let payload = r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_jwt"}}"#;
        let encoded = URL_SAFE
            .encode(payload.as_bytes())
            .trim_end_matches('=')
            .to_string();
        let token = format!("aaa.{encoded}.bbb");
        let mut jwt_headers = HeaderMap::new();
        jwt_headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        assert_eq!(
            resolve_chatgpt_account_id(&jwt_headers),
            Some("acct_jwt".to_string())
        );
    }

    #[test]
    fn parse_openai_chat_request_extracts_model_and_message_count() {
        let body = Bytes::from_static(
            br#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        );
        let (model, messages_count, streaming) = parse_openai_chat_request(&body).unwrap();
        assert_eq!(model, "gpt-4o-mini");
        assert_eq!(messages_count, 1);
        assert!(streaming);
    }

    #[test]
    fn optimize_openai_chat_body_compresses_large_text_messages() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let messages = (0..6)
            .map(|index| {
                serde_json::json!({
                    "role": "user",
                    "content": format!("{long_text}\nmessage {index}"),
                })
            })
            .collect::<Vec<_>>();
        let body = Bytes::from(
            serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": messages,
                "stream": true,
                "stream_options": {"include_usage": true},
            })
            .to_string(),
        );

        let optimization = optimize_openai_chat_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "gpt-4o-mini",
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        let optimized_content = optimized["messages"][0]["content"].as_str().unwrap();
        assert_ne!(optimized_content, format!("{long_text}\nmessage 0"));
        assert_eq!(optimized["stream"], serde_json::json!(true));
        assert_eq!(
            optimized["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn apply_token_budget_window_inserts_python_style_dropped_context_marker() {
        let tokenizer = get_tokenizer("gpt-4o-mini");
        let messages = vec![
            serde_json::json!({"role": "system", "content": "system prompt"}),
            serde_json::json!({"role": "user", "content": "old context ".repeat(200)}),
            serde_json::json!({"role": "assistant", "content": "assistant reply ".repeat(200)}),
            serde_json::json!({"role": "user", "content": "middle question ".repeat(200)}),
            serde_json::json!({"role": "assistant", "content": "middle answer ".repeat(200)}),
            serde_json::json!({"role": "user", "content": "recent question ".repeat(200)}),
            serde_json::json!({"role": "assistant", "content": "recent answer ".repeat(200)}),
        ];

        let (windowed, transforms) = apply_token_budget_window(messages, tokenizer.as_ref(), 4000);

        assert!(windowed.iter().any(|message| {
            message["content"]
                .as_str()
                .unwrap_or_default()
                .starts_with("<headroom:dropped_context reason=\"token_cap\"")
        }));
        assert_eq!(transforms.len(), 1);
        assert!(transforms[0].starts_with("window_cap:"));
    }

    #[test]
    fn apply_token_budget_window_drops_tool_units_atomically() {
        let tokenizer = get_tokenizer("gpt-4");
        let messages = vec![
            serde_json::json!({"role": "system", "content": "system prompt"}),
            serde_json::json!({"role": "user", "content": "question ".repeat(140)}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_old", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
            }),
            serde_json::json!({
                "role": "tool",
                "tool_call_id": "call_old",
                "content": "tool output ".repeat(220)
            }),
            serde_json::json!({"role": "user", "content": "middle question ".repeat(140)}),
            serde_json::json!({"role": "assistant", "content": "middle answer ".repeat(140)}),
            serde_json::json!({"role": "user", "content": "recent question ".repeat(140)}),
            serde_json::json!({"role": "assistant", "content": "recent answer ".repeat(140)}),
        ];

        let (windowed, transforms) = apply_token_budget_window(messages, tokenizer.as_ref(), 4000);

        assert_eq!(transforms.len(), 1);
        assert!(transforms[0].starts_with("window_cap:"));
        assert!(!windowed.iter().any(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some("call_old")
        }));
        assert!(!windowed.iter().any(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(|calls| {
                    calls
                        .iter()
                        .any(|call| call.get("id").and_then(Value::as_str) == Some("call_old"))
                })
                .unwrap_or(false)
        }));
    }

    #[test]
    fn aggregate_request_log_stats_builds_dashboard_totals() {
        let entries = vec![
            serde_json::json!({
                "timestamp": "100",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 100,
                "input_tokens_optimized": 60,
                "output_tokens": 25,
                "tokens_saved": 40,
                "optimization_latency_ms": 12.5,
                "ttfb_ms": 180.0,
                "tags": {"route_mode": "direct"},
            }),
            serde_json::json!({
                "timestamp": "200",
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "input_tokens_original": 80,
                "input_tokens_optimized": 80,
                "output_tokens": 10,
                "tokens_saved": 0,
                "ttfb_ms": 250.0,
                "tags": {"route": "compress"},
            }),
            serde_json::json!({
                "timestamp": "300",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 50,
                "input_tokens_optimized": 20,
                "output_tokens": 5,
                "tokens_saved": 30,
                "optimization_latency_ms": 7.5,
                "ttfb_ms": 120.0,
                "tags": {"route_mode": "shadow"},
            }),
        ];

        let stats = aggregate_request_log_stats(&entries, None);

        assert_eq!(stats.total_input_tokens_before_compression, 230);
        assert_eq!(stats.total_input_tokens, 160);
        assert_eq!(stats.total_output_tokens, 40);
        assert_eq!(stats.total_tokens_saved, 70);
        assert_eq!(stats.savings_percent, 30.43);
        assert_eq!(stats.overhead_average_ms, 10.0);
        assert_eq!(stats.overhead_min_ms, 7.5);
        assert_eq!(stats.overhead_max_ms, 12.5);
        assert_eq!(stats.ttfb_average_ms, 183.33);
        assert_eq!(stats.ttfb_min_ms, 120.0);
        assert_eq!(stats.ttfb_max_ms, 250.0);
        assert_eq!(stats.total_compression_savings_usd, 0.000011);
        assert_eq!(stats.total_input_cost_usd, 0.000252);
        assert_eq!(stats.by_provider["openai"], serde_json::json!(2));
        assert_eq!(stats.by_provider["anthropic"], serde_json::json!(1));
        assert_eq!(stats.by_model["gpt-4o-mini"], serde_json::json!(2));
        assert_eq!(stats.by_stack["direct"], serde_json::json!(1));
        assert_eq!(stats.by_stack["shadow"], serde_json::json!(1));
        assert_eq!(stats.by_stack["compress"], serde_json::json!(1));
        assert_eq!(
            stats.cost_per_model["gpt-4o-mini"],
            serde_json::json!({
                "requests": 2,
                "tokens_saved": 70,
                "tokens_sent": 80,
                "savings_usd": 0.000011,
                "reduction_pct": 46.67,
            })
        );
        assert_eq!(
            stats.cost_per_model["claude-3-5-sonnet"],
            serde_json::json!({
                "requests": 1,
                "tokens_saved": 0,
                "tokens_sent": 80,
                "savings_usd": 0.0,
                "reduction_pct": 0.0,
            })
        );
        assert_eq!(stats.savings_history.len(), 2);
        assert_eq!(
            stats.savings_history[1],
            serde_json::json!(["1970-01-01T00:05:00Z", 70])
        );
        assert_eq!(stats.display_session["requests"], serde_json::json!(3));
        assert_eq!(
            stats.display_session["started_at"],
            serde_json::json!("1970-01-01T00:01:40Z")
        );
        assert_eq!(
            stats.display_session["last_activity_at"],
            serde_json::json!("1970-01-01T00:05:00Z")
        );
        assert_eq!(
            stats.display_session["total_input_tokens"],
            serde_json::json!(160)
        );
        assert_eq!(
            stats.display_session["compression_savings_usd"],
            serde_json::json!(0.000011)
        );
        assert_eq!(
            stats.display_session["total_input_cost_usd"],
            serde_json::json!(0.000252)
        );
        assert_eq!(
            stats.persistent_savings["lifetime"]["tokens_saved"],
            serde_json::json!(70)
        );
        assert_eq!(
            stats.persistent_savings["lifetime"]["total_input_tokens"],
            serde_json::json!(160)
        );
        assert_eq!(
            stats.persistent_savings["lifetime"]["compression_savings_usd"],
            serde_json::json!(0.000011)
        );
        assert_eq!(
            stats.persistent_savings["lifetime"]["total_input_cost_usd"],
            serde_json::json!(0.000252)
        );
    }

    #[test]
    fn build_stats_history_payload_builds_rollups_from_request_logs() {
        let entries = vec![
            serde_json::json!({
                "timestamp": "1743037200",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 100,
                "input_tokens_optimized": 60,
                "output_tokens": 0,
                "tokens_saved": 40,
                "tags": {"route_mode": "direct"},
            }),
            serde_json::json!({
                "timestamp": "1743040800",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 50,
                "input_tokens_optimized": 40,
                "output_tokens": 0,
                "tokens_saved": 10,
                "tags": {"route_mode": "direct"},
            }),
            serde_json::json!({
                "timestamp": "1743127200",
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "input_tokens_original": 80,
                "input_tokens_optimized": 80,
                "output_tokens": 0,
                "tokens_saved": 0,
                "tags": {"route_mode": "shadow"},
            }),
            serde_json::json!({
                "timestamp": "1743213600",
                "provider": "headroom",
                "model": "gpt-4o-mini",
                "input_tokens_original": 40,
                "input_tokens_optimized": 15,
                "output_tokens": 0,
                "tokens_saved": 25,
                "tags": {"route": "compress"},
            }),
        ];

        let payload = build_stats_history_payload(&entries, "compact", None);

        assert_eq!(payload["schema_version"], serde_json::json!(2));
        assert_eq!(payload["lifetime"]["requests"], serde_json::json!(4));
        assert_eq!(payload["lifetime"]["tokens_saved"], serde_json::json!(75));
        assert_eq!(
            payload["lifetime"]["total_input_tokens"],
            serde_json::json!(195)
        );
        assert_eq!(
            payload["lifetime"]["compression_savings_usd"],
            serde_json::json!(0.000011)
        );
        assert_eq!(
            payload["lifetime"]["total_input_cost_usd"],
            serde_json::json!(0.000257)
        );
        assert_eq!(
            payload["display_session"]["savings_percent"],
            serde_json::json!(27.78)
        );
        assert_eq!(
            payload["history_summary"]["stored_points"],
            serde_json::json!(3)
        );
        assert_eq!(
            payload["history_summary"]["returned_points"],
            serde_json::json!(3)
        );
        assert_eq!(payload["series"]["daily"].as_array().unwrap().len(), 2);
        assert_eq!(
            payload["series"]["daily"][0]["total_input_tokens_delta"],
            serde_json::json!(100)
        );
        assert_eq!(
            payload["series"]["daily"][1]["total_tokens_saved"],
            serde_json::json!(75)
        );

        let csv = build_stats_history_csv(&entries, "daily", None);
        assert!(csv.starts_with("timestamp,tokens_saved,compression_savings_usd_delta"));
        assert!(csv.lines().nth(1).unwrap_or_default().contains(",50,"));
    }

    #[test]
    fn build_stats_history_payload_honors_history_modes() {
        let entries = vec![
            serde_json::json!({
                "timestamp": "100",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 100,
                "input_tokens_optimized": 60,
                "output_tokens": 0,
                "tokens_saved": 40,
                "tags": {"route_mode": "direct"},
            }),
            serde_json::json!({
                "timestamp": "200",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens_original": 50,
                "input_tokens_optimized": 20,
                "output_tokens": 0,
                "tokens_saved": 30,
                "tags": {"route_mode": "direct"},
            }),
        ];

        let full = build_stats_history_payload(&entries, "full", None);
        let none = build_stats_history_payload(&entries, "none", None);

        assert_eq!(full["history"].as_array().unwrap().len(), 2);
        assert_eq!(none["history"], serde_json::json!([]));
        assert_eq!(
            none["history_summary"]["compacted"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn empty_prefix_cache_stats_matches_dashboard_shape() {
        let prefix_cache = empty_prefix_cache_stats();

        assert_eq!(prefix_cache["totals"]["requests"], serde_json::json!(0));
        assert_eq!(
            prefix_cache["totals"]["net_savings_usd"],
            serde_json::json!(0.0)
        );
        assert_eq!(
            prefix_cache["totals"]["observed_ttl_buckets"]["5m"]["tokens"],
            serde_json::json!(0)
        );
        assert_eq!(
            prefix_cache["totals"]["observed_ttl_mix"]["active_buckets"],
            serde_json::json!([])
        );
        assert_eq!(prefix_cache["by_provider"], serde_json::json!({}));
    }

    #[test]
    fn optimize_anthropic_batch_create_body_compresses_large_request_messages() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let requests = (0..3)
            .map(|index| {
                serde_json::json!({
                    "custom_id": format!("req-{index}"),
                    "params": {
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 128,
                        "messages": (0..4)
                            .map(|message_index| serde_json::json!({
                                "role": "user",
                                "content": format!("{long_text}\nmessage {index}-{message_index}"),
                            }))
                            .collect::<Vec<_>>(),
                    }
                })
            })
            .collect::<Vec<_>>();
        let body = Bytes::from(
            serde_json::json!({
                "requests": requests
            })
            .to_string(),
        );

        let optimization = optimize_anthropic_batch_create_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);
        assert_eq!(optimization.requests_count, 3);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        let optimized_content = optimized["requests"][0]["params"]["messages"][0]["content"]
            .as_str()
            .unwrap();
        assert_ne!(optimized_content, format!("{long_text}\nmessage 0-0"));
    }

    #[test]
    fn optimize_anthropic_messages_body_compresses_large_messages() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let body = Bytes::from(
            serde_json::json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "messages": [{
                    "role": "user",
                    "content": format!("{long_text}\nmessage 0"),
                }]
            })
            .to_string(),
        );

        let optimization = optimize_anthropic_messages_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "claude-3-5-sonnet-20241022",
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        let optimized_content = optimized["messages"][0]["content"].as_str().unwrap();
        assert_ne!(optimized_content, format!("{long_text}\nmessage 0"));
    }

    #[test]
    fn optimize_openai_batch_jsonl_content_compresses_large_chat_lines() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let original_message = format!("{long_text}\nmessage 0");
        let jsonl = format!(
            "{}\n{}",
            serde_json::json!({
                "custom_id": "req-0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{
                        "role": "user",
                        "content": original_message,
                    }],
                }
            }),
            serde_json::json!({
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{
                        "role": "user",
                        "content": "short",
                    }],
                }
            })
        );

        let optimization = optimize_openai_batch_jsonl_content(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &jsonl,
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_eq!(optimization.stats.total_requests, 2);

        let first_line: Value = serde_json::from_str(
            optimization
                .content
                .lines()
                .next()
                .expect("optimized first jsonl line"),
        )
        .unwrap();
        let optimized_content = first_line["body"]["messages"][0]["content"]
            .as_str()
            .unwrap();
        assert_ne!(optimized_content, original_message);
    }

    #[test]
    fn validate_request_content_length_rejects_oversized_body() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_LENGTH,
            HeaderValue::from_static("128"),
        );
        let error = validate_request_content_length(&headers, 64).unwrap_err();
        assert!(matches!(
            error,
            ProxyError::RequestTooLarge {
                actual_bytes: 128,
                max_bytes: 64
            }
        ));
    }

    #[test]
    fn validate_request_array_length_rejects_oversized_array() {
        let error =
            validate_request_array_length("messages", MAX_REQUEST_ARRAY_LENGTH + 1).unwrap_err();
        assert!(matches!(
            error,
            ProxyError::RequestArrayTooLarge {
                field: "messages",
                actual_length,
                max_length: MAX_REQUEST_ARRAY_LENGTH
            } if actual_length == MAX_REQUEST_ARRAY_LENGTH + 1
        ));
    }

    #[test]
    fn is_loopback_addr_accepts_ipv4_ipv6_and_ipv6_mapped_loopback() {
        assert!(is_loopback_addr("127.0.0.1:8080".parse().unwrap()));
        assert!(is_loopback_addr("[::1]:8080".parse().unwrap()));
        assert!(is_loopback_addr("[::ffff:127.0.0.1]:8080".parse().unwrap()));
        assert!(!is_loopback_addr("203.0.113.10:8080".parse().unwrap()));
    }

    #[test]
    fn parse_gemini_generate_content_request_extracts_contract_fields() {
        let body = Bytes::from_static(
            br#"{"contents":[{"parts":[{"text":"hi"}]}],"systemInstruction":{"parts":[{"text":"be terse"}]}}"#,
        );
        let (contents_count, system_instruction_present) =
            parse_gemini_generate_content_request(&body).unwrap();
        assert_eq!(contents_count, 1);
        assert!(system_instruction_present);
    }

    #[test]
    fn parse_gemini_generate_content_request_allows_missing_contents() {
        let body = Bytes::from_static(br#"{"systemInstruction":{"parts":[{"text":"be terse"}]}}"#);
        let (contents_count, system_instruction_present) =
            parse_gemini_generate_content_request(&body).unwrap();
        assert_eq!(contents_count, 0);
        assert!(system_instruction_present);
    }

    #[test]
    fn parse_gemini_generate_content_request_rejects_non_array_contents() {
        let body = Bytes::from_static(br#"{"contents":"hello"}"#);
        let error = parse_gemini_generate_content_request(&body).unwrap_err();
        assert!(matches!(
            error,
            ProxyError::InvalidRequest(message) if message == "field `contents` must be an array"
        ));
    }

    #[test]
    fn parse_gemini_generate_content_request_rejects_invalid_system_instruction_shape() {
        let body = Bytes::from_static(
            br#"{"contents":[{"parts":[{"text":"hi"}]}],"systemInstruction":"be terse"}"#,
        );
        let error = parse_gemini_generate_content_request(&body).unwrap_err();
        assert!(matches!(
            error,
            ProxyError::InvalidRequest(message) if message == "field `systemInstruction` must be an object"
        ));
    }

    #[test]
    fn optimize_gemini_count_tokens_body_compresses_large_text_contents() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let contents = (0..6)
            .map(|index| {
                serde_json::json!({
                    "role": "user",
                    "parts": [{"text": format!("{long_text}\nmessage {index}")}],
                })
            })
            .collect::<Vec<_>>();
        let body = Bytes::from(
            serde_json::json!({
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": "be terse"}],
                }
            })
            .to_string(),
        );

        let optimization = optimize_gemini_count_tokens_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "gemini-2.0-flash",
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        let optimized_text = optimized["contents"][0]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert_ne!(optimized_text, format!("{long_text}\nmessage 0"));
        assert_eq!(
            optimized["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("be terse")
        );
    }

    #[test]
    fn optimize_gemini_generate_content_body_compresses_large_text_contents() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let body = Bytes::from(
            serde_json::json!({
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": format!("{long_text}\nmessage 0")}],
                    },
                    {
                        "role": "model",
                        "parts": [{"text": "ack"}],
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": "be terse"}]
                }
            })
            .to_string(),
        );

        let optimization = optimize_gemini_generate_content_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "gemini-2.0-flash",
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        assert_ne!(
            optimized["contents"][0]["parts"][0]["text"],
            serde_json::json!(format!("{long_text}\nmessage 0"))
        );
        assert_eq!(
            optimized["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("be terse")
        );
    }

    #[test]
    fn optimize_google_cloudcode_stream_body_compresses_large_text_contents() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let body = Bytes::from(
            serde_json::json!({
                "project": "demo",
                "model": "gemini-3.1-pro-high",
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": format!("{long_text}\nmessage 0")}],
                    }],
                    "systemInstruction": {
                        "parts": [{"text": "be terse"}]
                    }
                }
            })
            .to_string(),
        );

        let optimization = optimize_google_cloudcode_stream_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "gemini-3.1-pro-high",
            false,
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        assert_ne!(optimization.primary_body_bytes, body);

        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        assert_ne!(
            optimized["request"]["contents"][0]["parts"][0]["text"],
            serde_json::json!(format!("{long_text}\nmessage 0"))
        );
        assert_eq!(
            optimized["request"]["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("be terse")
        );
    }

    #[test]
    fn optimize_google_cloudcode_stream_body_preserves_antigravity_system_instruction() {
        let long_text = (0..40)
            .map(|index| {
                format!(
                    "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
                )
            })
            .collect::<String>();
        let body = Bytes::from(
            serde_json::json!({
                "project": "demo",
                "model": "gemini-3.1-pro-high",
                "requestType": "agent",
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": format!("{long_text}\nmessage 0")}],
                    }],
                    "systemInstruction": {
                        "parts": [{"text": "keep me"}]
                    }
                }
            })
            .to_string(),
        );

        let optimization = optimize_google_cloudcode_stream_body(
            &Arc::new(Mutex::new(CompressionCache::default())),
            &ProductStore::default(),
            &TelemetryStore::default(),
            &body,
            "gemini-3.1-pro-high",
            true,
        )
        .unwrap();

        assert_eq!(optimization.compression_status, "compressed");
        let optimized: Value = serde_json::from_slice(&optimization.primary_body_bytes).unwrap();
        assert_eq!(
            optimized["request"]["systemInstruction"]["parts"][0]["text"],
            serde_json::json!("keep me")
        );
    }

    #[test]
    fn parse_google_cloudcode_stream_request_extracts_contract_fields() {
        let headers = HeaderMap::new();
        let body = Bytes::from_static(
            br#"{"project":"demo","model":"gemini-3.1-pro-high","request":{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}}"#,
        );
        let (model, contents_count, antigravity) =
            parse_google_cloudcode_stream_request(&headers, &body).unwrap();
        assert_eq!(model, "gemini-3.1-pro-high");
        assert_eq!(contents_count, 1);
        assert!(!antigravity);
    }

    #[test]
    fn parse_google_cloudcode_stream_request_detects_antigravity() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::USER_AGENT,
            HeaderValue::from_static("Antigravity/1.2.3"),
        );
        let body = Bytes::from_static(
            br#"{"project":"demo","model":"claude-sonnet-4-6","requestType":"agent","userAgent":"antigravity","request":{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}}"#,
        );
        let (model, contents_count, antigravity) =
            parse_google_cloudcode_stream_request(&headers, &body).unwrap();
        assert_eq!(model, "claude-sonnet-4-6");
        assert_eq!(contents_count, 1);
        assert!(antigravity);
    }

    #[test]
    fn resolve_cloudcode_base_url_uses_antigravity_target() {
        let default_base: url::Url = "https://cloudcode-pa.googleapis.com/".parse().unwrap();
        let resolved = resolve_cloudcode_base_url(true, &default_base);
        assert_eq!(
            resolved.as_str(),
            "https://daily-cloudcode-pa.sandbox.googleapis.com/"
        );
    }

    #[test]
    fn parse_anthropic_messages_request_extracts_contract_fields() {
        let body = Bytes::from_static(
            br#"{"model":"claude-haiku-4-5","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        );
        let (model, messages_count, max_tokens, streaming) =
            parse_anthropic_messages_request(&body).unwrap();
        assert_eq!(model, "claude-haiku-4-5");
        assert_eq!(messages_count, 1);
        assert_eq!(max_tokens, 16);
        assert!(!streaming);
    }

    #[test]
    fn parse_anthropic_messages_request_allows_missing_passthrough_fields() {
        let body = Bytes::from_static(br#"{}"#);
        let (model, messages_count, max_tokens, streaming) =
            parse_anthropic_messages_request(&body).unwrap();
        assert_eq!(model, "unknown");
        assert_eq!(messages_count, 0);
        assert_eq!(max_tokens, 0);
        assert!(!streaming);
    }

    #[test]
    fn parse_anthropic_messages_request_rejects_invalid_field_types() {
        let body = Bytes::from_static(br#"{"model":123,"max_tokens":"16","messages":"hello"}"#);
        let error = parse_anthropic_messages_request(&body).unwrap_err();
        assert!(matches!(
            error,
            ProxyError::InvalidRequest(message) if message == "field `model` must be a string"
        ));
    }

    #[test]
    fn parse_anthropic_count_tokens_request_extracts_contract_fields() {
        let body = Bytes::from_static(
            br#"{"model":"claude-haiku-4-5","system":"be terse","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let (model, messages_count, system_present) =
            parse_anthropic_count_tokens_request(&body).unwrap();
        assert_eq!(model, "claude-haiku-4-5");
        assert_eq!(messages_count, 1);
        assert!(system_present);
    }

    #[test]
    fn parse_anthropic_count_tokens_request_allows_passthrough_system_shape() {
        let body = Bytes::from_static(
            br#"{"model":"claude-haiku-4-5","system":{"text":"be terse"},"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let (model, messages_count, system_present) =
            parse_anthropic_count_tokens_request(&body).unwrap();
        assert_eq!(model, "claude-haiku-4-5");
        assert_eq!(messages_count, 1);
        assert!(system_present);
    }

    #[test]
    fn gemini_model_from_path_extracts_model_name() {
        let model = gemini_model_from_path("/v1beta/models/gemini-2.0-flash:generateContent");
        assert_eq!(model.as_deref(), Some("gemini-2.0-flash"));
        let count_tokens = gemini_model_from_suffix(
            "/v1beta/models/gemini-2.0-flash:countTokens",
            ":countTokens",
        );
        assert_eq!(count_tokens.as_deref(), Some("gemini-2.0-flash"));
        assert!(is_gemini_streaming_query(Some("key=test&alt=sse")));
        assert!(!is_gemini_streaming_query(Some("key=test")));
    }

    #[test]
    fn detect_text_content_kind_identifies_search_results() {
        let content =
            "src/lib.rs:10:fn main() {\nsrc/lib.rs:11:    println!(\"hi\");\nsrc/lib.rs:12:}";
        assert_eq!(
            detect_text_content_kind(content),
            CompressContentKind::SearchResults
        );
    }

    #[test]
    fn detect_text_content_kind_identifies_build_output() {
        let content = "[12:00:00] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example\ncargo error: linker failed";
        assert_eq!(
            detect_text_content_kind(content),
            CompressContentKind::BuildOutput
        );
    }

    #[test]
    fn detect_text_content_kind_identifies_html() {
        let content = "<!DOCTYPE html><html><head><title>x</title></head><body><main><div>Hello</div></main></body></html>";
        assert_eq!(detect_text_content_kind(content), CompressContentKind::Html);
    }

    #[test]
    fn extract_html_text_removes_boilerplate_and_scripts() {
        let content = r#"<!DOCTYPE html>
<html>
  <head>
    <title>demo</title>
    <style>.hidden { display:none; }</style>
    <script>console.log("ignore me");</script>
  </head>
  <body>
    <nav>Top navigation</nav>
    <main>
      <article>
        <h1>Deployment readiness update</h1>
        <p>The rollout completed successfully across three regions.</p>
      </article>
    </main>
    <footer>copyright 2025</footer>
  </body>
</html>"#;

        let extracted = extract_html_text(content);
        assert!(extracted.contains("demo"));
        assert!(extracted.contains("Deployment readiness update"));
        assert!(extracted.contains("rollout completed successfully"));
        assert!(!extracted.contains("console.log"));
        assert!(!extracted.contains("Top navigation"));
        assert!(!extracted.contains("copyright 2025"));
    }

    #[test]
    fn detect_text_content_kind_keeps_json_with_warning_fields_as_structured() {
        let content =
            r#"[{"status":"warn","message":"request completed","payload":{"region":"us-east-1"}}]"#;
        assert_eq!(
            detect_text_content_kind(content),
            CompressContentKind::StructuredJson
        );
    }

    #[test]
    fn detect_text_content_kind_identifies_plain_text() {
        let content = "Paragraph one\n\nThis is generic prose about deployment readiness.\nIt does not look like code or logs.";
        assert_eq!(
            detect_text_content_kind(content),
            CompressContentKind::PlainText
        );
    }

    #[test]
    fn detect_text_content_exposes_json_metadata_and_confidence() {
        let content = r#"[{"id":1},{"id":2}]"#;
        let detected = detect_text_content(content);
        assert_eq!(detected.kind, CompressContentKind::StructuredJson);
        assert_eq!(detected.confidence, 1.0);
        assert_eq!(detected.metadata.get("item_count"), Some(&Value::from(2)));
        assert_eq!(
            detected.metadata.get("is_dict_array"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn detect_text_content_does_not_treat_json_object_as_structured_json() {
        let content = r#"{"id":1,"message":"plain payload"}"#;
        let detected = detect_text_content(content);
        assert_ne!(detected.kind, CompressContentKind::StructuredJson);
    }

    #[test]
    fn detect_text_content_prefers_build_output_when_json_is_embedded_in_logs() {
        let content = "ERROR invalid config\n{\"key\":\"value\"}";
        let detected = detect_text_content(content);
        assert_eq!(detected.kind, CompressContentKind::BuildOutput);
        assert!(detected.confidence >= 0.5);
    }

    #[test]
    fn detect_text_content_counts_warnings_as_error_matches_for_logs() {
        let content = [
            "WARN auth token rotation lagging behind schedule",
            "INFO continuing rollout",
            "ERROR deployment failed due to timeout",
        ]
        .join("\n");
        let detected = detect_text_content(&content);
        assert_eq!(detected.kind, CompressContentKind::BuildOutput);
        assert_eq!(
            detected.metadata.get("error_matches"),
            Some(&Value::from(2))
        );
        assert_eq!(detected.metadata.get("warning_matches"), None);
        assert_eq!(detected.metadata.get("security_matches"), None);
    }

    #[test]
    fn detect_text_content_uses_search_thresholds() {
        let content = [
            "src/lib.rs:10:fn main() {",
            "plain text",
            "plain text",
            "plain text",
            "plain text",
        ]
        .join("\n");
        let detected = detect_text_content(&content);
        assert_eq!(detected.kind, CompressContentKind::PlainText);
    }

    #[test]
    fn detect_text_content_extracts_code_language_metadata() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n\nstruct Config {\n    enabled: bool,\n}\n\nimpl Config {\n    fn new() -> Self {\n        Self { enabled: true }\n    }\n}";
        let detected = detect_text_content(content);
        assert_eq!(detected.kind, CompressContentKind::SourceCode);
        assert_eq!(
            detected.metadata.get("language"),
            Some(&Value::String("rust".to_string()))
        );
        assert!(detected.confidence >= 0.5);
    }

    #[test]
    fn split_mixed_content_sections_preserves_code_fences_and_search_runs() {
        let content = "Narrative intro\n\n```rust\nfn main() {}\n```\n\nsrc/lib.rs:10:error path\nsrc/lib.rs:11:helper path";
        let sections = split_mixed_content_sections(content);
        assert_eq!(sections.len(), 3);
        assert!(matches!(
            &sections[1],
            MixedContentSection::CodeFence { language, .. } if language == "rust"
        ));
        assert!(
            matches!(&sections[2], MixedContentSection::Raw(raw) if raw.contains("src/lib.rs:10"))
        );
    }

    #[test]
    fn split_mixed_content_sections_detects_unfenced_code_blocks() {
        let content = "Narrative intro about rollout analysis and more context.\n\nfn helper(value: i32) -> i32 {\n    value + 1\n}\nstruct Config {\n    enabled: bool,\n}\nimpl Config {\n    fn new() -> Self {\n        Self { enabled: true }\n    }\n}";
        let sections = split_mixed_content_sections(content);
        assert_eq!(sections.len(), 2);
        assert!(
            matches!(&sections[1], MixedContentSection::CodeBlock(raw) if raw.contains("fn helper"))
        );
    }

    #[test]
    fn is_mixed_content_detects_prose_plus_search_results() {
        let content = "Incident summary for rollout validation. This section explains the operational context. Another sentence adds more narrative detail. More prose keeps the detector honest. Yet another sentence for the prose heuristic. Final sentence before matches.\n\nsrc/lib.rs:10:error path\nsrc/lib.rs:11:helper path";
        assert!(is_mixed_content(content));
    }

    #[test]
    fn is_mixed_content_detects_prose_plus_unfenced_code() {
        let content = "Incident summary for rollout validation. This section explains the operational context. Another sentence adds more narrative detail. More prose keeps the detector honest. Yet another sentence for the prose heuristic. Final sentence before code.\n\nfn helper(value: i32) -> i32 {\n    value + 1\n}\nstruct Config {\n    enabled: bool,\n}\nimpl Config {\n    fn new() -> Self {\n        Self { enabled: true }\n    }\n}";
        assert!(is_mixed_content(content));
    }

    #[test]
    fn compression_bias_tracks_target_ratio() {
        let default_options = |target_ratio| CompressOptions {
            compress_user_messages: false,
            compress_system_messages: true,
            protect_recent: 4,
            min_tokens_to_compress: 50,
            target_ratio,
            protect_analysis_context: true,
            token_budget: None,
        };
        let low = compression_bias(&CompressOptions {
            target_ratio: Some(0.15),
            ..default_options(None)
        });
        let high = compression_bias(&CompressOptions {
            target_ratio: Some(0.70),
            ..default_options(None)
        });
        assert!(low < 1.0);
        assert!(high > 1.0);
    }

    #[test]
    fn build_text_compressor_uses_target_ratio_budget() {
        let default_options = CompressOptions {
            compress_user_messages: false,
            compress_system_messages: true,
            protect_recent: 4,
            min_tokens_to_compress: 50,
            target_ratio: Some(0.2),
            protect_analysis_context: true,
            token_budget: None,
        };
        let text = (1..=100)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let compressor = build_text_compressor(&text, &default_options);
        let result = compressor.compress(&text, "");
        assert!(result.compressed_line_count <= 30);
    }
}
