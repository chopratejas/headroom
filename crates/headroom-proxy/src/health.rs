//! Health endpoints. These are intercepted by Rust and never forwarded.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::proxy::AppState;

const SERVICE_NAME: &str = "headroom-proxy";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct UpstreamProbe {
    ready: bool,
    status_code: Option<u16>,
    endpoint: &'static str,
    error: Option<String>,
}

/// Process liveness: 200 if the proxy process is up.
pub async fn livez(State(state): State<AppState>) -> impl IntoResponse {
    Json(livez_payload(&state))
}

/// Aggregate health view. Mirrors the Python contract shape closely enough for
/// readiness probes, operators, and future enterprise extensions.
pub async fn health(State(state): State<AppState>) -> Response {
    let payload = health_payload(&state, true).await;
    (StatusCode::OK, Json(payload)).into_response()
}

/// Readiness view. Returns 503 if required components are not ready.
pub async fn readyz(State(state): State<AppState>) -> Response {
    let payload = health_payload(&state, false).await;
    let status = if payload["ready"].as_bool().unwrap_or(false) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(payload)).into_response()
}

/// Backward-compatible local health endpoint.
pub async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": SERVICE_NAME,
        "status": "healthy",
        "alive": true,
        "version": VERSION,
        "uptime_seconds": uptime_seconds(&state),
    }))
}

/// Backward-compatible upstream health endpoint.
pub async fn healthz_upstream(State(state): State<AppState>) -> Response {
    let probe = probe_upstream(&state).await;
    let status = if probe.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "ok": probe.ready,
            "endpoint": probe.endpoint,
            "upstream_status": probe.status_code,
            "error": probe.error,
        })),
    )
        .into_response()
}

fn livez_payload(state: &AppState) -> Value {
    json!({
        "service": SERVICE_NAME,
        "status": "healthy",
        "alive": true,
        "version": VERSION,
        "timestamp_unix_seconds": unix_timestamp_seconds(),
        "uptime_seconds": uptime_seconds(state),
    })
}

async fn health_payload(state: &AppState, include_config: bool) -> Value {
    let upstream = probe_upstream(state).await;
    let startup_ready = true;
    let http_client_ready = true;
    let ready = startup_ready && http_client_ready && upstream.ready;
    let route_manifest = route_manifest();
    let local_state_backends = state.local_state_backends();

    let mut payload = json!({
    "service": SERVICE_NAME,
    "status": if ready { "healthy" } else { "unhealthy" },
    "ready": ready,
    "version": VERSION,
    "timestamp_unix_seconds": unix_timestamp_seconds(),
    "uptime_seconds": uptime_seconds(state),
    "checks": {
        "startup": component_health(true, startup_ready, json!({})),
        "http_client": component_health(true, http_client_ready, json!({})),
        "upstream": component_health(
            true,
            upstream.ready,
            json!({
                "endpoint": upstream.endpoint,
                "upstream_status": upstream.status_code,
                "error": upstream.error,
            }),
        ),
        "cache": component_health(false, true, json!({})),
        "memory": component_health(false, true, json!({})),
    },
        "runtime": {
            "request_pipeline": {
                "plugins_enabled": state.runtime.is_enabled(),
                "plugin_count": state.runtime.plugin_count(),
            },
            "websocket_sessions": {
                "active_sessions": state.ws_sessions.active_count(),
                "active_relay_tasks": state.ws_sessions.active_relay_task_count(),
            },
            "route_manifest": route_manifest.clone(),
            "local_state_backends": local_state_backends,
            "upstream": {
                "base_url": state.config.upstream.as_str(),
                "openai_api_url": state.config.openai_api_url.as_str(),
                "anthropic_api_url": state.config.anthropic_api_url.as_str(),
                "gemini_api_url": state.config.gemini_api_url.as_str(),
                "databricks_api_url": state.config.databricks_api_url.as_str(),
                "cloudcode_api_url": state.config.cloudcode_api_url.as_str(),
                "response_cache_enabled": state.config.response_cache_enabled,
                "response_cache_max_entries": state.config.response_cache_max_entries,
                "response_cache_ttl_seconds": state.config.response_cache_ttl.as_secs(),
                "rewrite_host": state.config.rewrite_host,
                "native_openai_chat": state.config.native_openai_chat,
                "native_openai_chat_shadow": state.config.native_openai_chat_shadow,
                "native_anthropic_messages": state.config.native_anthropic_messages,
                "native_anthropic_messages_shadow": state.config.native_anthropic_messages_shadow,
                "native_anthropic_count_tokens": state.config.native_anthropic_count_tokens,
                "native_gemini_generate_content": state.config.native_gemini_generate_content,
                "native_gemini_generate_content_shadow": state.config.native_gemini_generate_content_shadow,
                "native_gemini_count_tokens": state.config.native_gemini_count_tokens,
                "native_gemini_count_tokens_shadow": state.config.native_gemini_count_tokens_shadow,
                "native_google_cloudcode_stream": state.config.native_google_cloudcode_stream,
                "native_google_cloudcode_stream_shadow": state.config.native_google_cloudcode_stream_shadow,
            },
        },
    });

    if let Some(deployment) = deployment_payload() {
        payload["deployment"] = deployment;
    }

    if include_config {
        payload["config"] = json!({
            "listen": state.config.listen.to_string(),
            "upstream": state.config.upstream.as_str(),
            "openai_api_url": state.config.openai_api_url.as_str(),
            "anthropic_api_url": state.config.anthropic_api_url.as_str(),
            "gemini_api_url": state.config.gemini_api_url.as_str(),
            "databricks_api_url": state.config.databricks_api_url.as_str(),
            "cloudcode_api_url": state.config.cloudcode_api_url.as_str(),
            "response_cache_enabled": state.config.response_cache_enabled,
            "response_cache_max_entries": state.config.response_cache_max_entries,
            "response_cache_ttl_seconds": state.config.response_cache_ttl.as_secs(),
            "upstream_timeout_seconds": state.config.upstream_timeout.as_secs(),
            "upstream_connect_timeout_seconds": state.config.upstream_connect_timeout.as_secs(),
            "max_body_bytes": state.config.max_body_bytes,
            "rewrite_host": state.config.rewrite_host,
            "native_openai_chat": state.config.native_openai_chat,
            "native_openai_chat_shadow": state.config.native_openai_chat_shadow,
            "native_anthropic_messages": state.config.native_anthropic_messages,
            "native_anthropic_messages_shadow": state.config.native_anthropic_messages_shadow,
            "native_anthropic_count_tokens": state.config.native_anthropic_count_tokens,
            "native_gemini_generate_content": state.config.native_gemini_generate_content,
            "native_gemini_generate_content_shadow": state.config.native_gemini_generate_content_shadow,
            "native_gemini_count_tokens": state.config.native_gemini_count_tokens,
            "native_gemini_count_tokens_shadow": state.config.native_gemini_count_tokens_shadow,
            "native_google_cloudcode_stream": state.config.native_google_cloudcode_stream,
            "native_google_cloudcode_stream_shadow": state.config.native_google_cloudcode_stream_shadow,
            "graceful_shutdown_timeout_seconds": state.config.graceful_shutdown_timeout.as_secs(),
            "route_manifest": route_manifest,
        });
    }

    payload
}

fn route_manifest() -> Value {
    json!({
        "version": 1,
        "families": [
            {
                "name": "health",
                "mode": "native",
                "routes": ["/livez", "/readyz", "/health", "/healthz", "/healthz/upstream", "/metrics"]
            },
            {
                "name": "admin-dashboard",
                "mode": "native-local",
                "routes": ["/dashboard"]
            },
            {
                "name": "debug-introspection",
                "mode": "native-local",
                "routes": ["/debug/memory", "/debug/tasks", "/debug/ws-sessions", "/debug/warmup"]
            },
            {
                "name": "openai-chat",
                "mode": "feature-flagged-native",
                "routes": ["/v1/chat/completions"]
            },
            {
                "name": "openai-responses",
                "mode": "native-direct",
                "routes": ["/v1/responses", "/v1/codex/responses", "/backend-api/responses", "/backend-api/codex/responses"]
            },
            {
                "name": "openai-batches",
                "mode": "native-direct",
                "routes": ["/v1/batches", "/v1/batches/{batch_id}", "/v1/batches/{batch_id}/cancel"]
            },
            {
                "name": "openai-model-metadata",
                "mode": "native-direct",
                "routes": ["/v1/models", "/v1/models/{model_id}"]
            },
            {
                "name": "openai-utilities",
                "mode": "native-direct",
                "routes": ["/v1/embeddings", "/v1/moderations", "/v1/images/generations", "/v1/audio/transcriptions", "/v1/audio/speech"]
            },
            {
                "name": "anthropic-messages",
                "mode": "feature-flagged-native",
                "routes": ["/v1/messages"]
            },
            {
                "name": "anthropic-count-tokens",
                "mode": "feature-flagged-native",
                "routes": ["/v1/messages/count_tokens"]
            },
            {
                "name": "anthropic-batches",
                "mode": "native-direct",
                "routes": ["/v1/messages/batches", "/v1/messages/batches/{batch_id}", "/v1/messages/batches/{batch_id}/results", "/v1/messages/batches/{batch_id}/cancel"]
            },
            {
                "name": "databricks-invocations",
                "mode": "native-direct",
                "routes": ["/serving-endpoints/{model}/invocations"]
            },
            {
                "name": "gemini-native",
                "mode": "feature-flagged-native",
                "routes": ["/v1beta/models/{model}:generateContent", "/v1beta/models/{model}:streamGenerateContent", "/v1beta/models/{model}:countTokens"]
            },
            {
                "name": "gemini-passthrough",
                "mode": "native-direct",
                "routes": ["/v1beta/models", "/v1beta/models/{model_name}", "/v1beta/models/{model}:embedContent", "/v1beta/models/{model}:batchEmbedContents", "/v1beta/models/{model}:batchGenerateContent", "/v1beta/batches/{batch_name}", "/v1beta/batches/{batch_name}:cancel", "/v1beta/cachedContents", "/v1beta/cachedContents/{cache_id}"]
            },
            {
                "name": "google-cloudcode",
                "mode": "feature-flagged-native",
                "routes": ["/v1internal:streamGenerateContent", "/v1/v1internal:streamGenerateContent"]
            },
            {
                "name": "telemetry",
                "mode": "native-local",
                "routes": ["/v1/telemetry", "/v1/telemetry/export", "/v1/telemetry/import", "/v1/telemetry/tools", "/v1/telemetry/tools/{signature_hash}"]
            },
            {
                "name": "admin-local",
                "mode": "native-local",
                "routes": ["/stats", "/stats-history", "/debug/memory", "/transformations/feed", "/subscription-window", "/quota", "/cache/clear"]
            },
            {
                "name": "ccr-read",
                "mode": "native-local",
                "routes": ["/v1/retrieve", "/v1/retrieve/{hash_key}", "/v1/retrieve/stats", "/v1/retrieve/tool_call", "/v1/feedback", "/v1/feedback/{tool_name}"]
            },
            {
                "name": "toin",
                "mode": "native-local",
                "routes": ["/v1/toin/stats", "/v1/toin/patterns", "/v1/toin/pattern/{hash_prefix}"]
            },
            {
                "name": "compression",
                "mode": "native-local",
                "routes": ["/v1/compress"]
            }
        ]
    })
}

fn component_health(enabled: bool, ready: bool, details: Value) -> Value {
    let mut payload = json!({
        "enabled": enabled,
        "ready": if enabled { ready } else { true },
        "status": if !enabled {
            "disabled"
        } else if ready {
            "healthy"
        } else {
            "unhealthy"
        },
    });

    if let (Some(dst), Some(src)) = (payload.as_object_mut(), details.as_object()) {
        for (key, value) in src {
            dst.insert(key.clone(), value.clone());
        }
    }

    payload
}

async fn probe_upstream(state: &AppState) -> UpstreamProbe {
    for endpoint in ["/readyz", "/healthz"] {
        let response = state.client.get(probe_url(state, endpoint)).send().await;
        match response {
            Ok(resp) if resp.status().is_success() => {
                return UpstreamProbe {
                    ready: true,
                    status_code: Some(resp.status().as_u16()),
                    endpoint,
                    error: None,
                }
            }
            Ok(resp) if resp.status() == StatusCode::NOT_FOUND => continue,
            Ok(resp) => {
                return UpstreamProbe {
                    ready: false,
                    status_code: Some(resp.status().as_u16()),
                    endpoint,
                    error: None,
                }
            }
            Err(error) => {
                return UpstreamProbe {
                    ready: false,
                    status_code: None,
                    endpoint,
                    error: Some(error.to_string()),
                }
            }
        }
    }

    UpstreamProbe {
        ready: false,
        status_code: None,
        endpoint: "/readyz",
        error: Some("upstream readiness endpoint not found".to_string()),
    }
}

fn probe_url(state: &AppState, endpoint: &str) -> reqwest::Url {
    let mut url = state.config.upstream.clone();
    url.set_path(endpoint);
    url.set_query(None);
    url
}

fn deployment_payload() -> Option<Value> {
    let profile = std::env::var("HEADROOM_DEPLOYMENT_PROFILE").ok()?;
    Some(json!({
        "profile": profile,
        "preset": std::env::var("HEADROOM_DEPLOYMENT_PRESET").ok(),
        "runtime": std::env::var("HEADROOM_DEPLOYMENT_RUNTIME").ok(),
        "supervisor": std::env::var("HEADROOM_DEPLOYMENT_SUPERVISOR").ok(),
        "scope": std::env::var("HEADROOM_DEPLOYMENT_SCOPE").ok(),
    }))
}

fn unix_timestamp_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn uptime_seconds(state: &AppState) -> f64 {
    state.started_at.elapsed().as_secs_f64()
}
