//! Core reverse-proxy router and HTTP forwarding handler.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, State, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderName, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::{any, get};
use axum::Router;
#[cfg(test)]
use bytes::Bytes;
use futures_util::{StreamExt as _, TryStreamExt};
#[cfg(test)]
use http_body_util::BodyExt;

use headroom_core::context::IntelligentContextManager;

use crate::compression;
use crate::config::Config;
use crate::error::ProxyError;
use crate::headers::{build_forward_request_headers, filter_response_headers};
use crate::health::{healthz, healthz_upstream};
use crate::websocket::ws_handler;

/// Shared state passed to every handler.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: reqwest::Client,
    /// Optional shared `IntelligentContextManager`. Constructed only
    /// when `config.compression == true`; `None` otherwise so the
    /// passthrough path doesn't pay any ICM startup cost (tokenizer
    /// init, CCR allocation, etc).
    pub icm: Option<Arc<IntelligentContextManager>>,
}

impl AppState {
    pub fn new(config: Config) -> Result<Self, ProxyError> {
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

        // Construct ICM only when compression is enabled. ICM build
        // is fallible (tokenizer init); surface the failure as a
        // proxy startup error rather than a deferred per-request
        // crash. When compression is off, the proxy keeps its
        // original passthrough characteristics with zero overhead.
        let icm = if config.compression {
            Some(compression::build_icm().map_err(ProxyError::CompressionStartup)?)
        } else {
            None
        };

        Ok(Self {
            config: Arc::new(config),
            client,
            icm,
        })
    }
}

/// Build the axum app. `/healthz` and `/healthz/upstream` are intercepted;
/// everything else hits the catch-all forwarder. WebSocket upgrades are
/// handled inside the catch-all handler when an `Upgrade: websocket` header
/// is present.
pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/healthz/upstream", get(healthz_upstream))
        .fallback(any(catch_all))
        .with_state(state)
}

/// Catch-all handler. If the request is a WebSocket upgrade, hand off to the
/// ws module; otherwise forward as plain HTTP.
async fn catch_all(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    ws: Option<WebSocketUpgrade>,
    req: Request<Body>,
) -> Response<Body> {
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

/// True if `Content-Type` is `application/json` (with any optional
/// parameters like `; charset=utf-8`). Compression only inspects JSON
/// bodies — multipart uploads, form-encoded posts, and binary
/// payloads stream through untouched.
fn is_application_json(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            // Take the media-type portion before any ';'. Trim and
            // compare case-insensitively per RFC 7231 §3.1.1.1.
            let media_type = s.split(';').next().unwrap_or("").trim();
            media_type.eq_ignore_ascii_case("application/json")
        })
        .unwrap_or(false)
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
    let start = Instant::now();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();

    let upstream_url = build_upstream_url(&state.config.upstream, &uri)?;

    // Forwarded-Host: prefer client's Host. Forwarded-Proto: assume http for
    // now (we don't terminate TLS in this binary; if a TLS terminator is in
    // front, it should rewrite this — which we'd handle by not overwriting
    // an existing one in a future change).
    let forwarded_host = req
        .headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

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

    // ─── COMPRESSION GATE ──────────────────────────────────────────────
    //
    // Streaming-by-default is the proxy's contract for everything that
    // isn't explicitly an LLM-shape request. To inspect a body for
    // compression we have to buffer, which is incompatible with
    // streaming — so we make the buffering decision *here*, on a
    // narrow gate:
    //
    //   - Compression enabled in config?       (`state.config.compression`)
    //   - Method is POST?                      (we only compress request bodies)
    //   - Path matches a known LLM endpoint?   (`compression::is_compressible_path`)
    //   - Content-Type is application/json?    (skip multipart, form, binary)
    //   - ICM was successfully built?          (Some by construction when compression is on)
    //
    // ALL of those true → buffer + run ICM + forward modified body.
    // ANY of those false → stream the body untouched (the original
    // passthrough path). This keeps WebSocket upgrades, healthchecks,
    // tool-API endpoints, and SSE streaming from paying any
    // buffering cost.
    let should_compress = state.config.compression
        && method == axum::http::Method::POST
        && compression::is_compressible_path(uri.path())
        && is_application_json(req.headers())
        && state.icm.is_some();

    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    let upstream_resp = if should_compress {
        // Buffer up to `compression_max_body_bytes`. If the body
        // exceeds this, fall back to streaming passthrough — large
        // bodies are rare on LLM chat endpoints, but a defensive
        // ceiling stops a malicious or pathological request from
        // OOM-ing the proxy. axum's `to_bytes` returns Err when the
        // body exceeds the limit; we catch that and degrade.
        let max = state.config.compression_max_body_bytes as usize;
        let buffered = match to_bytes(req.into_body(), max).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    limit_bytes = max,
                    error = %e,
                    "compression: body exceeds limit, falling back to streaming passthrough \
                     is impossible (body already partially consumed) — failing the request",
                );
                // Once `req.into_body()` is consumed by `to_bytes` we
                // can no longer stream. The defensive choice is to
                // fail the request loudly. Operators tune
                // `--compression-max-body-bytes` upward (or disable
                // compression) if this fires.
                return Err(ProxyError::InvalidHeader(format!(
                    "request body exceeds compression buffer limit ({max} bytes): {e}"
                )));
            }
        };

        // Run the compressor. Failures degrade to passthrough by
        // returning the original buffered bytes.
        let icm = state.icm.as_ref().expect("checked above");
        let outcome = compression::maybe_compress(&buffered, icm);

        let body_to_send = match outcome {
            compression::Outcome::Compressed {
                body,
                tokens_before,
                tokens_after,
                strategies_applied,
                markers_inserted,
            } => {
                tracing::info!(
                    request_id = %request_id,
                    path = %path_for_log,
                    tokens_before = tokens_before,
                    tokens_after = tokens_after,
                    tokens_freed = tokens_before.saturating_sub(tokens_after),
                    strategies = ?strategies_applied,
                    markers = markers_inserted.len(),
                    "compression applied"
                );
                body
            }
            compression::Outcome::NoCompression { tokens_before } => {
                tracing::debug!(
                    request_id = %request_id,
                    path = %path_for_log,
                    tokens_before = tokens_before,
                    "compression: under budget, no work"
                );
                buffered
            }
            compression::Outcome::Passthrough { reason } => {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    reason = ?reason,
                    "compression: passthrough on parse/serialize"
                );
                buffered
            }
        };

        // Forward the (possibly modified) body. reqwest sets its own
        // Content-Length from the body bytes — the existing
        // `build_forward_request_headers` already strips the
        // client-supplied Content-Length for us.
        state
            .client
            .request(reqwest_method, upstream_url.clone())
            .headers(outgoing_headers)
            .body(body_to_send)
            .send()
            .await?
    } else {
        // Pure streaming path — the original passthrough behaviour.
        let body_stream =
            TryStreamExt::map_err(req.into_body().into_data_stream(), std::io::Error::other);
        let reqwest_body = reqwest::Body::wrap_stream(body_stream);
        state
            .client
            .request(reqwest_method, upstream_url.clone())
            .headers(outgoing_headers)
            .body(reqwest_body)
            .send()
            .await?
    };

    let upstream_status = upstream_resp.status();
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
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    tracing::info!(
        request_id = %request_id,
        method = %method,
        path = %path_for_log,
        upstream_status = upstream_status.as_u16(),
        latency_ms = start.elapsed().as_millis() as u64,
        protocol = "http",
        "forwarded"
    );

    Ok(response)
}

fn ensure_request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
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
}
