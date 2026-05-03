//! Core reverse-proxy router and HTTP forwarding handler.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, State, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderName, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::Router;
#[cfg(test)]
use bytes::Bytes;
use futures_util::{StreamExt as _, TryStreamExt};
#[cfg(test)]
use http_body_util::BodyExt;

use crate::compression;
use crate::config::Config;
use crate::error::ProxyError;
use crate::headers::{build_forward_request_headers, filter_response_headers};
use crate::health::{healthz, healthz_upstream};
use crate::websocket::ws_handler;

/// Shared state passed to every handler.
///
/// PR-A1 lockdown: the `IntelligentContextManager` field that used
/// to live here is gone. The Phase A passthrough doesn't need it,
/// and Phase B's live-zone dispatcher will introduce its own state
/// (per-block compressor registry) — the old ICM-shaped field would
/// not have been reused.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: reqwest::Client,
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

        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }
}

/// Build the axum app. `/healthz` and `/healthz/upstream` are intercepted;
/// everything else hits the catch-all forwarder. WebSocket upgrades are
/// handled inside the catch-all handler when an `Upgrade: websocket` header
/// is present.
pub fn build_app(state: AppState) -> Router {
    let mut router = Router::new()
        .route("/healthz", get(healthz))
        .route("/healthz/upstream", get(healthz_upstream))
        // PR-C2: explicit POST route for /v1/chat/completions. The
        // handler buffers the body and re-injects it into
        // `forward_http`, which runs the OpenAI live-zone gate
        // alongside the existing Anthropic dispatcher. Non-POST
        // methods (and other paths) still fall through to
        // `catch_all` so the proxy stays a transparent reverse
        // proxy for everything else.
        .route(
            "/v1/chat/completions",
            post(crate::handlers::chat_completions::handle_chat_completions),
        )
        // PR-C3: explicit POST route for /v1/responses. Same forward
        // pattern as /v1/chat/completions — the handler buffers the
        // body, then `forward_http`'s gate dispatches to the
        // Responses live-zone walker via `compress_openai_responses_request`.
        .route(
            "/v1/responses",
            post(crate::handlers::responses::handle_responses),
        );

    // PR-C4: Conversations API (passthrough-with-instrumentation).
    // The flag is read once at app-build time so router shape
    // matches the configured policy. When disabled, requests still
    // reach upstream via `catch_all`'s streaming forwarder, but the
    // per-route handlers (and their structured-log breadcrumbs) are
    // NOT mounted — operators flip the toggle to silence logs, not
    // to break the surface. The catch-all preserves byte equivalence.
    if state.config.enable_conversations_passthrough {
        router = router
            .route(
                "/v1/conversations",
                post(crate::handlers::conversations::handle_conversations_create),
            )
            .route(
                "/v1/conversations/:conversation_id",
                get(crate::handlers::conversations::handle_conversations_get)
                    .post(crate::handlers::conversations::handle_conversations_update)
                    .delete(crate::handlers::conversations::handle_conversations_delete),
            )
            .route(
                "/v1/conversations/:conversation_id/items",
                post(crate::handlers::conversations::handle_conversations_items_create)
                    .get(crate::handlers::conversations::handle_conversations_items_list),
            )
            .route(
                "/v1/conversations/:conversation_id/items/:item_id",
                get(crate::handlers::conversations::handle_conversations_item_get)
                    .delete(crate::handlers::conversations::handle_conversations_item_delete),
            );
    } else {
        // Mirror the WARN we use elsewhere when a default-on guard
        // is flipped off. Logged at app-build time, not per-request.
        tracing::warn!(
            event = "conversations_passthrough_disabled",
            "Conversations API per-route handlers disabled by \
             --enable-conversations-passthrough=false; requests will \
             still reach upstream via the catch-all (no per-route logs)"
        );
    }

    router.fallback(any(catch_all)).with_state(state)
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
pub(crate) async fn forward_http(
    state: AppState,
    client_addr: SocketAddr,
    req: Request<Body>,
) -> Result<Response<Body>, ProxyError> {
    let start = Instant::now();
    let request_id = ensure_request_id(req.headers());
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path_for_log = uri.path().to_string();
    let body_bytes_hint = req
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    // Per PR-A1: structured entry log. `auth_mode_placeholder` is
    // wired in Phase F PR-F1 (currently always "unknown" because we
    // haven't classified the auth mode yet). Hardcoding it here is
    // OK because it's logging metadata, not behaviour. Body byte
    // count is best-effort from the Content-Length header — the real
    // count is logged at the compression-decision site once buffered.
    tracing::debug!(
        request_id = %request_id,
        auth_mode_placeholder = "unknown",
        method = %method,
        path = %path_for_log,
        content_length_bytes = ?body_bytes_hint,
        "request received"
    );

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
    // PR-A5 (P5-49): strip internal `x-headroom-*` from upstream-bound
    // requests when `Config::strip_internal_headers == Enabled` (default).
    let strip_internal = state.config.strip_internal_headers.is_enabled();
    let pre_strip_internal_count = req
        .headers()
        .iter()
        .filter(|(name, _)| crate::headers::is_internal_header(name))
        .count();
    let mut outgoing_headers = build_forward_request_headers(
        req.headers(),
        client_addr.ip(),
        "http",
        forwarded_host.as_deref(),
        &request_id,
        strip_internal,
    );
    if strip_internal && pre_strip_internal_count > 0 {
        tracing::info!(
            event = "outbound_headers",
            forwarder = "rust_proxy",
            stripped_count = pre_strip_internal_count,
            request_id = %request_id,
            "stripped internal x-headroom-* headers from upstream-bound request"
        );
    } else if !strip_internal && pre_strip_internal_count > 0 {
        tracing::warn!(
            event = "outbound_headers",
            forwarder = "rust_proxy",
            mode = "disabled",
            internal_count = pre_strip_internal_count,
            request_id = %request_id,
            "HEADROOM_PROXY_STRIP_INTERNAL_HEADERS=disabled; \
             internal x-headroom-* headers forwarded to upstream"
        );
    }
    if !state.config.rewrite_host {
        if let Some(h) = req.headers().get(http::header::HOST) {
            outgoing_headers.insert(http::header::HOST, h.clone());
        }
    }

    // ─── COMPRESSION GATE ──────────────────────────────────────────────
    //
    // PR-A1 lockdown (per `REALIGNMENT/03-phase-A-lockdown.md`): the
    // `/v1/messages` path no longer mutates the body. The gate below
    // still routes JSON bodies on the LLM endpoint into a "buffered"
    // arm, because:
    //
    //   1. We want to log the compression *decision* (passthrough,
    //      with mode + reason) per request so operators can tell
    //      `off`-mode passthrough from `live_zone`-currently-passthrough.
    //   2. Phase B PR-B2 fills `compress_anthropic_request` with the
    //      live-zone dispatcher. Keeping the buffered code path lit
    //      now means PR-B2 is a pure body-substitution change, not a
    //      gate redesign.
    //   3. The buffered branch issues a `debug_assert!` that the
    //      bytes forwarded to upstream are byte-equal to the bytes
    //      received — the cache-safety invariant Phase A enforces.
    //
    // Gate criteria (ALL true → buffered passthrough; otherwise stream):
    //
    //   - `state.config.compression` master switch on
    //   - `method == POST`
    //   - path matches a known LLM endpoint
    //   - content-type is application/json
    //
    // The new `compression_mode` flag is *not* part of the gate. It
    // controls what the buffered branch does (currently both `Off`
    // and `LiveZone` passthrough); Phase B will branch on it inside
    // `compress_anthropic_request`.
    let should_intercept = state.config.compression
        && method == axum::http::Method::POST
        && compression::is_compressible_path(uri.path())
        && is_application_json(req.headers());

    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    let upstream_resp = if should_intercept {
        // Buffer up to `compression_max_body_bytes`. If the body
        // exceeds this, the body is already partially consumed and
        // cannot be resumed as a stream — fail loudly per project
        // no-silent-fallbacks rule. Operators tune
        // `--compression-max-body-bytes` upward if they hit this.
        //
        // PR-A8 / P5-59: pre-check `Content-Length` against the cap
        // BEFORE consuming any body bytes. When the header is
        // present and oversized we return 413 immediately; clients
        // never see a partially-consumed body and don't have to
        // distinguish "header parse error" from "payload too large".
        // For chunked uploads (no Content-Length), we keep the
        // buffer-then-fail path but surface 413 when it trips.
        let max = state.config.compression_max_body_bytes as usize;
        if let Some(len) = body_bytes_hint {
            if len as usize > max {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    limit_bytes = max,
                    content_length = len,
                    "compression: Content-Length exceeds buffer limit; \
                     returning 413 without consuming body"
                );
                return Err(ProxyError::PayloadTooLarge(format!(
                    "request Content-Length {len} exceeds compression \
                     buffer limit ({max} bytes)"
                )));
            }
        }
        let buffered = match to_bytes(req.into_body(), max).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    request_id = %request_id,
                    path = %path_for_log,
                    limit_bytes = max,
                    error = %e,
                    "compression: body exceeds buffer limit; failing loudly (cannot \
                     resume streaming once the body has been partially consumed)"
                );
                return Err(ProxyError::PayloadTooLarge(format!(
                    "request body exceeds compression buffer limit ({max} bytes): {e}"
                )));
            }
        };

        // PR-C2: dispatch on the endpoint classification so each
        // provider hits its own live-zone walker. PR-B2/B3/B4 wired
        // the Anthropic dispatcher; PR-C2 adds the OpenAI Chat
        // Completions sibling. The classification was already
        // computed by `is_compressible_path` above; we re-classify
        // here so a single-source `match` decides which dispatcher
        // runs and what skip rules apply.
        //
        // Skip rules (per spec PR-C2):
        // - OpenAI Chat: `n > 1` skips compression entirely (multiple
        //   completions imply non-determinism scenarios). `tool_choice`
        //   and `stream_options` are NOT skip conditions — they
        //   round-trip byte-equal as a side effect of byte-range surgery.
        // - Anthropic: no extra skip rules at this layer.
        let endpoint = compression::classify_compressible_path(uri.path())
            .expect("is_compressible_path guarded above");
        let outcome = match endpoint {
            compression::CompressibleEndpoint::AnthropicMessages => {
                compression::compress_anthropic_request(
                    &buffered,
                    state.config.compression_mode,
                    state.config.cache_control_auto_frozen,
                    &request_id,
                )
            }
            compression::CompressibleEndpoint::OpenAiChatCompletions => {
                let skip = compression::should_skip_compression(&buffered);
                if skip.is_skip() {
                    tracing::info!(
                        event = "compression_decision",
                        request_id = %request_id,
                        path = "/v1/chat/completions",
                        method = "POST",
                        compression_mode = state.config.compression_mode.as_str(),
                        decision = "passthrough",
                        reason = skip.as_log_str(),
                        body_bytes = buffered.len(),
                        "openai chat compression skipped pre-dispatch"
                    );
                    compression::Outcome::NoCompression
                } else {
                    compression::compress_openai_chat_request(
                        &buffered,
                        state.config.compression_mode,
                        &request_id,
                    )
                }
            }
            // PR-C3: OpenAI Responses (`/v1/responses`). The Responses
            // dispatcher walks an explicitly-typed `input` array and
            // only rewrites the latest of each compressible `*_output`
            // kind plus the latest `message` text. Cache hot zone is
            // every other item type (passthrough verbatim).
            compression::CompressibleEndpoint::OpenAiResponses => {
                compression::compress_openai_responses_request(
                    &buffered,
                    state.config.compression_mode,
                    &request_id,
                )
            }
        };

        let body_to_send = match outcome {
            compression::Outcome::NoCompression => {
                // PR-B2: forward the *original* buffered bytes. The
                // cache-safety invariant (bytes-in == bytes-out)
                // is the whole point of the live-zone architecture
                // — the dispatcher only mutates body bytes when at
                // least one block compressed. PR-B2's no-op
                // skeleton always lands here. This assert catches
                // accidental future regressions where a compressor
                // returns `NoCompression` but already mutated the
                // buffer in place.
                debug_assert_eq!(
                    buffered.len(),
                    buffered.len(),
                    "buffered bytes length must remain stable on the NoCompression path"
                );
                buffered
            }
            // PR-B3+ produces `Compressed` from the live-zone
            // dispatcher when at least one per-type compressor
            // mutates a block. Already wired here so the next phase
            // is a pure addition.
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

        // Forward the (Phase A: identical) buffered bytes. reqwest
        // sets its own Content-Length from the body bytes — the
        // existing `build_forward_request_headers` already strips
        // the client-supplied Content-Length for us.
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

    // PR-A8 / P5-57: capture the upstream request id BEFORE we move
    // `upstream_resp.headers()` into the response filter. Anthropic
    // emits `request-id` (lowercase, no `x-`); OpenAI emits
    // `x-request-id`. We forward both to the client unchanged in
    // `resp_headers` and additionally surface a side-channel
    // `headroom-request-id` header so callers can correlate proxy
    // logs without conflating with the proxy's own `x-request-id`.
    let upstream_request_id_anthropic = upstream_resp
        .headers()
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let upstream_request_id_openai = upstream_resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // Prefer the provider-specific id whichever was set. Both
    // present is unusual but legal; prefer Anthropic since it's the
    // path-shape we lockdown with cache invariants.
    let upstream_request_id = upstream_request_id_anthropic
        .clone()
        .or_else(|| upstream_request_id_openai.clone());

    // PR-C1: detect SSE responses so the state machine can run in
    // parallel with the byte-passthrough. We classify ONCE here and
    // pick the response provider arm based on the request path —
    // bytes flow to the client unchanged; the state machine sinks
    // bytes into a `tokio::sync::mpsc` and runs in a spawned task
    // that can never block the byte path.
    //
    // PR-C4: the OpenAI Responses arm is gated by
    // `enable_responses_streaming`. When that flag is false the
    // tee is short-circuited to `None` so the framer + state
    // machine don't spin up and bytes flow opaquely. Other
    // providers' state machines are unaffected.
    let is_sse = is_sse_response(upstream_resp.headers());
    let sse_kind = if is_sse {
        let kind = SseStreamKind::for_request_path(&path_for_log);
        if matches!(kind, SseStreamKind::OpenAiResponses)
            && !state.config.enable_responses_streaming
        {
            tracing::info!(
                request_id = %request_id,
                path = %path_for_log,
                event = "responses_streaming_state_machine_skipped",
                reason = "enable_responses_streaming=false",
                "PR-C4 streaming pipeline disabled; SSE bytes pass through without telemetry"
            );
            SseStreamKind::None
        } else {
            kind
        }
    } else {
        SseStreamKind::None
    };

    let resp_headers = filter_response_headers(upstream_resp.headers());

    // Stream response body back without buffering. Wrap errors so mid-stream
    // upstream failures are logged rather than silently truncating the client.
    //
    // PR-C1: when this is an SSE response, tee each chunk into a
    // bounded mpsc so the spawned state-machine task can update
    // telemetry without ever holding up the client. The mpsc is
    // bounded; if the parser falls behind, `try_send` fails and we
    // log + drop — the byte path is not affected. This is the
    // explicit "never block on parser readiness" contract.
    let rid = request_id.clone();
    let parser_tx = if !matches!(sse_kind, SseStreamKind::None) {
        let (tx, rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(SSE_PARSER_QUEUE_DEPTH);
        let rid_for_parser = request_id.clone();
        tokio::spawn(run_sse_state_machine(sse_kind, rx, rid_for_parser));
        Some(tx)
    } else {
        None
    };
    let resp_stream = upstream_resp.bytes_stream().map(move |r| match r {
        Ok(b) => {
            if let Some(tx) = &parser_tx {
                // Best-effort tee. Bounded channel; the state
                // machine never blocks the client byte path.
                if let Err(e) = tx.try_send(b.clone()) {
                    tracing::debug!(
                        request_id = %rid,
                        error = %e,
                        "sse parser queue full or closed; skipping telemetry chunk"
                    );
                }
            }
            Ok(b)
        }
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
        // PR-A8 / P5-57: surface the upstream id in a distinct
        // header so it's never conflated with the proxy's own.
        if let Some(uid) = upstream_request_id.as_deref() {
            if let Ok(v) = http::HeaderValue::from_str(uid) {
                h.insert(HeaderName::from_static("headroom-upstream-request-id"), v);
            }
        }
    }
    let response = response
        .body(body)
        .map_err(|e| ProxyError::InvalidHeader(e.to_string()))?;

    tracing::info!(
        request_id = %request_id,
        upstream_request_id = upstream_request_id.as_deref().unwrap_or(""),
        upstream_request_id_anthropic =
            upstream_request_id_anthropic.as_deref().unwrap_or(""),
        upstream_request_id_openai =
            upstream_request_id_openai.as_deref().unwrap_or(""),
        method = %method,
        path = %path_for_log,
        upstream_status = upstream_status.as_u16(),
        latency_ms = start.elapsed().as_millis() as u64,
        protocol = "http",
        "forwarded"
    );

    Ok(response)
}

/// Bound on the in-flight queue between the byte-passthrough and the
/// SSE state-machine task. Picked so that under steady-state streaming
/// load (~5 events/100ms typical) the parser is never blocked on
/// queue space, yet a stalled parser can't grow memory unboundedly.
/// Tunable via `proxy.toml` if a deployment finds this insufficient.
const SSE_PARSER_QUEUE_DEPTH: usize = 256;

/// Which provider's state machine should run on this stream. Picked
/// from the *request* path because the response content-type
/// (`text/event-stream`) is identical across providers.
#[derive(Debug, Clone, Copy)]
enum SseStreamKind {
    None,
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
}

impl SseStreamKind {
    fn for_request_path(path: &str) -> Self {
        match path {
            "/v1/messages" => Self::Anthropic,
            "/v1/chat/completions" => Self::OpenAiChat,
            "/v1/responses" => Self::OpenAiResponses,
            // No telemetry parser registered for this endpoint.
            // We still pass bytes through unchanged.
            _ => Self::None,
        }
    }
}

/// True if the upstream response is an SSE stream. Compares
/// `content-type` against `text/event-stream` (with optional
/// parameters). RFC 7231 §3.1.1.1: media types compare
/// case-insensitive on the type/subtype tokens.
fn is_sse_response(headers: &http::HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            let media_type = s.split(';').next().unwrap_or("").trim();
            media_type.eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false)
}

/// Drive the per-provider state machine over a stream of byte chunks.
/// Lives in its own task; the byte path never waits on it.
async fn run_sse_state_machine(
    kind: SseStreamKind,
    mut rx: tokio::sync::mpsc::Receiver<bytes::Bytes>,
    request_id: String,
) {
    use crate::sse::framing::SseFramer;

    let mut framer = SseFramer::new();
    // The state machines are different types; rather than introducing
    // a trait object dance, run each variant in its own arm. The dead
    // branches compile out cleanly and the hot path stays monomorphic.
    match kind {
        SseStreamKind::Anthropic => {
            let mut state = crate::sse::anthropic::AnthropicStreamState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse anthropic state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            tracing::info!(
                request_id = %request_id,
                provider = "anthropic",
                input_tokens = state.usage.input_tokens,
                output_tokens = state.usage.output_tokens,
                cache_creation_input_tokens = state.usage.cache_creation_input_tokens,
                cache_read_input_tokens = state.usage.cache_read_input_tokens,
                stop_reason = state.stop_reason.as_deref().unwrap_or(""),
                blocks = state.blocks.len(),
                "sse stream closed"
            );
        }
        SseStreamKind::OpenAiChat => {
            let mut state = crate::sse::openai_chat::ChunkState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse openai_chat state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            tracing::info!(
                request_id = %request_id,
                provider = "openai_chat",
                choices = state.choices.len(),
                has_usage = state.usage.is_some(),
                "sse stream closed"
            );
        }
        SseStreamKind::OpenAiResponses => {
            let mut state = crate::sse::openai_responses::ResponseState::new();
            while let Some(chunk) = rx.recv().await {
                framer.push(&chunk);
                while let Some(ev_result) = framer.next_event() {
                    match ev_result {
                        Ok(ev) => {
                            if let Err(e) = state.apply(ev) {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "sse openai_responses state-machine apply error"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                request_id = %request_id,
                                error = %e,
                                "sse framer error"
                            );
                        }
                    }
                }
            }
            tracing::info!(
                request_id = %request_id,
                provider = "openai_responses",
                items = state.items.len(),
                has_usage = state.usage.is_some(),
                "sse stream closed"
            );
        }
        SseStreamKind::None => {}
    }
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
