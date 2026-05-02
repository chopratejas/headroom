//! Real end-to-end tests: Rust proxy (compression on) → OpenAI Chat Completions.
//!
//! Spawns the Rust proxy bound to `https://api.openai.com` with
//! `--compression` enabled, then sends real `/v1/chat/completions`
//! requests through it. Verifies:
//!
//! 1. Small request returns a real completion unchanged.
//! 2. Oversized request gets compressed AND still returns a real
//!    completion (i.e. the compression didn't corrupt the body in a
//!    way the upstream would reject).
//! 3. Streaming request works through the proxy (request is buffered,
//!    response stream passes through chunk-by-chunk).
//!
//! Skipped unless both:
//! - `HEADROOM_E2E=1`
//! - `OPENAI_API_KEY` is set (read from .env at the repo root)
//!
//! Run with:
//!     HEADROOM_E2E=1 cargo test -p headroom-proxy \
//!         --test e2e_openai_compression -- --nocapture

mod common;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use headroom_proxy::{build_app, AppState, Config};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use url::Url;

const E2E_GUARD: &str = "HEADROOM_E2E";

fn e2e_enabled() -> bool {
    std::env::var(E2E_GUARD).ok().as_deref() == Some("1")
}

/// Locate repo root by walking up from CARGO_MANIFEST_DIR until we find .env.
fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if p.join(".env").exists() && p.join("Cargo.toml").exists() {
            return p;
        }
        if !p.pop() {
            panic!("could not locate repo root (no .env found)");
        }
    }
}

/// Best-effort .env loader. Does NOT print values.
fn load_dotenv() {
    let env_path = repo_root().join(".env");
    let Ok(contents) = std::fs::read_to_string(&env_path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if v.is_empty() {
            continue;
        }
        if std::env::var(k).is_err() {
            // SAFETY: this is test setup, single-threaded before any
            // tokio task spawns concurrent work.
            std::env::set_var(k, v);
        }
    }
}

struct ProxyHandle {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl ProxyHandle {
    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

/// Start a Rust proxy with compression ON, pointed at the real
/// OpenAI API. The harness's default `start_proxy_with` only takes
/// http upstreams (it builds a `Url` first), so we replicate that
/// here for an https upstream.
async fn start_proxy_compression_on() -> ProxyHandle {
    let upstream: Url = "https://api.openai.com"
        .parse()
        .expect("valid OpenAI base URL");
    let mut config = Config::for_test(upstream);
    config.compression = true;
    // OpenAI replies in seconds for non-streaming and within ~60s for
    // long streams. Use a generous timeout so a slow model doesn't
    // make the test flake.
    config.upstream_timeout = Duration::from_secs(120);
    config.upstream_connect_timeout = Duration::from_secs(10);
    // OpenAI rejects requests with the wrong Host header; preserve
    // the client's Host so reqwest sets `api.openai.com` upstream.
    config.rewrite_host = true;

    let state = AppState::new(config).expect("app state");
    let app = build_app(state).into_make_service_with_connect_info::<SocketAddr>();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    ProxyHandle {
        addr,
        shutdown: Some(tx),
        task,
    }
}

/// Skip helper: returns Some(api_key) when ready to run, None when
/// the test should bail out cleanly.
fn require_e2e() -> Option<String> {
    if !e2e_enabled() {
        eprintln!("skipping: HEADROOM_E2E != 1");
        return None;
    }
    load_dotenv();
    match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => Some(k),
        _ => {
            eprintln!("skipping: OPENAI_API_KEY not set in .env");
            None
        }
    }
}

#[tokio::test]
async fn small_request_round_trips_through_proxy() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    let payload = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Reply with exactly the word 'pong'."}
        ],
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("post through proxy");

    assert!(
        resp.status().is_success(),
        "OpenAI returned {}: {}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("response is JSON");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    assert!(
        !content.is_empty(),
        "expected non-empty content; got: {body}"
    );
    eprintln!("✓ small_request_round_trips: model said {content:?}");

    proxy.shutdown().await;
}

#[tokio::test]
async fn oversized_request_compresses_and_still_returns_completion() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    // Build a payload that EXCEEDS the model's input budget so ICM
    // is forced to fire. gpt-4o-mini has a 128K context window; with
    // max_tokens=4096 reserved for output, the input budget is ~124K
    // tokens.
    //
    // Note: OpenAI rejects requests where max_tokens > the model's
    // output cap (16384 for gpt-4o-mini). So we can't shrink the
    // input budget by inflating max_tokens — we have to inflate the
    // input itself past the 124K-token ceiling and rely on ICM to
    // trim it back below that line.
    //
    // Density: each turn is ~"padding word " × 1000 ≈ 13K chars ≈
    // 3K tokens. 80 turns ≈ 240K tokens of history — well above the
    // 124K input budget. ICM must drop ~half the history.
    let mut messages: Vec<Value> = vec![json!({
        "role": "system",
        "content": "You are a terse assistant. Respond with exactly one word."
    })];
    for i in 0..80 {
        messages.push(json!({
            "role": if i % 2 == 0 { "user" } else { "assistant" },
            "content": format!("noise turn {i}: ").to_string()
                + &"padding word ".repeat(1_000),
        }));
    }
    // The final user turn — explicit instruction. ICM's safety rails
    // keep last_n_turns intact; this must survive every drop.
    messages.push(json!({
        "role": "user",
        "content": "Reply with exactly the word: pong",
    }));

    let payload = json!({
        "model": "gpt-4o-mini",
        // Modest output cap — well within the model's 16384 limit.
        "max_tokens": 50,
        "messages": messages,
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("post through proxy");

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "expected 2xx after compression; got {status}: {text}"
    );
    let body: Value = serde_json::from_str(&text).expect("response is JSON");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    assert!(
        !content.is_empty(),
        "expected non-empty content after compression; got: {body}"
    );
    // Usage block tells us how many tokens OpenAI actually saw — a
    // strong signal that ICM trimmed before we hit the upstream.
    if let Some(prompt_tokens) = body["usage"]["prompt_tokens"].as_u64() {
        eprintln!(
            "✓ oversized_request: model said {content:?}, OpenAI saw {prompt_tokens} prompt tokens"
        );
        // ~50 padding turns × 80-token repeat × ~3 char/token would
        // be enormous. After compression we expect << 100K prompt
        // tokens — anything below the 128K window is a successful
        // compression-then-completion.
        assert!(
            prompt_tokens < 128_000,
            "compressed prompt tokens ({prompt_tokens}) should fit within window"
        );
    } else {
        eprintln!("✓ oversized_request: model said {content:?} (no usage block)");
    }

    proxy.shutdown().await;
}

#[tokio::test]
async fn streaming_request_round_trips_through_proxy() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    let payload = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 30,
        "stream": true,
        "messages": [
            {"role": "user", "content": "Count to three. One number per line."}
        ],
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("post through proxy");

    assert!(
        resp.status().is_success(),
        "stream init failed: {} {}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );
    // SSE chunks should arrive — proxy forwards the response stream
    // chunk-by-chunk because compression only affects the request.
    let mut stream = resp.bytes_stream();
    let mut total_bytes = 0usize;
    let mut chunks = 0usize;
    let mut saw_done = false;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.expect("stream chunk");
        total_bytes += bytes.len();
        chunks += 1;
        if let Ok(s) = std::str::from_utf8(&bytes) {
            if s.contains("[DONE]") {
                saw_done = true;
            }
        }
    }
    assert!(chunks > 1, "expected multiple SSE chunks, got {chunks}");
    assert!(total_bytes > 0);
    assert!(saw_done, "expected `[DONE]` sentinel in stream");
    eprintln!("✓ streaming_request: {chunks} chunks, {total_bytes} bytes total");

    proxy.shutdown().await;
}

// Force a use of the harness's `Arc` type so the `unused_imports`
// lint doesn't kick in on the lone-test path; tests above don't need
// it directly but having it imported keeps the file self-consistent
// when extending later. (No-op at runtime.)
#[allow(dead_code)]
fn _harness_arc_kept_alive() -> Arc<()> {
    Arc::new(())
}

#[allow(dead_code)]
fn _common_module_referenced() {
    // Reference the module so cargo doesn't complain about unused
    // mod common when all tests skip on a CI that has no key.
    let _ = std::any::type_name::<common::ProxyHandle>();
}
