//! Real end-to-end tests for token-mode compression against live OpenAI.
//!
//! Boots the Rust proxy with `--compression` ON and points it at
//! `https://api.openai.com`. Verifies:
//!
//! 1. `/v1/chat/completions` with a fat tool message in the new turn
//!    returns a real completion (= proxy didn't corrupt the body) AND
//!    the OpenAI-reported `usage.prompt_tokens` is significantly less
//!    than what an uncompressed proxy would have sent.
//! 2. Same for `/v1/responses` with a `function_call_output` item.
//! 3. Cross-turn prefix-cache preservation: with compression on, two
//!    requests sharing a ~5K-token prefix produce a cache hit on the
//!    second turn (`cached_tokens >= 1024`).
//!
//! Skipped unless both:
//! - `HEADROOM_E2E=1`
//! - `OPENAI_API_KEY` is set (read from .env)
//!
//! Run with:
//!     HEADROOM_E2E=1 cargo test -p headroom-proxy \
//!         --test e2e_token_mode -- --nocapture --test-threads=1

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use headroom_proxy::{build_app, AppState, Config};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use url::Url;

const E2E_GUARD: &str = "HEADROOM_E2E";

fn e2e_enabled() -> bool {
    std::env::var(E2E_GUARD).ok().as_deref() == Some("1")
}

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if p.join(".env").exists() && p.join("Cargo.toml").exists() {
            return p;
        }
        if !p.pop() {
            panic!("could not locate repo root");
        }
    }
}

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

async fn start_proxy_compression_on() -> ProxyHandle {
    let upstream: Url = "https://api.openai.com".parse().expect("valid URL");
    let mut config = Config::for_test(upstream);
    config.compression = true;
    config.upstream_timeout = Duration::from_secs(120);
    config.upstream_connect_timeout = Duration::from_secs(10);
    config.rewrite_host = true;

    let state = AppState::new(config).expect("app state");
    let app = build_app(state).into_make_service_with_connect_info::<SocketAddr>();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
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

/// Build-log style content that the LogOffload pipeline collapses
/// via Drain-style template mining. Lots of repeated lines that
/// differ only in numeric variants — exactly the shape LogTemplate
/// is designed for.
fn big_tool_output() -> String {
    let mut lines = Vec::with_capacity(2000);
    for i in 0..2000 {
        // Vary one number per line so the lines aren't byte-equal
        // but DO match a single template: `[INFO] processed item NN
        // in MMms (status=ok, retries=0)`.
        let ms = (i * 37) % 250;
        lines.push(format!(
            "[INFO] processed item {i} in {ms}ms (status=ok, retries=0)"
        ));
    }
    lines.join("\n")
}

#[tokio::test]
async fn chat_completions_tool_message_compressed_end_to_end() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    let big = big_tool_output();
    let payload = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Reply with the word 'pong'."},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "list", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big.clone()}
        ]
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("post");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(status.is_success(), "expected 2xx, got {status}: {text}");
    let body: Value = serde_json::from_str(&text).expect("response is JSON");
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "expected non-empty completion, got: {body}"
    );
    if let Some(prompt) = body["usage"]["prompt_tokens"].as_u64() {
        eprintln!("✓ chat: model received {prompt} prompt tokens");
        // Sanity: tool message itself was ~120KB / ~30K tokens raw.
        // Even a modest LogTemplate compression should drop that
        // well below 10K tokens (templates collapse ~99% of the noise).
        assert!(
            prompt < 10_000,
            "expected significant compression; got {prompt}"
        );
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn responses_function_call_output_compressed_end_to_end() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    let big = big_tool_output();
    let payload = json!({
        "model": "gpt-4o-mini",
        "max_output_tokens": 50,
        "input": [
            {"type": "message", "role": "user", "content": "Reply with the word 'pong'."},
            {"type": "function_call", "call_id": "c1",
             "name": "list", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": big.clone()}
        ]
    });

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .expect("post");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(status.is_success(), "expected 2xx, got {status}: {text}");
    let body: Value = serde_json::from_str(&text).expect("response is JSON");
    let output_arr = body["output"].as_array().expect("response.output array");
    assert!(!output_arr.is_empty(), "expected at least one output item");
    if let Some(prompt) = body["usage"]["input_tokens"].as_u64() {
        eprintln!("✓ responses: model received {prompt} input tokens");
        assert!(
            prompt < 10_000,
            "expected significant compression; got {prompt}"
        );
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn prefix_cache_preserved_through_proxy() {
    let Some(api_key) = require_e2e() else { return };
    let proxy = start_proxy_compression_on().await;

    let big_system = "You are a meticulous senior engineer. ".repeat(500);
    let payload_for = |user: &str| {
        json!({
            "model": "gpt-4o-mini",
            "max_tokens": 30,
            "messages": [
                {"role": "system", "content": big_system},
                {"role": "user", "content": user},
            ],
        })
    };

    let client = reqwest::Client::new();
    let send = |body: Value| {
        let url = format!("{}/v1/chat/completions", proxy.url());
        let key = api_key.clone();
        let c = client.clone();
        async move {
            c.post(url)
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .expect("post")
        }
    };

    let r1 = send(payload_for("First call: say hi.")).await;
    assert!(r1.status().is_success());
    let _: Value = r1.json().await.unwrap();

    tokio::time::sleep(Duration::from_secs(2)).await;

    let r2 = send(payload_for("Second call: say bye.")).await;
    assert!(r2.status().is_success());
    let r2_json: Value = r2.json().await.unwrap();
    let cached = r2_json["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total = r2_json["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    eprintln!("second call: cached_tokens={cached} of {total} prompt_tokens");
    assert!(
        cached >= 1024,
        "expected ≥1024 cached tokens; got {cached} of {total}. \
         Compression must not modify the prefix bytes."
    );
    proxy.shutdown().await;
}
