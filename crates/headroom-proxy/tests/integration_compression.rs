//! End-to-end integration tests for the compression interceptor.
//!
//! These tests boot a real Rust proxy in front of a wiremock upstream
//! and verify the request body that arrives at the upstream — i.e. we
//! observe the *actual* compression effect on the wire, not the
//! library outcome in isolation.
//!
//! Coverage:
//! - Anthropic & OpenAI: off → passthrough, on+short → unchanged,
//!   on+oversized → trimmed.
//! - Compression-on, non-JSON body: skipped (Content-Type gate).
//! - Compression-on, non-LLM path: skipped (path gate).
//! - OpenAI-specific: system message survives compression.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a /v1/messages handler that captures the upstream request body
/// into the returned Arc<Mutex<...>> for assertions, and returns 200 OK.
async fn mount_anthropic_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// Build a payload that's large enough to force ICM to trim. Uses the
/// same pattern as the `compresses_when_over_budget` unit test in the
/// anthropic module: huge max_tokens eats the budget, leaving very few
/// tokens for input and forcing drops.
fn oversized_anthropic_payload() -> Value {
    let messages: Vec<Value> = (0..30)
        .map(|i| {
            json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("padding token {i} ").repeat(20),
            })
        })
        .collect();
    json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 199_500,
        "messages": messages,
    })
}

#[tokio::test]
async fn compression_off_passes_body_unchanged() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {
        // compression remains off (Config::for_test default)
    })
    .await;

    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_eq!(got, body, "compression off — body must be byte-identical");
    proxy.shutdown().await;
}

#[tokio::test]
async fn compression_on_short_body_passes_through() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = json!({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let in_messages = payload["messages"].as_array().unwrap().len();
    let out_messages = got_json["messages"].as_array().unwrap().len();
    assert_eq!(
        out_messages, in_messages,
        "small request stays under budget; messages array unchanged"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn compression_on_oversized_body_trims_messages() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let in_messages = payload["messages"].as_array().unwrap().len();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_ne!(got, body, "ICM should have trimmed something");
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let out_messages = got_json["messages"].as_array().unwrap().len();
    assert!(
        out_messages < in_messages,
        "expected fewer messages after compression: in={in_messages}, out={out_messages}"
    );
    // Other fields preserved verbatim.
    assert_eq!(got_json["model"], payload["model"]);
    assert_eq!(got_json["max_tokens"], payload["max_tokens"]);
    proxy.shutdown().await;
}

#[tokio::test]
async fn compression_on_non_json_skips() {
    let upstream = MockServer::start().await;
    let captured = mount_anthropic_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // Path matches /v1/messages but Content-Type isn't JSON. The gate
    // must skip and stream verbatim — even though the body would
    // otherwise be massive enough to compress.
    let body = vec![0xAAu8; 64 * 1024];
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/octet-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_eq!(got, body, "non-JSON content-type must bypass compression");
    proxy.shutdown().await;
}

#[tokio::test]
async fn compression_on_non_llm_path_skips() {
    let upstream = MockServer::start().await;
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/some/other/api"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string("ok")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;

    // Same oversized JSON payload, but at a non-LLM path. The path
    // gate must skip and the body must arrive verbatim.
    let payload = oversized_anthropic_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/some/other/api", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_eq!(got, body, "non-LLM path must bypass compression");
    proxy.shutdown().await;
}

// ─── OpenAI /v1/chat/completions ─────────────────────────────────────

async fn mount_openai_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

/// Big OpenAI payload: 40 alternating user/assistant turns plus a
/// system message. max_tokens=127_000 reserves nearly the whole
/// gpt-4o-mini 128K window for output, forcing input compression.
fn oversized_openai_payload() -> Value {
    let mut messages: Vec<Value> = vec![json!({
        "role": "system",
        "content": "Always respond concisely.",
    })];
    messages.extend((0..40).map(|i| {
        json!({
            "role": if i % 2 == 0 { "user" } else { "assistant" },
            "content": format!("history turn {i} ").repeat(40),
        })
    }));
    json!({
        "model": "gpt-4o-mini",
        "max_tokens": 127_000,
        "messages": messages,
    })
}

#[tokio::test]
async fn openai_compression_off_passes_body_unchanged() {
    let upstream = MockServer::start().await;
    let captured = mount_openai_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |_| {}).await;

    let payload = oversized_openai_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_eq!(got, body, "compression off — body must be byte-identical");
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_compression_on_short_body_passes_through() {
    let upstream = MockServer::start().await;
    let captured = mount_openai_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let payload = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user",   "content": "hi"},
        ],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let in_messages = payload["messages"].as_array().unwrap().len();
    let out_messages = got_json["messages"].as_array().unwrap().len();
    assert_eq!(out_messages, in_messages);
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_compression_on_oversized_body_trims_messages_keeps_system() {
    let upstream = MockServer::start().await;
    let captured = mount_openai_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let payload = oversized_openai_payload();
    let body = serde_json::to_vec(&payload).unwrap();
    let in_messages = payload["messages"].as_array().unwrap().len();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_ne!(got, body, "ICM should have trimmed something");
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let out_msgs = got_json["messages"].as_array().unwrap();
    assert!(
        out_msgs.len() < in_messages,
        "expected fewer messages after compression: in={in_messages}, out={}",
        out_msgs.len()
    );
    // OSS-defining safety contract: the system message must survive.
    assert!(
        out_msgs
            .iter()
            .any(|m| m.get("role").and_then(Value::as_str) == Some("system")),
        "system message must be preserved after OpenAI compression"
    );
    // model + max_tokens preserved verbatim
    assert_eq!(got_json["model"], payload["model"]);
    assert_eq!(got_json["max_tokens"], payload["max_tokens"]);
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_max_completion_tokens_field_works() {
    // Reasoning models (o1/o3) only accept max_completion_tokens.
    // Verify the gate accepts that field shape.
    let upstream = MockServer::start().await;
    let captured = mount_openai_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let payload = json!({
        "model": "gpt-4o-mini",
        "max_completion_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let got = captured.lock().unwrap().clone().expect("upstream got body");
    // Small request → no compression → body unchanged
    assert_eq!(got, body);
    proxy.shutdown().await;
}
