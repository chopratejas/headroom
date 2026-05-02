//! End-to-end integration tests for the token-mode compression interceptor.
//!
//! Boots the Rust proxy in front of a wiremock upstream and verifies
//! the body that arrives at the upstream — i.e. the actual on-the-wire
//! effect of compression, not the library outcome in isolation.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn capture_path(upstream: &MockServer, p: &'static str) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    let p_owned = p.to_string();
    futures::executor::block_on(async {
        Mock::given(method("POST"))
            .and(path(p_owned))
            .respond_with(move |req: &wiremock::Request| {
                *captured_clone.lock().unwrap() = Some(req.body.clone());
                ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
            })
            .mount(upstream)
            .await;
    });
    captured
}

fn big_json_array() -> String {
    let row = r#"{"id":1,"name":"alice","email":"a@example.com","tags":["x","y","z"]}"#;
    format!("[{}]", vec![row; 1000].join(","))
}

#[tokio::test]
async fn compression_off_passes_body_unchanged() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/messages");
    let proxy = start_proxy_with(&upstream.uri(), |_| {}).await;

    let payload = json!({
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": big_json_array()}
            ]}
        ]
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
    let got = captured.lock().unwrap().clone().unwrap();
    assert_eq!(got, body);
    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_tool_result_in_new_turn_compresses() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/messages");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let big = big_json_array();
    let payload = json!({
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": big.clone()}
            ]}
        ]
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
    let got = captured.lock().unwrap().clone().unwrap();
    let _ = body.len();
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let new_tool_content = got_json["messages"][0]["content"][0]["content"]
        .as_str()
        .unwrap();
    assert!(new_tool_content.len() < big.len());
    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_prefix_messages_byte_identical() {
    // Proves the prefix-cache contract: bytes 0..LU pass through
    // exactly as the client sent them.
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/messages");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let big = big_json_array();
    let payload = json!({
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "earlier turn — preserve me"},
            {"role": "assistant", "content": "earlier reply — preserve me"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": big.clone()}
            ]}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let _ = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let got = captured.lock().unwrap().clone().unwrap();
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(
        got_json["messages"][0]["content"],
        "earlier turn — preserve me"
    );
    assert_eq!(
        got_json["messages"][1]["content"],
        "earlier reply — preserve me"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_tool_message_in_new_turn_compresses() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/chat/completions");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let big = big_json_array();
    let payload = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "list users"},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "list", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big.clone()}
        ]
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
    let got = captured.lock().unwrap().clone().unwrap();
    let _ = body.len();
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let new_tool = got_json["messages"][2]["content"].as_str().unwrap();
    assert!(new_tool.len() < big.len());
    proxy.shutdown().await;
}

#[tokio::test]
async fn responses_function_call_output_in_new_turn_compresses() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/responses");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let big = big_json_array();
    let payload = json!({
        "model": "gpt-4o-mini",
        "input": [
            {"type": "message", "role": "user", "content": "list users"},
            {"type": "function_call", "call_id": "c1",
             "name": "list", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": big.clone()}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let got = captured.lock().unwrap().clone().unwrap();
    let _ = body.len();
    let got_json: Value = serde_json::from_slice(&got).unwrap();
    let new_output = got_json["input"][2]["output"].as_str().unwrap();
    assert!(new_output.len() < big.len());
    proxy.shutdown().await;
}

#[tokio::test]
async fn responses_previous_response_id_skips_completely() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/responses");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let payload = json!({
        "model": "gpt-4o-mini",
        "previous_response_id": "resp_xyz",
        "input": "any new text would be here"
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let _ = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    let got = captured.lock().unwrap().clone().unwrap();
    assert_eq!(
        got, body,
        "previous_response_id mode must be byte-identical"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn non_json_skips() {
    let upstream = MockServer::start().await;
    let captured = capture_path(&upstream, "/v1/messages");
    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let body = vec![0xAAu8; 64 * 1024];
    let _ = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("content-type", "application/octet-stream")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    let got = captured.lock().unwrap().clone().unwrap();
    assert_eq!(got, body);
    proxy.shutdown().await;
}

#[tokio::test]
async fn non_llm_path_skips() {
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

    let proxy = start_proxy_with(&upstream.uri(), |c| c.compression = true).await;

    let big = big_json_array();
    let payload = json!({
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": big}
        ]}]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let _ = reqwest::Client::new()
        .post(format!("{}/some/other/api", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    let got = captured.lock().unwrap().clone().unwrap();
    assert_eq!(got, body);
    proxy.shutdown().await;
}
