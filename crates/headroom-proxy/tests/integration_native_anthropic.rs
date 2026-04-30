//! Feature-flagged native `/v1/messages` route scaffolding.

mod common;

use std::time::Duration;

use std::sync::{Arc, Mutex};

use common::{
    start_proxy_with_config, start_proxy_with_config_and_runtime, start_streaming_upstream,
};
use futures_util::StreamExt;
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct RecordingPlugin {
    events: Arc<Mutex<Vec<PipelineEvent>>>,
}

impl PipelinePlugin for RecordingPlugin {
    fn on_event(&self, event: &mut PipelineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

#[tokio::test]
async fn native_anthropic_messages_route_forwards_when_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}]
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["content"][0]["text"], "ok");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_use_anthropic_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_anthropic = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "msg_direct",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "direct"}]
                })),
        )
        .mount(&direct_anthropic)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    config.anthropic_api_url = Url::parse(&direct_anthropic.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["content"][0]["text"], "direct");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_compress_large_messages_before_forwarding() {
    let upstream = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    let original_text = format!("{long_text}\nmessage 0");
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
            assert_ne!(body["messages"][0]["content"], serde_json::json!(original_text));
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "msg_direct",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "direct"}]
                }))
        })
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": format!("{long_text}\nmessage 0")}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_reuse_buffered_response_cache() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "msg_cache",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "cached"}]
                })),
        )
        .expect(1)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "claude-haiku-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}]
    });

    let first = client
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);

    let second = client
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 200);
    assert_eq!(
        second.json::<serde_json::Value>().await.unwrap()["content"][0]["text"],
        "cached"
    );

    let stats = client
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(stats["response_cache"]["hits"], 1);
    assert_eq!(stats["recent_requests"].as_array().unwrap().len(), 2);
    assert_eq!(stats["recent_requests"][0]["provider"], "anthropic");
    assert_eq!(stats["recent_requests"][1]["provider"], "anthropic");
    assert_eq!(stats["recent_requests"][0]["cache_hit"], false);
    assert_eq!(stats["recent_requests"][1]["cache_hit"], true);

    let feed = client
        .get(format!("{}/transformations/feed?limit=2", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 2);
    assert_eq!(feed["transformations"][0]["provider"], "anthropic");
    assert_eq!(feed["transformations"][1]["provider"], "anthropic");
    assert_eq!(
        feed["transformations"][0]["tags"]["route_mode"],
        "native_anthropic_messages"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_route_emits_native_route_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}]
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let routed = recorded
        .iter()
        .find(|event| event.stage == PipelineStage::InputRouted)
        .expect("input routed event");
    assert_eq!(routed.context.operation, "proxy.anthropic_messages");
    assert_eq!(routed.context.model, "claude-haiku-4-5");
    assert_eq!(
        routed.metadata.get("route_mode"),
        Some(&MetadataValue::String(
            "native_anthropic_messages".to_string(),
        ))
    );
}

#[tokio::test]
async fn native_anthropic_messages_emit_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}]
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": format!("{long_text}\nmessage 0")}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let compressed = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputCompressed
                && event.metadata.get("compression_status")
                    == Some(&MetadataValue::String("compressed".to_string()))
        })
        .expect("compressed input event");
    assert_eq!(compressed.context.operation, "proxy.anthropic_messages");
}

#[tokio::test]
async fn native_anthropic_messages_route_forwards_missing_passthrough_fields() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "stop_sequence": serde_json::Value::Null,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })))
        .mount(&upstream)
        .await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.json::<serde_json::Value>().await.unwrap();
    assert_eq!(body["id"], "msg_123");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_shadow_records_match_for_non_streaming_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_shadow",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}]
        })))
        .expect(2)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    config.native_anthropic_messages_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = reqwest::get(format!("{}/metrics", proxy.url()))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("headroom_anthropic_shadow_comparisons_total 1"));
    assert!(metrics.contains("headroom_anthropic_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_anthropic_shadow_mismatches_total 0"));
    assert!(metrics.contains("headroom_anthropic_shadow_skipped_total 0"));

    let health: serde_json::Value = reqwest::get(format!("{}/health", proxy.url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["config"]["native_anthropic_messages_shadow"], true);

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_shadow_compares_direct_execution_against_legacy_upstream() {
    let legacy_upstream = MockServer::start().await;
    let direct_anthropic = MockServer::start().await;
    let body = serde_json::json!({
        "id": "msg_shadow",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}]
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
        .mount(&direct_anthropic)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&legacy_upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_anthropic_messages = true;
    config.native_anthropic_messages_shadow = true;
    config.anthropic_api_url = Url::parse(&direct_anthropic.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = reqwest::get(format!("{}/metrics", proxy.url()))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("headroom_anthropic_shadow_comparisons_total 1"));
    assert!(metrics.contains("headroom_anthropic_shadow_matches_total 1"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_anthropic_messages_shadow_compares_streaming_requests() {
    let upstream = start_streaming_upstream(
        "/v1/messages",
        "text/event-stream",
        vec![
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"he\"}}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"llo\"}}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_anthropic_messages = true;
    config.native_anthropic_messages_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp
        .bytes_stream()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .fold(String::new(), |mut out, chunk| {
            out.push_str(&String::from_utf8_lossy(&chunk));
            out
        });
    assert!(body.contains("\"text\":\"he\""));
    assert!(body.contains("\"text\":\"llo\""));

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_anthropic_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_anthropic_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_anthropic_shadow_skipped_total 0"));
    assert_eq!(
        upstream.requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "primary and shadow requests should both hit the upstream"
    );

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn native_anthropic_messages_shadow_tolerates_streaming_chunk_differences() {
    let legacy_upstream = start_streaming_upstream(
        "/v1/messages",
        "text/event-stream",
        vec![
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hello\"}}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;
    let direct_anthropic = start_streaming_upstream(
        "/v1/messages",
        "text/event-stream",
        vec![
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"he\"}}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"llo\"}}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_anthropic_messages = true;
    config.native_anthropic_messages_shadow = true;
    config.anthropic_api_url = Url::parse(&format!("http://{}", direct_anthropic.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", "test-key")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_anthropic_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_anthropic_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_anthropic_shadow_mismatches_total 0"));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    direct_anthropic.task.abort();
}

async fn wait_for_metrics(proxy_url: &str, expected: &str) -> String {
    for _ in 0..20 {
        let metrics = reqwest::get(format!("{proxy_url}/metrics"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if metrics.contains(expected) {
            return metrics;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    reqwest::get(format!("{proxy_url}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}
