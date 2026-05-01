//! Feature-flagged native `/v1/chat/completions` route scaffolding.

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
async fn native_openai_chat_route_forwards_when_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "ok");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_uses_openai_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(502))
        .mount(&legacy_upstream)
        .await;

    let direct_openai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "chatcmpl_direct",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "direct"}}]
                })),
        )
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_openai_chat = true;
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "direct");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_streaming_injects_include_usage() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("[DONE]"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_reuses_buffered_response_cache_and_cache_clear_invalidates_it() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl_cache",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "cached"}}]
        })))
        .expect(2)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let first = client
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);

    let second = client
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 200);
    assert_eq!(
        second.json::<serde_json::Value>().await.unwrap()["choices"][0]["message"]["content"],
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
    assert_eq!(stats["response_cache"]["misses"], 1);
    assert_eq!(stats["recent_requests"].as_array().unwrap().len(), 2);
    assert_eq!(stats["recent_requests"][0]["provider"], "openai");
    assert_eq!(stats["recent_requests"][0]["model"], "gpt-4o-mini");
    assert_eq!(stats["recent_requests"][0]["cache_hit"], false);
    assert_eq!(stats["recent_requests"][1]["cache_hit"], true);
    assert_eq!(
        stats["recent_requests"][0]["input_tokens_original"],
        stats["recent_requests"][0]["input_tokens_optimized"]
    );

    let feed = client
        .get(format!("{}/transformations/feed?limit=2", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["log_full_messages"], false);
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 2);
    assert_eq!(feed["transformations"][0]["provider"], "openai");
    assert_eq!(feed["transformations"][1]["provider"], "openai");
    assert_eq!(feed["transformations"][0]["cache_hit"], false);
    assert_eq!(feed["transformations"][1]["cache_hit"], true);
    assert_eq!(
        feed["transformations"][0]["tags"]["route_mode"],
        "native_openai_chat"
    );

    let metrics = client
        .get(format!("{}/metrics", proxy.url()))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("headroom_response_cache_hits_total 1"));
    assert!(metrics.contains("headroom_response_cache_misses_total 1"));

    let cleared = client
        .post(format!("{}/cache/clear", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(cleared["previous_response_size"], 1);

    let third = client
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(third.status(), 200);

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_route_emits_native_route_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl_2",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "stream": true,
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
    assert_eq!(routed.context.operation, "proxy.openai_chat");
    assert_eq!(routed.context.model, "gpt-4o-mini");
    assert_eq!(
        routed.metadata.get("route_mode"),
        Some(&headroom_runtime::MetadataValue::String(
            "native_openai_chat".to_string(),
        ))
    );
}

#[tokio::test]
async fn native_openai_chat_route_rejects_invalid_requests() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("missing string field `model`"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_emits_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "chatcmpl_compressed",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

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

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": messages
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
        .expect("compressed input stage event");
    assert_eq!(compressed.context.operation, "proxy.openai_chat");
    assert_eq!(compressed.context.model, "gpt-4o-mini");
}

#[tokio::test]
async fn native_openai_chat_shadow_records_match_for_non_streaming_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl_shadow",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })))
        .expect(2)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    config.native_openai_chat_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
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
    assert!(metrics.contains("headroom_openai_shadow_comparisons_total 1"));
    assert!(metrics.contains("headroom_openai_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_openai_shadow_mismatches_total 0"));
    assert!(metrics.contains("headroom_openai_shadow_skipped_total 0"));

    let feed: serde_json::Value =
        reqwest::get(format!("{}/transformations/feed?limit=1", proxy.url()))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 1);
    assert_eq!(feed["transformations"][0]["provider"], "openai");
    assert_eq!(
        feed["transformations"][0]["tags"]["route_mode"],
        "native_openai_chat"
    );

    let health: serde_json::Value = reqwest::get(format!("{}/health", proxy.url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["config"]["native_openai_chat_shadow"], true);

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_shadow_compares_direct_execution_against_legacy_upstream() {
    let legacy_upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl_shadow",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })))
        .expect(1)
        .mount(&legacy_upstream)
        .await;

    let direct_openai = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl_shadow",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })))
        .expect(1)
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_openai_chat = true;
    config.native_openai_chat_shadow = true;
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
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
    assert!(metrics.contains("headroom_openai_shadow_comparisons_total 1"));
    assert!(metrics.contains("headroom_openai_shadow_matches_total 1"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_rejects_oversized_request_body() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_openai_chat = true;
    config.max_body_bytes = 32;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "this request body is intentionally too large"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 413);
    let body = resp.text().await.unwrap();
    assert!(body.contains("request too large"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_openai_chat_shadow_compares_streaming_requests() {
    let upstream = start_streaming_upstream(
        "/v1/chat/completions",
        "text/event-stream",
        vec![
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_openai_chat = true;
    config.native_openai_chat_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
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
    assert!(body.contains("\"content\":\"he\""));
    assert!(body.contains("\"content\":\"llo\""));

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_openai_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_openai_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_openai_shadow_skipped_total 0"));
    assert_eq!(
        upstream.requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "primary and shadow requests should both hit the upstream"
    );

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn native_openai_chat_shadow_tolerates_streaming_chunk_differences() {
    let legacy_upstream = start_streaming_upstream(
        "/v1/chat/completions",
        "text/event-stream",
        vec![
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;
    let direct_openai = start_streaming_upstream(
        "/v1/chat/completions",
        "text/event-stream",
        vec![
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_openai_chat = true;
    config.native_openai_chat_shadow = true;
    config.openai_api_url = Url::parse(&format!("http://{}", direct_openai.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_openai_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_openai_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_openai_shadow_mismatches_total 0"));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    direct_openai.task.abort();
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
