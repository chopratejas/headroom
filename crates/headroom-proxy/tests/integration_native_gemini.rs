//! Feature-flagged native Gemini `:generateContent` route scaffolding.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

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
async fn native_gemini_generate_content_route_forwards_when_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_json(serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "ok"}]}
                    }]
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["candidates"][0]["content"]["parts"][0]["text"], "ok");

    let client = reqwest::Client::new();
    let feed = client
        .get(format!("{}/transformations/feed?limit=1", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 1);
    assert_eq!(feed["transformations"][0]["provider"], "gemini");
    assert_eq!(feed["transformations"][0]["model"], "gemini-2.0-flash");
    assert_eq!(
        feed["transformations"][0]["tags"]["route_mode"],
        "native_gemini_generate_content"
    );
    assert!(
        feed["transformations"][0]["input_tokens_original"]
            .as_u64()
            .unwrap()
            > 0
    );

    let stats = client
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(stats["recent_requests"].as_array().unwrap().len(), 1);
    assert_eq!(stats["recent_requests"][0]["provider"], "gemini");
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_uses_gemini_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_gemini = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "direct"}]}
                    }]
                })),
        )
        .mount(&direct_gemini)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    config.gemini_api_url = Url::parse(&direct_gemini.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        json["candidates"][0]["content"]["parts"][0]["text"],
        "direct"
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_compresses_large_text_contents_before_forwarding() {
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
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
            assert_ne!(
                body["contents"][0]["parts"][0]["text"],
                serde_json::json!(original_text)
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "candidates": [{
                        "content": {"parts": [{"text": "direct"}]}
                    }]
                }))
        })
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": format!("{long_text}\nmessage 0")}],
                }
            ],
            "systemInstruction": {"parts": [{"text": "be terse"}]}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_route_emits_native_route_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]}
            }]
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}],
            "systemInstruction": {"parts": [{"text": "be terse"}]}
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
    assert_eq!(routed.context.operation, "proxy.gemini_generate_content");
    assert_eq!(routed.context.model, "gemini-2.0-flash");
    assert_eq!(
        routed.metadata.get("route_mode"),
        Some(&MetadataValue::String(
            "native_gemini_generate_content".to_string(),
        ))
    );
}

#[tokio::test]
async fn native_gemini_generate_content_emits_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]}
            }]
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": format!("{long_text}\nmessage 0")}],
            }],
            "systemInstruction": {"parts": [{"text": "be terse"}]}
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
    assert_eq!(
        compressed.context.operation,
        "proxy.gemini_generate_content"
    );
}

#[tokio::test]
async fn native_gemini_generate_content_route_forwards_missing_contents() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": []
        })))
        .mount(&upstream)
        .await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "systemInstruction": {"parts": [{"text": "hello"}]}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.json::<serde_json::Value>().await.unwrap();
    assert_eq!(body["candidates"], serde_json::json!([]));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_rejects_oversized_contents_array() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config(config).await;

    let contents = (0..10_001)
        .map(|index| serde_json::json!({"parts": [{"text": format!("item-{index}")}]}))
        .collect::<Vec<_>>();
    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({ "contents": contents }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("array field `contents` exceeds max length 10000"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_shadow_records_match_for_non_streaming_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]}
            }]
        })))
        .expect(2)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    config.native_gemini_generate_content_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_gemini_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_gemini_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_gemini_shadow_mismatches_total 0"));
    assert!(metrics.contains("headroom_gemini_shadow_skipped_total 0"));

    let feed: serde_json::Value =
        reqwest::get(format!("{}/transformations/feed?limit=1", proxy.url()))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 1);
    assert_eq!(feed["transformations"][0]["provider"], "gemini");
    assert_eq!(
        feed["transformations"][0]["tags"]["route_mode"],
        "native_gemini_generate_content"
    );

    let health: serde_json::Value = reqwest::get(format!("{}/health", proxy.url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        health["config"]["native_gemini_generate_content_shadow"],
        true
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_shadow_compares_direct_execution_against_legacy_upstream() {
    let legacy_upstream = MockServer::start().await;
    let direct_gemini = MockServer::start().await;
    let body = serde_json::json!({
        "candidates": [{
            "content": {"parts": [{"text": "ok"}]}
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
        .mount(&direct_gemini)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&legacy_upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_gemini_generate_content = true;
    config.native_gemini_generate_content_shadow = true;
    config.gemini_api_url = Url::parse(&direct_gemini.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_gemini_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_gemini_shadow_matches_total 1"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_generate_content_shadow_compares_streaming_requests() {
    let upstream = start_streaming_upstream(
        "/v1beta/models/gemini-2.0-flash:generateContent",
        "text/event-stream",
        vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}]}}]}\n\n",
            "data: {\"done\":true}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_gemini_generate_content = true;
    config.native_gemini_generate_content_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key&alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
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

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_gemini_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_gemini_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_gemini_shadow_skipped_total 0"));
    assert_eq!(
        upstream.requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "primary and shadow requests should both hit the upstream"
    );

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn native_gemini_stream_generate_content_route_forwards_when_enabled() {
    let upstream = start_streaming_upstream(
        "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}]}}]}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_gemini_generate_content = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
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

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn native_gemini_stream_generate_content_shadow_compares_streaming_requests() {
    let upstream = start_streaming_upstream(
        "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}]}}]}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_gemini_generate_content = true;
    config.native_gemini_generate_content_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
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

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_gemini_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_gemini_shadow_matches_total 1"));
    assert_eq!(
        upstream.requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "primary and shadow requests should both hit the upstream"
    );

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn native_gemini_stream_generate_content_shadow_tolerates_streaming_chunk_differences() {
    let legacy_upstream = start_streaming_upstream(
        "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}\n\n",
            "data: {\"done\":true}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;
    let direct_gemini = start_streaming_upstream(
        "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}]}}]}\n\n",
            "data: {\"done\":true}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_gemini_generate_content = true;
    config.native_gemini_generate_content_shadow = true;
    config.gemini_api_url = Url::parse(&format!("http://{}", direct_gemini.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics =
        wait_for_metrics(&proxy.url(), "headroom_gemini_shadow_comparisons_total 1").await;
    assert!(metrics.contains("headroom_gemini_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_gemini_shadow_mismatches_total 0"));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    direct_gemini.task.abort();
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
