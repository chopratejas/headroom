//! Feature-flagged native Google Cloud Code Assist stream alias scaffolding.

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
use wiremock::matchers::{method, path};
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
async fn google_cloudcode_stream_alias_forwards_when_enabled() {
    let upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"he\"}\n\n",
            "data: {\"chunk\":\"llo\"}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "userAgent": "pi-coding-agent",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
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
    assert!(body.contains("\"chunk\":\"he\""));
    assert!(body.contains("\"chunk\":\"llo\""));

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn google_cloudcode_stream_alias_uses_cloudcode_target_url_for_primary_execution() {
    let legacy_upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec!["data: {\"chunk\":\"legacy\"}\n\n"],
        Duration::from_millis(10),
    )
    .await;
    let cloudcode_target = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec!["data: {\"chunk\":\"direct\"}\n\n"],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    config.cloudcode_api_url = Url::parse(&format!("http://{}", cloudcode_target.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("\"chunk\":\"direct\""));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    cloudcode_target.task.abort();
}

#[tokio::test]
async fn google_cloudcode_stream_alias_compresses_large_text_contents_before_forwarding() {
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
        .and(path("/v1internal:streamGenerateContent"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
            assert_ne!(
                body["request"]["contents"][0]["parts"][0]["text"],
                serde_json::json!(original_text)
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: {\"chunk\":\"ok\"}\n\n")
        })
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_google_cloudcode_stream = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": format!("{long_text}\nmessage 0")}],
                }],
                "systemInstruction": {"parts": [{"text": "be terse"}]}
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("\"chunk\":\"ok\""));

    proxy.shutdown().await;
}

#[tokio::test]
async fn google_cloudcode_stream_v1_alias_forwards_when_enabled() {
    let upstream = start_streaming_upstream(
        "/v1/v1internal:streamGenerateContent",
        "text/event-stream",
        vec!["data: {\"chunk\":\"ok\"}\n\n"],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("\"chunk\":\"ok\""));

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn google_cloudcode_stream_alias_emits_native_route_metadata() {
    let upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec!["data: {\"chunk\":\"ok\"}\n\n"],
        Duration::from_millis(10),
    )
    .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
    upstream.task.abort();

    let recorded = events.lock().expect("poisoned");
    let routed_mode = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("route_mode")
                    == Some(&MetadataValue::String(
                        "native_google_cloudcode_stream".to_string(),
                    ))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("antigravity") == Some(&MetadataValue::Bool(false))
        })
        .expect("input routed metadata event");
    assert_eq!(
        routed_mode.context.operation,
        "proxy.google_cloudcode_stream"
    );
    assert_eq!(routed_mode.context.model, "gemini-3.1-pro-high");
    assert_eq!(
        routed_metadata.context.operation,
        "proxy.google_cloudcode_stream"
    );
    assert_eq!(routed_metadata.context.model, "gemini-3.1-pro-high");
    assert_eq!(
        routed_metadata.metadata.get("antigravity"),
        Some(&MetadataValue::Bool(false))
    );
}

#[tokio::test]
async fn google_cloudcode_stream_alias_emits_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1internal:streamGenerateContent"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: {\"chunk\":\"ok\"}\n\n"),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_google_cloudcode_stream = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": format!("{long_text}\nmessage 0")}],
                }],
                "systemInstruction": {"parts": [{"text": "be terse"}]}
            }
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
        "proxy.google_cloudcode_stream"
    );
}

#[tokio::test]
async fn google_cloudcode_stream_alias_shadow_compares_streaming_requests() {
    let upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"he\"}\n\n",
            "data: {\"chunk\":\"llo\"}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config = Config::for_test(Url::parse(&format!("http://{}", upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    config.native_google_cloudcode_stream_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
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
    assert!(body.contains("\"chunk\":\"he\""));
    assert!(body.contains("\"chunk\":\"llo\""));

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_google_cloudcode_stream_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_google_cloudcode_stream_shadow_matches_total 1"));
    assert_eq!(
        upstream.requests.load(std::sync::atomic::Ordering::Relaxed),
        2,
        "primary and shadow requests should both hit the upstream"
    );

    proxy.shutdown().await;
    upstream.task.abort();
}

#[tokio::test]
async fn google_cloudcode_stream_shadow_compares_direct_execution_against_legacy_upstream() {
    let legacy_upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"he\"}\n\n",
            "data: {\"chunk\":\"llo\"}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;
    let cloudcode_target = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"he\"}\n\n",
            "data: {\"chunk\":\"llo\"}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    config.native_google_cloudcode_stream_shadow = true;
    config.cloudcode_api_url = Url::parse(&format!("http://{}", cloudcode_target.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_google_cloudcode_stream_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_google_cloudcode_stream_shadow_matches_total 1"));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    cloudcode_target.task.abort();
}

#[tokio::test]
async fn google_cloudcode_stream_shadow_tolerates_streaming_chunk_differences() {
    let legacy_upstream = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"hello\"}\n\n",
            "data: {\"done\":true}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;
    let cloudcode_target = start_streaming_upstream(
        "/v1internal:streamGenerateContent",
        "text/event-stream",
        vec![
            "data: {\"chunk\":\"he\"}\n\n",
            "data: {\"chunk\":\"llo\"}\n\n",
            "data: {\"done\":true}\n\n",
        ],
        Duration::from_millis(10),
    )
    .await;

    let mut config =
        Config::for_test(Url::parse(&format!("http://{}", legacy_upstream.addr)).unwrap());
    config.native_google_cloudcode_stream = true;
    config.native_google_cloudcode_stream_shadow = true;
    config.cloudcode_api_url = Url::parse(&format!("http://{}", cloudcode_target.addr)).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1internal:streamGenerateContent?alt=sse",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "project": "test-project",
            "model": "gemini-3.1-pro-high",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_google_cloudcode_stream_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_google_cloudcode_stream_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_google_cloudcode_stream_shadow_mismatches_total 0"));

    proxy.shutdown().await;
    legacy_upstream.task.abort();
    cloudcode_target.task.abort();
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
