//! Feature-flagged native Gemini `:countTokens` route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{start_proxy_with_config, start_proxy_with_config_and_runtime};
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
async fn native_gemini_count_tokens_route_forwards_when_enabled() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .and(body_json(serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "totalTokens": 42
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
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
    assert_eq!(json["totalTokens"], 42);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_route_handles_large_text_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .and(body_json(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat"}]}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "totalTokens": 25
                })),
        )
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": "repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat"
                }]
            }]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["totalTokens"], 25);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_use_gemini_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_gemini = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "totalTokens": 77
                })),
        )
        .mount(&direct_gemini)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    config.gemini_api_url = Url::parse(&direct_gemini.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
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
    assert_eq!(json["totalTokens"], 77);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_route_emits_native_route_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "totalTokens": 42
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
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
    assert_eq!(routed.context.operation, "proxy.gemini_count_tokens");
    assert_eq!(routed.context.model, "gemini-2.0-flash");
    assert_eq!(
        routed.metadata.get("route_mode"),
        Some(&MetadataValue::String(
            "native_gemini_count_tokens".to_string(),
        ))
    );
}

#[tokio::test]
async fn native_gemini_count_tokens_route_emits_compressed_input_stage_for_large_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "totalTokens": 42
        })))
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    let contents = (0..6)
        .map(|index| {
            serde_json::json!({
                "role": "user",
                "parts": [{"text": format!("{long_text}\nmessage {index}")}],
            })
        })
        .collect::<Vec<_>>();

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": contents
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
    assert_eq!(compressed.context.operation, "proxy.gemini_count_tokens");
    assert_eq!(compressed.context.model, "gemini-2.0-flash");
}

#[tokio::test]
async fn native_gemini_count_tokens_route_forwards_missing_contents() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "totalTokens": 0
        })))
        .mount(&upstream)
        .await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
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
    assert_eq!(body["totalTokens"], 0);
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_route_rejects_invalid_system_instruction_shape() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}],
            "systemInstruction": "be terse"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("field `systemInstruction` must be an object"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_shadow_records_match_for_non_streaming_requests() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "totalTokens": 42
        })))
        .expect(2)
        .mount(&upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    config.native_gemini_count_tokens_shadow = true;
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_gemini_count_tokens_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_gemini_count_tokens_shadow_matches_total 1"));
    assert!(metrics.contains("headroom_gemini_count_tokens_shadow_mismatches_total 0"));

    let health: serde_json::Value = reqwest::get(format!("{}/health", proxy.url()))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["config"]["native_gemini_count_tokens_shadow"], true);

    proxy.shutdown().await;
}

#[tokio::test]
async fn native_gemini_count_tokens_shadow_compares_direct_execution_against_legacy_upstream() {
    let legacy_upstream = MockServer::start().await;
    let direct_gemini = MockServer::start().await;
    let body = serde_json::json!({
        "totalTokens": 42
    });
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
        .mount(&direct_gemini)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:countTokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&legacy_upstream)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.native_gemini_count_tokens = true;
    config.native_gemini_count_tokens_shadow = true;
    config.gemini_api_url = Url::parse(&direct_gemini.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{}/v1beta/models/gemini-2.0-flash:countTokens?key=test-key",
            proxy.url()
        ))
        .json(&serde_json::json!({
            "contents": [{"parts": [{"text": "hello"}]}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let metrics = wait_for_metrics(
        &proxy.url(),
        "headroom_gemini_count_tokens_shadow_comparisons_total 1",
    )
    .await;
    assert!(metrics.contains("headroom_gemini_count_tokens_shadow_matches_total 1"));

    proxy.shutdown().await;
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
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    reqwest::get(format!("{proxy_url}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}
