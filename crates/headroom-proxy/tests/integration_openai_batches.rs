//! Explicit OpenAI batch route scaffolding.

mod common;

use std::sync::{Arc, Mutex};

use common::{
    start_proxy_with_config, start_proxy_with_config_and_runtime, start_proxy_with_runtime,
};
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::matchers::{method, path, path_regex, query_param};
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
async fn openai_batch_routes_forward_to_upstream() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file-123/content"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(
                serde_json::json!({
                    "custom_id": "req-0",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                })
                .to_string(),
            ),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"id": "file-compressed"})),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/batches"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"result":"create"})),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/batches"))
        .and(query_param("limit", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_string("list"))
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path_regex(r"^/v1/batches/[^/]+$"))
        .respond_with(ResponseTemplate::new(200).set_body_string("get"))
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/v1/batches/[^/]+/cancel$"))
        .respond_with(ResponseTemplate::new(200).set_body_string("cancel"))
        .mount(&upstream)
        .await;

    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .post(format!("{}/v1/batches", proxy.url()))
            .json(
                &serde_json::json!({"input_file_id":"file-123","endpoint":"/v1/chat/completions"})
            )
            .send()
            .await
            .unwrap()
            .json::<serde_json::Value>()
            .await
            .unwrap(),
        serde_json::json!({"result":"create"})
    );
    assert_eq!(
        client
            .get(format!("{}/v1/batches?limit=5", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "list"
    );
    assert_eq!(
        client
            .get(format!("{}/v1/batches/batch_123", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "get"
    );
    assert_eq!(
        client
            .post(format!("{}/v1/batches/batch_123/cancel", proxy.url()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "cancel"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_batch_routes_use_openai_target_url_for_primary_execution() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/files/file-123/content"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(
                serde_json::json!({
                    "custom_id": "req-0",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                })
                .to_string(),
            ),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"id": "file-compressed"})),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/batches"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"result":"create-direct"})),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/batches"))
        .and(query_param("limit", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_string("list-direct"))
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;
    let client = reqwest::Client::new();

    assert_eq!(
        client
            .post(format!("{}/v1/batches", proxy.url()))
            .json(
                &serde_json::json!({"input_file_id":"file-123","endpoint":"/v1/chat/completions"})
            )
            .send()
            .await
            .unwrap()
            .json::<serde_json::Value>()
            .await
            .unwrap(),
        serde_json::json!({"result":"create-direct"})
    );
    let list = client
        .get(format!("{}/v1/batches?limit=5", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(list.status(), 200);
    assert_eq!(list.text().await.unwrap(), "list-direct");

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_batch_get_route_emits_metadata() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/batches/batch_123"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "id": "batch_123" })),
        )
        .mount(&upstream)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/v1/batches/batch_123", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let routed_mode = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("route_mode")
                    == Some(&MetadataValue::String("openai_batch_get".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("batch_id")
                    == Some(&MetadataValue::String("batch_123".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.openai_batch_get");
    assert_eq!(
        routed_metadata.metadata.get("endpoint"),
        Some(&MetadataValue::String("batches".to_string()))
    );
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}

#[tokio::test]
async fn openai_batch_create_rewrites_chat_completion_input_file_before_create() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();
    let original_message = format!("{long_text}\nmessage 0");
    let batch_jsonl = serde_json::json!({
        "custom_id": "req-0",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": original_message,
            }],
        }
    })
    .to_string();

    Mock::given(method("GET"))
        .and(path("/v1/files/file-123/content"))
        .respond_with(ResponseTemplate::new(200).set_body_string(batch_jsonl))
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(move |req: &wiremock::Request| {
            let body = String::from_utf8(req.body.clone()).expect("multipart body");
            assert!(body.contains("name=\"purpose\""));
            assert!(body.contains("\r\n\r\nbatch\r\n"));
            assert!(body.contains("filename=\"compressed_file-123.jsonl\""));
            assert!(!body.contains(&original_message));
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "file-compressed"
            }))
        })
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/batches"))
        .respond_with(|req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
            assert_eq!(body["input_file_id"], serde_json::json!("file-compressed"));
            assert_eq!(
                body["metadata"]["headroom_original_file_id"],
                serde_json::json!("file-123")
            );
            assert_eq!(
                body["metadata"]["headroom_compressed"],
                serde_json::json!("true")
            );
            assert_eq!(
                body["metadata"]["headroom_total_requests"],
                serde_json::json!("1")
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "id": "batch_123",
                    "input_file_id": "file-compressed"
                }))
        })
        .mount(&direct_openai)
        .await;

    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config(config).await;

    let response = reqwest::Client::new()
        .post(format!("{}/v1/batches", proxy.url()))
        .header("authorization", "Bearer test-key")
        .json(&serde_json::json!({
            "input_file_id": "file-123",
            "endpoint": "/v1/chat/completions"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-headroom-tokens-saved").is_some());
    assert!(response
        .headers()
        .get("x-headroom-savings-percent")
        .is_some());
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["id"], serde_json::json!("batch_123"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_batch_create_emits_compressed_input_stage_for_chat_completion_batches() {
    let legacy_upstream = MockServer::start().await;
    let direct_openai = MockServer::start().await;
    let long_text = (0..40)
        .map(|index| {
            format!(
                "[12:00:{index:02}] INFO Starting build\nERROR failed to compile crate\nFAILED tests::example_{index}\ncargo error: linker failed\n"
            )
        })
        .collect::<String>();

    Mock::given(method("GET"))
        .and(path("/v1/files/file-123/content"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(
                serde_json::json!({
                    "custom_id": "req-0",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [{
                            "role": "user",
                            "content": format!("{long_text}\nmessage 0"),
                        }],
                    }
                })
                .to_string(),
            ),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"id": "file-compressed"})),
        )
        .mount(&direct_openai)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/batches"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"id": "batch_123"})),
        )
        .mount(&direct_openai)
        .await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let mut config = Config::for_test(Url::parse(&legacy_upstream.uri()).unwrap());
    config.openai_api_url = Url::parse(&direct_openai.uri()).unwrap();
    let proxy = start_proxy_with_config_and_runtime(config, runtime).await;

    let response = reqwest::Client::new()
        .post(format!("{}/v1/batches", proxy.url()))
        .json(&serde_json::json!({
            "input_file_id": "file-123",
            "endpoint": "/v1/chat/completions"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    proxy.shutdown().await;

    let recorded = events.lock().expect("poisoned");
    let compressed_event = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputCompressed
                && event.metadata.get("compression_status")
                    == Some(&MetadataValue::String("compressed".to_string()))
        })
        .expect("compressed input event");
    assert_eq!(
        compressed_event.context.operation,
        "proxy.openai_batch_create"
    );
    assert_eq!(compressed_event.context.provider, "openai");
}
