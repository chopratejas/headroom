//! Explicit CCR/telemetry/TOIN/product route scaffolding.

mod common;

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use common::start_proxy_with_runtime;
use headroom_proxy::Config;
use headroom_runtime::{
    MetadataValue, PipelineDispatcher, PipelineEvent, PipelinePlugin, PipelineStage,
};
use url::Url;
use wiremock::MockServer;

struct RecordingPlugin {
    events: Arc<Mutex<Vec<PipelineEvent>>>,
}

impl PipelinePlugin for RecordingPlugin {
    fn on_event(&self, event: &mut PipelineEvent) {
        self.events.lock().expect("poisoned").push(event.clone());
    }
}

fn assert_router_strategy_marker(response: &serde_json::Value, expected_prefix: &str) {
    let transforms = response["transforms_applied"].as_array().unwrap();
    assert!(transforms
        .iter()
        .filter_map(serde_json::Value::as_str)
        .any(|value| value.starts_with(expected_prefix)));
}

fn assert_transform_summary_matches_applied(response: &serde_json::Value) {
    let transforms = response["transforms_applied"].as_array().unwrap();
    let mut expected = BTreeMap::new();
    for transform in transforms.iter().filter_map(serde_json::Value::as_str) {
        *expected.entry(transform.to_string()).or_insert(0u64) += 1;
    }
    let actual = response["transforms_summary"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.as_u64().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn compress_route_is_native() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let invalid = client
        .post(format!("{}/v1/compress", proxy.url()))
        .header("content-type", "application/json")
        .body("{")
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(invalid["error"]["type"], "invalid_request");

    let missing_model = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "messages": [{"role": "tool", "content": "[]"}]
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        missing_model["error"]["message"],
        "Missing required field: model"
    );

    let bypass_body = serde_json::json!({
        "messages": [{"role": "tool", "content": "{\"hello\":\"world\"}"}]
    });
    let bypass = client
        .post(format!("{}/v1/compress", proxy.url()))
        .header("x-headroom-bypass", "true")
        .json(&bypass_body)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(bypass["messages"], bypass_body["messages"]);
    assert_eq!(bypass["transforms_applied"], serde_json::json!([]));
    assert!(bypass
        .as_object()
        .unwrap()
        .get("transforms_summary")
        .is_none());
    assert!(bypass.as_object().unwrap().get("route_counts").is_none());

    let big_content = serde_json::to_string(&serde_json::Value::Array(
        (0..24)
            .map(|index| {
                serde_json::json!({
                    "id": index,
                    "status": if index % 5 == 0 { "warn" } else { "ok" },
                    "service": "billing-api",
                    "duration_ms": 120 + index,
                    "message": format!("request {} completed with verbose payload", index),
                    "payload": {
                        "user_id": format!("user-{:03}", index),
                        "region": "us-east-1",
                        "feature_flag": "enterprise-rollout",
                    }
                })
            })
            .collect(),
    ))
    .unwrap();

    let compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": big_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&compressed, "router:smart_crusher:");
    assert_transform_summary_matches_applied(&compressed);
    assert_eq!(compressed["ccr_hashes"], serde_json::json!([]));
    assert!(compressed
        .as_object()
        .unwrap()
        .get("route_counts")
        .is_none());
    assert!(
        compressed["tokens_after"].as_u64().unwrap()
            < compressed["tokens_before"].as_u64().unwrap()
    );
    assert_ne!(
        compressed["messages"][0]["content"],
        serde_json::json!(big_content)
    );

    let feed = client
        .get(format!("{}/transformations/feed?limit=1", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feed["log_full_messages"], false);
    assert_eq!(feed["transformations"].as_array().unwrap().len(), 1);
    assert_eq!(feed["transformations"][0]["provider"], "headroom");
    assert_eq!(feed["transformations"][0]["model"], "gpt-4o");
    assert_eq!(
        feed["transformations"][0]["input_tokens_original"],
        compressed["tokens_before"]
    );
    assert_eq!(
        feed["transformations"][0]["input_tokens_optimized"],
        compressed["tokens_after"]
    );
    assert_eq!(
        feed["transformations"][0]["tokens_saved"],
        compressed["tokens_saved"]
    );
    assert_eq!(
        feed["transformations"][0]["transforms_applied"],
        compressed["transforms_applied"]
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
    assert_eq!(stats["recent_requests"][0]["provider"], "headroom");
    assert_eq!(stats["recent_requests"][0]["model"], "gpt-4o");
    assert!(stats["recent_requests"][0]
        .as_object()
        .unwrap()
        .get("request_messages")
        .is_none());

    let empty = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({"model": "gpt-4o", "messages": []}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(empty["tokens_before"], 0);
    assert_eq!(empty["tokens_after"], 0);
    assert_eq!(empty["compression_ratio"], 1.0);
    assert!(empty
        .as_object()
        .unwrap()
        .get("transforms_summary")
        .is_none());
    assert!(empty.as_object().unwrap().get("route_counts").is_none());

    let ignored_system_override = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": big_content}],
            "config": {"protect_recent": 0, "compress_system_messages": false}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_ne!(
        ignored_system_override["messages"][0]["content"],
        serde_json::json!(big_content)
    );
    assert_router_strategy_marker(&ignored_system_override, "router:smart_crusher:");
    assert_transform_summary_matches_applied(&ignored_system_override);
    assert!(ignored_system_override
        .as_object()
        .unwrap()
        .get("route_counts")
        .is_none());

    let html_content = r#"<!DOCTYPE html>
<html>
  <head>
    <title>demo</title>
    <style>.hidden { display:none; }</style>
    <script>console.log("ignore me");</script>
  </head>
  <body>
    <header>Navigation chrome</header>
    <main>
      <article>
        <h1>Deployment readiness update</h1>
        <p>The enterprise rollout completed successfully across three regions.</p>
        <p>Latency stayed within budget and no incidents were reported.</p>
      </article>
    </main>
    <footer>copyright 2025</footer>
  </body>
</html>"#;
    let compressed_html = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": html_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&compressed_html, "router:html:");
    assert_transform_summary_matches_applied(&compressed_html);
    assert!(compressed_html
        .as_object()
        .unwrap()
        .get("route_counts")
        .is_none());
    let compressed_html_text = compressed_html["messages"][0]["content"].as_str().unwrap();
    assert!(compressed_html_text.contains("Deployment readiness update"));
    assert!(compressed_html_text.contains("enterprise rollout completed successfully"));
    assert!(!compressed_html_text.contains("console.log"));
    assert!(!compressed_html_text.contains("Navigation chrome"));

    let short_tool_message = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": "short tool output that should stay untouched"}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        short_tool_message["messages"][0]["content"],
        serde_json::json!("short tool output that should stay untouched")
    );
    assert_eq!(
        short_tool_message["transforms_applied"],
        serde_json::json!(["router:noop"])
    );
    assert_transform_summary_matches_applied(&short_tool_message);

    let ignored_min_tokens_override = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": big_content}],
            "config": {"protect_recent": 0, "min_tokens_to_compress": 100000}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&ignored_min_tokens_override, "router:smart_crusher:");
    assert_transform_summary_matches_applied(&ignored_min_tokens_override);
    assert!(ignored_min_tokens_override
        .as_object()
        .unwrap()
        .get("route_counts")
        .is_none());

    let recent_code = r#"fn sum(a: i32, b: i32) -> i32 {
    a + b
}

#[tokio::test]
async fn compress_route_rejects_oversized_request_body() {
    let upstream = MockServer::start().await;
    let mut config = Config::for_test(Url::parse(&upstream.uri()).unwrap());
    config.max_body_bytes = 32;
    let proxy = common::start_proxy_with_config(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": "this request body should exceed the configured max"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 413);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("request too large")
    );
    proxy.shutdown().await;
}

struct Accumulator {
    total: i32,
}

impl Accumulator {
    fn push(&mut self, value: i32) {
        self.total += value;
    }
}"#;
    let protected_recent_code = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": recent_code}],
            "config": {"protect_recent": 1}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        protected_recent_code["messages"][0]["content"],
        serde_json::json!(recent_code)
    );
    assert_eq!(
        protected_recent_code["transforms_applied"],
        serde_json::json!(["router:protected:recent_code"])
    );

    let protected_analysis_code = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "analyze this Rust code and explain how helper works"},
                {"role": "tool", "content": recent_code}
            ],
            "config": {"protect_recent": 0, "protect_analysis_context": true, "compress_user_messages": false}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        protected_analysis_code["messages"][1]["content"],
        serde_json::json!(recent_code)
    );
    assert_eq!(
        protected_analysis_code["transforms_applied"],
        serde_json::json!([
            "router:protected:user_message",
            "router:protected:analysis_context"
        ])
    );
    assert_transform_summary_matches_applied(&protected_analysis_code);

    let search_content = (1..=40)
        .map(|line| {
            format!(
                "src/lib.rs:{line}:{}",
                if line == 7 || line == 24 {
                    "error handler for request validation"
                } else {
                    "regular helper match"
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let search_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": search_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&search_compressed, "router:search:");
    assert_transform_summary_matches_applied(&search_compressed);
    assert!(search_compressed["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("Retrieve more: hash="));
    assert_eq!(search_compressed["ccr_hashes"].as_array().unwrap().len(), 1);

    let build_content = (1..=60)
        .map(|line| match line {
            10 => "ERROR failed to compile crate".to_string(),
            11 => "  --> src/main.rs:10:5".to_string(),
            12 => "warning: unused variable".to_string(),
            40 => "Traceback (most recent call last)".to_string(),
            41 => "  File \"main.py\", line 12, in <module>".to_string(),
            42 => "ValueError: broken".to_string(),
            55 => "Build failed after 3 errors".to_string(),
            _ => format!("INFO line {}", line),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let build_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": build_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&build_compressed, "router:log:");
    assert_transform_summary_matches_applied(&build_compressed);
    assert!(build_compressed["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("lines omitted"));

    let text_content = (1..=130)
        .map(|line| match line {
            1 => "# Incident summary".to_string(),
            27 => "Important: rollout blocked by proxy policy mismatch".to_string(),
            64 => "Action item: verify enterprise telemetry sink before retry".to_string(),
            129 => "Final note: readiness checks stayed green during rollback".to_string(),
            _ => format!("General narrative line {line} about operational context"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let text_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": text_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&text_compressed, "router:text:");
    assert_transform_summary_matches_applied(&text_compressed);
    assert!(text_compressed["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("Retrieve more: hash="));
    assert_eq!(text_compressed["ccr_hashes"].as_array().unwrap().len(), 1);

    let mixed_search = (1..=35)
        .map(|line| match line {
            7 | 21 => format!("src/lib.rs:{line}:critical error handler path"),
            _ => format!("src/lib.rs:{line}:regular helper match"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mixed_content = format!(
        "Incident summary for rollout validation. This section explains the operational context. Another sentence adds more narrative detail. More prose keeps the detector honest. Yet another sentence for the prose heuristic. Final sentence before matches.\n\n```rust\nfn important() {{\n    println!(\"keep this code fenced\");\n}}\n```\n\n{mixed_search}"
    );
    let mixed_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": mixed_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&mixed_compressed, "router:mixed:");
    assert_transform_summary_matches_applied(&mixed_compressed);
    let mixed_output = mixed_compressed["messages"][0]["content"].as_str().unwrap();
    assert!(mixed_output.contains("```rust"));
    assert!(mixed_output.contains("Retrieve more: hash="));

    let unfenced_mixed_content = format!(
        "Incident summary for rollout validation. This section explains the operational context. Another sentence adds more narrative detail. More prose keeps the detector honest. Yet another sentence for the prose heuristic. Final sentence before the code block.\n\nfn helper(value: i32) -> i32 {{\n    value + 1\n}}\nstruct Config {{\n    enabled: bool,\n}}\nimpl Config {{\n    fn new() -> Self {{\n        Self {{ enabled: true }}\n    }}\n}}\n\n{mixed_search}"
    );
    let unfenced_mixed_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": unfenced_mixed_content}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_router_strategy_marker(&unfenced_mixed_compressed, "router:mixed:");
    assert_transform_summary_matches_applied(&unfenced_mixed_compressed);
    let unfenced_mixed_output = unfenced_mixed_compressed["messages"][0]["content"]
        .as_str()
        .unwrap();
    assert!(unfenced_mixed_output.contains("Retrieve more: hash="));
    assert!(unfenced_mixed_output.contains("fn helper(value: i32) -> i32 {"));
    assert!(unfenced_mixed_output.contains("struct Config {"));

    let already_compressed =
        "[20 lines compressed to 5. Retrieve more: hash=1234567890abcdef12345678]";
    let pinned = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": already_compressed}],
            "config": {"protect_recent": 0}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(
        pinned["messages"][0]["content"],
        serde_json::json!(already_compressed)
    );
    assert_eq!(
        pinned["transforms_applied"],
        serde_json::json!(["router:noop"])
    );
    assert_transform_summary_matches_applied(&pinned);

    let diff_tail = (0..60)
        .map(|index| format!("+    println!(\"extra line {}\");", index))
        .collect::<Vec<_>>()
        .join("\n");
    let diff_content = format!(
        r#"diff --git a/src/main.rs b/src/main.rs
index 1111111..2222222 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,9 +1,69 @@
 use std::collections::HashMap;
 
 pub fn main() {{
-    println!("hello");
+    println!("hello world");
+    println!("more diagnostics");
 }}
 
 pub fn helper() {{
-    todo!()
+    let mut map = HashMap::new();
+    map.insert("a", 1);
+    map.insert("b", 2);
+    println!("{{:?}}", map);
+    println!("finished");
{diff_tail}
 }}
"#
    );
    let diff_compressed = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "summarize the diff"},
                {"role": "tool", "content": diff_content}
            ],
            "config": {"compress_user_messages": false, "protect_recent": 0, "target_ratio": 0.95}
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    let diff_transforms = diff_compressed["transforms_applied"].as_array().unwrap();
    assert_eq!(
        diff_transforms[0],
        serde_json::json!("router:protected:user_message")
    );
    assert!(diff_transforms[1]
        .as_str()
        .unwrap()
        .starts_with("router:diff_compressor:"));
    assert_transform_summary_matches_applied(&diff_compressed);
    assert!(
        diff_compressed["tokens_after"].as_u64().unwrap()
            < diff_compressed["tokens_before"].as_u64().unwrap()
    );
    assert!(diff_compressed["ccr_hashes"].is_array());

    proxy.shutdown().await;
}

#[tokio::test]
async fn compress_route_emits_metadata() {
    let upstream = MockServer::start().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let big_content = serde_json::to_string(&serde_json::Value::Array(
        (0..24)
            .map(|index| {
                serde_json::json!({
                    "id": index,
                    "status": if index % 4 == 0 { "warn" } else { "ok" },
                    "service": "catalog-api",
                    "duration_ms": 220 + index,
                    "message": format!("row {} with oversized JSON output", index),
                    "payload": {
                        "user_id": format!("user-{:03}", index),
                        "region": "us-west-2",
                        "feature_flag": "smart-crusher",
                    }
                })
            })
            .collect(),
    ))
    .unwrap();

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "tool", "content": big_content}],
            "config": {"protect_recent": 0}
        }))
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
                    == Some(&MetadataValue::String("compress_messages".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("endpoint")
                    == Some(&MetadataValue::String("compress".to_string()))
        })
        .expect("input routed metadata event");
    let compressed = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputCompressed
                && event.metadata.get("compression_status")
                    == Some(&MetadataValue::String("compressed".to_string()))
        })
        .expect("input compressed event");
    assert_eq!(routed_mode.context.operation, "proxy.compress_messages");
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
    assert_eq!(
        compressed.metadata.get("transform_count"),
        Some(&MetadataValue::String("1".to_string()))
    );
}

#[tokio::test]
async fn ccr_and_toin_routes_are_native() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let retrieve_missing = client
        .post(format!("{}/v1/retrieve", proxy.url()))
        .json(&serde_json::json!({"hash":"abc"}))
        .send()
        .await
        .unwrap();
    assert_eq!(retrieve_missing.status(), 404);

    let retrieve_invalid = client
        .post(format!("{}/v1/retrieve/tool_call", proxy.url()))
        .json(&serde_json::json!({
            "tool_call": {
                "id": "toolu_123",
                "name": "not_headroom_retrieve",
                "input": {"hash": "1234567890abcdef12345678"}
            },
            "provider": "anthropic"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(retrieve_invalid.status(), 400);

    let retrieve_tool_call = client
        .post(format!("{}/v1/retrieve/tool_call", proxy.url()))
        .json(&serde_json::json!({
            "tool_call": {
                "id": "toolu_123",
                "name": "headroom_retrieve",
                "input": {"hash": "1234567890abcdef12345678"}
            },
            "provider": "anthropic"
        }))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(retrieve_tool_call["success"], false);
    assert_eq!(
        retrieve_tool_call["tool_result"]["tool_use_id"],
        "toolu_123"
    );

    let stats = client
        .get(format!("{}/v1/retrieve/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(stats["store"]["entry_count"], 0);
    assert_eq!(stats["recent_retrievals"].as_array().unwrap().len(), 0);

    let feedback = client
        .get(format!("{}/v1/feedback", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feedback["feedback"]["tools_tracked"], 0);

    let feedback_tool = client
        .get(format!("{}/v1/feedback/tool-a", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(feedback_tool["tool_name"], "tool-a");
    assert_eq!(feedback_tool["hints"]["max_items"], 15);

    let toin_stats = client
        .get(format!("{}/v1/toin/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(toin_stats["enabled"], true);
    assert_eq!(toin_stats["patterns_tracked"], 0);

    let toin_patterns = client
        .get(format!("{}/v1/toin/patterns", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(toin_patterns, serde_json::json!([]));

    let toin_missing = client
        .get(format!("{}/v1/toin/pattern/abcd", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(toin_missing.status(), 404);

    proxy.shutdown().await;
}

#[tokio::test]
async fn telemetry_routes_are_native() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let stats = client
        .get(format!("{}/v1/telemetry", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(stats.status(), 200);
    assert_eq!(
        stats.json::<serde_json::Value>().await.unwrap()["enabled"],
        true
    );

    let import = client
        .post(format!("{}/v1/telemetry/import", proxy.url()))
        .json(&serde_json::json!({
            "summary": {
                "total_compressions": 4,
                "total_retrievals": 2,
                "total_tokens_saved": 120
            },
            "tool_stats": {
                "abc123": {
                    "sample_size": 7,
                    "avg_compression_ratio": 0.25,
                    "avg_token_reduction": 0.5
                }
            },
            "recommendations": {
                "abc123": {
                    "confidence": 0.9
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(import.status(), 200);
    assert_eq!(
        import.json::<serde_json::Value>().await.unwrap()["current_stats"]["total_tokens_saved"],
        120
    );

    let tools = client
        .get(format!("{}/v1/telemetry/tools", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(tools["tool_count"], 1);
    assert_eq!(tools["tools"]["abc123"]["sample_size"], 7);

    let detail = client
        .get(format!("{}/v1/telemetry/tools/abc123", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(detail["signature_hash"], "abc123");
    assert_eq!(detail["recommendations"]["confidence"], 0.9);

    let missing = client
        .get(format!("{}/v1/telemetry/tools/missing", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);

    let export = client
        .get(format!("{}/v1/telemetry/export", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(export["summary"]["total_compressions"], 4);
    assert_eq!(export["tool_stats"]["abc123"]["sample_size"], 7);

    proxy.shutdown().await;
}

#[tokio::test]
async fn compress_cache_can_be_cleared() {
    let upstream = MockServer::start().await;
    let proxy = common::start_proxy(&upstream.uri()).await;
    let client = reqwest::Client::new();

    let content = (1..=120)
        .map(|line| match line {
            20 => "Important: rollout blocked by proxy policy mismatch".to_string(),
            60 => "Action item: verify enterprise telemetry sink before retry".to_string(),
            _ => format!("General narrative line {line} about operational context"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let request = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "tool", "content": content}],
        "config": {"protect_recent": 0}
    });

    let first = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert!(!first["transforms_applied"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "router:cache_hit"));
    assert_transform_summary_matches_applied(&first);

    let second = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert!(second["transforms_applied"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| value != "router:cache_hit"));
    assert!(second.as_object().unwrap().get("route_counts").is_none());
    assert_transform_summary_matches_applied(&second);

    let stats = client
        .get(format!("{}/stats", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(stats["compression_cache"]["hits"], 1);
    assert!(stats["compression_cache"]["size"].as_u64().unwrap() >= 1);

    let cleared = client
        .post(format!("{}/cache/clear", proxy.url()))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(cleared["status"], "cache cleared");
    assert!(cleared["previous_size"].as_u64().unwrap() >= 1);

    let third = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert!(!third["transforms_applied"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "router:cache_hit"));
    assert_transform_summary_matches_applied(&third);

    proxy.shutdown().await;
}

#[tokio::test]
async fn compress_honors_explicit_token_budget_with_dropped_context_marker() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy_with_runtime(&upstream.uri(), Arc::new(PipelineDispatcher::new())).await;
    let client = reqwest::Client::new();
    let request = serde_json::json!({
        "model": "gpt-4o",
        "token_budget": 4000,
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "old context ".repeat(200)},
            {"role": "assistant", "content": "assistant reply ".repeat(200)},
            {"role": "user", "content": "middle question ".repeat(200)},
            {"role": "assistant", "content": "middle answer ".repeat(200)},
            {"role": "user", "content": "recent question ".repeat(200)},
            {"role": "assistant", "content": "recent answer ".repeat(200)}
        ],
        "config": {"protect_recent": 0}
    });

    let body = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();

    assert!(body["transforms_applied"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap_or_default().starts_with("window_cap:")));
    assert!(body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["content"]
            .as_str()
            .unwrap_or_default()
            .starts_with("<headroom:dropped_context reason=\"token_cap\"")));
    assert_transform_summary_matches_applied(&body);

    proxy.shutdown().await;
}

#[tokio::test]
async fn compress_applies_model_context_limit_without_explicit_token_budget() {
    let upstream = MockServer::start().await;
    let proxy = start_proxy_with_runtime(&upstream.uri(), Arc::new(PipelineDispatcher::new())).await;
    let client = reqwest::Client::new();
    let messages = (0..80)
        .flat_map(|index| {
            [
                serde_json::json!({
                    "role": "user",
                    "content": format!("unique request {index} {}", (0..40).map(|n| format!("token{index}_{n}")).collect::<Vec<_>>().join(" "))
                }),
                serde_json::json!({
                    "role": "assistant",
                    "content": format!("unique response {index} {}", (0..40).map(|n| format!("reply{index}_{n}")).collect::<Vec<_>>().join(" "))
                }),
            ]
        })
        .collect::<Vec<_>>();
    let request = serde_json::json!({
        "model": "gpt-4",
        "messages": messages,
        "config": {"protect_recent": 0}
    });

    let body = client
        .post(format!("{}/v1/compress", proxy.url()))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();

    assert!(body["transforms_applied"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str().unwrap_or_default().starts_with("window_cap:")));
    assert!(body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["content"]
            .as_str()
            .unwrap_or_default()
            .starts_with("<headroom:dropped_context reason=\"token_cap\"")));
    assert_transform_summary_matches_applied(&body);

    proxy.shutdown().await;
}

#[tokio::test]
async fn ccr_feedback_tool_emits_metadata() {
    let upstream = MockServer::start().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let resp = reqwest::Client::new()
        .get(format!("{}/v1/feedback/tool-a", proxy.url()))
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
                    == Some(&MetadataValue::String("ccr_feedback_tool".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("tool_name")
                    == Some(&MetadataValue::String("tool-a".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.ccr_feedback_tool");
    assert_eq!(
        routed_metadata.metadata.get("endpoint"),
        Some(&MetadataValue::String("feedback/tool".to_string()))
    );
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}

#[tokio::test]
async fn telemetry_tool_detail_emits_metadata() {
    let upstream = MockServer::start().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(PipelineDispatcher::new().with_plugin(RecordingPlugin {
        events: events.clone(),
    }));
    let proxy = start_proxy_with_runtime(&upstream.uri(), runtime).await;

    let client = reqwest::Client::new();
    let import = client
        .post(format!("{}/v1/telemetry/import", proxy.url()))
        .json(&serde_json::json!({
            "tool_stats": {
                "abc123": {
                    "sample_size": 3
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(import.status(), 200);

    let resp = client
        .get(format!("{}/v1/telemetry/tools/abc123", proxy.url()))
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
                    == Some(&MetadataValue::String("telemetry_tool_detail".to_string()))
        })
        .expect("input routed route_mode event");
    let routed_metadata = recorded
        .iter()
        .find(|event| {
            event.stage == PipelineStage::InputRouted
                && event.metadata.get("signature_hash")
                    == Some(&MetadataValue::String("abc123".to_string()))
        })
        .expect("input routed metadata event");
    assert_eq!(routed_mode.context.operation, "proxy.telemetry_tool_detail");
    assert_eq!(
        routed_metadata.metadata.get("endpoint"),
        Some(&MetadataValue::String("telemetry/tools/detail".to_string()))
    );
    assert_eq!(
        routed_metadata.metadata.get("native_route"),
        Some(&MetadataValue::Bool(true))
    );
}
