//! Header passthrough + hop-by-hop filtering + X-Forwarded-* injection.

mod common;

use common::start_proxy;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn custom_headers_pass_through_both_ways() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/h"))
        .respond_with(move |req: &wiremock::Request| {
            assert_eq!(req.headers.get("authorization").unwrap(), "Bearer foo");
            assert_eq!(req.headers.get("x-custom").unwrap(), "bar");
            // Hop-by-hop must be stripped from the upstream-side request.
            assert!(req.headers.get("transfer-encoding").is_none());
            // X-Forwarded-* should be injected.
            let xff = req
                .headers
                .get("x-forwarded-for")
                .unwrap()
                .to_str()
                .unwrap();
            assert!(xff.contains("127.0.0.1"));
            assert!(req.headers.get("x-forwarded-proto").is_some());
            assert!(req.headers.get("x-forwarded-host").is_some());
            ResponseTemplate::new(200)
                .insert_header("x-server-side", "ack")
                .insert_header("x-multi", "v1")
                .append_header("x-multi", "v2")
                // Hop-by-hop on response side must be stripped by the proxy.
                .insert_header("connection", "close")
                .set_body_string("done")
        })
        .mount(&upstream)
        .await;

    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/h", proxy.url()))
        .header("authorization", "Bearer foo")
        .header("x-custom", "bar")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers().get("x-server-side").unwrap(), "ack");
    assert!(
        resp.headers().get("connection").is_none(),
        "hop-by-hop must be stripped"
    );
    let multi: Vec<_> = resp
        .headers()
        .get_all("x-multi")
        .iter()
        .map(|v| v.to_str().unwrap().to_string())
        .collect();
    assert_eq!(multi, vec!["v1".to_string(), "v2".to_string()]);
    proxy.shutdown().await;
}

#[tokio::test]
async fn xff_appends_existing_value() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/xff"))
        .respond_with(move |req: &wiremock::Request| {
            let xff = req
                .headers
                .get("x-forwarded-for")
                .unwrap()
                .to_str()
                .unwrap();
            // existing 1.2.3.4 must be preserved + appended.
            assert!(
                xff.starts_with("1.2.3.4"),
                "expected appended xff, got: {xff}"
            );
            assert!(xff.contains("127.0.0.1"));
            ResponseTemplate::new(200)
        })
        .mount(&upstream)
        .await;
    let proxy = start_proxy(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/xff", proxy.url()))
        .header("x-forwarded-for", "1.2.3.4")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    proxy.shutdown().await;
}
