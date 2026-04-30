//! Minimal Prometheus-style metrics for the Rust passthrough proxy.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::proxy::AppState;

#[derive(Default)]
pub struct ProxyMetrics {
    requests_total: AtomicU64,
    requests_in_flight: AtomicU64,
    request_errors_total: AtomicU64,
    request_duration_millis_total: AtomicU64,
    request_duration_count: AtomicU64,
    upstream_2xx_total: AtomicU64,
    upstream_4xx_total: AtomicU64,
    upstream_5xx_total: AtomicU64,
    openai_shadow_comparisons_total: AtomicU64,
    openai_shadow_matches_total: AtomicU64,
    openai_shadow_mismatches_total: AtomicU64,
    openai_shadow_skipped_total: AtomicU64,
    anthropic_shadow_comparisons_total: AtomicU64,
    anthropic_shadow_matches_total: AtomicU64,
    anthropic_shadow_mismatches_total: AtomicU64,
    anthropic_shadow_skipped_total: AtomicU64,
    gemini_shadow_comparisons_total: AtomicU64,
    gemini_shadow_matches_total: AtomicU64,
    gemini_shadow_mismatches_total: AtomicU64,
    gemini_shadow_skipped_total: AtomicU64,
    gemini_count_tokens_shadow_comparisons_total: AtomicU64,
    gemini_count_tokens_shadow_matches_total: AtomicU64,
    gemini_count_tokens_shadow_mismatches_total: AtomicU64,
    google_cloudcode_stream_shadow_comparisons_total: AtomicU64,
    google_cloudcode_stream_shadow_matches_total: AtomicU64,
    google_cloudcode_stream_shadow_mismatches_total: AtomicU64,
    response_cache_hits_total: AtomicU64,
    response_cache_misses_total: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyMetricsSnapshot {
    pub requests_total: u64,
    pub requests_in_flight: u64,
    pub request_errors_total: u64,
    pub request_duration_millis_total: u64,
    pub request_duration_count: u64,
    pub upstream_2xx_total: u64,
    pub upstream_4xx_total: u64,
    pub upstream_5xx_total: u64,
}

impl ProxyMetrics {
    pub fn record_request_started(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_completed(&self, status: u16, duration_seconds: f64) {
        self.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        self.request_duration_count.fetch_add(1, Ordering::Relaxed);
        self.request_duration_millis_total
            .fetch_add(duration_millis(duration_seconds), Ordering::Relaxed);
        match status / 100 {
            2 => {
                self.upstream_2xx_total.fetch_add(1, Ordering::Relaxed);
            }
            4 => {
                self.upstream_4xx_total.fetch_add(1, Ordering::Relaxed);
            }
            5 => {
                self.upstream_5xx_total.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    pub fn record_request_failed(&self, duration_seconds: f64) {
        self.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
        self.request_errors_total.fetch_add(1, Ordering::Relaxed);
        self.request_duration_count.fetch_add(1, Ordering::Relaxed);
        self.request_duration_millis_total
            .fetch_add(duration_millis(duration_seconds), Ordering::Relaxed);
    }

    pub fn record_openai_shadow_comparison(&self, matched: bool) {
        self.openai_shadow_comparisons_total
            .fetch_add(1, Ordering::Relaxed);
        if matched {
            self.openai_shadow_matches_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.openai_shadow_mismatches_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_openai_shadow_skipped(&self) {
        self.openai_shadow_skipped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_anthropic_shadow_comparison(&self, matched: bool) {
        self.anthropic_shadow_comparisons_total
            .fetch_add(1, Ordering::Relaxed);
        if matched {
            self.anthropic_shadow_matches_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.anthropic_shadow_mismatches_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_anthropic_shadow_skipped(&self) {
        self.anthropic_shadow_skipped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_gemini_shadow_comparison(&self, matched: bool) {
        self.gemini_shadow_comparisons_total
            .fetch_add(1, Ordering::Relaxed);
        if matched {
            self.gemini_shadow_matches_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.gemini_shadow_mismatches_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_gemini_shadow_skipped(&self) {
        self.gemini_shadow_skipped_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_gemini_count_tokens_shadow_comparison(&self, matched: bool) {
        self.gemini_count_tokens_shadow_comparisons_total
            .fetch_add(1, Ordering::Relaxed);
        if matched {
            self.gemini_count_tokens_shadow_matches_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.gemini_count_tokens_shadow_mismatches_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_google_cloudcode_stream_shadow_comparison(&self, matched: bool) {
        self.google_cloudcode_stream_shadow_comparisons_total
            .fetch_add(1, Ordering::Relaxed);
        if matched {
            self.google_cloudcode_stream_shadow_matches_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.google_cloudcode_stream_shadow_mismatches_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_response_cache_hit(&self) {
        self.response_cache_hits_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_response_cache_miss(&self) {
        self.response_cache_misses_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ProxyMetricsSnapshot {
        ProxyMetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_in_flight: self.requests_in_flight.load(Ordering::Relaxed),
            request_errors_total: self.request_errors_total.load(Ordering::Relaxed),
            request_duration_millis_total: self
                .request_duration_millis_total
                .load(Ordering::Relaxed),
            request_duration_count: self.request_duration_count.load(Ordering::Relaxed),
            upstream_2xx_total: self.upstream_2xx_total.load(Ordering::Relaxed),
            upstream_4xx_total: self.upstream_4xx_total.load(Ordering::Relaxed),
            upstream_5xx_total: self.upstream_5xx_total.load(Ordering::Relaxed),
        }
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        let requests_total = self.requests_total.load(Ordering::Relaxed);
        let requests_in_flight = self.requests_in_flight.load(Ordering::Relaxed);
        let request_errors_total = self.request_errors_total.load(Ordering::Relaxed);
        let request_duration_count = self.request_duration_count.load(Ordering::Relaxed);
        let request_duration_seconds_sum =
            self.request_duration_millis_total.load(Ordering::Relaxed) as f64 / 1000.0;

        push_metric_help(
            &mut out,
            "headroom_requests_total",
            "counter",
            "Total proxied HTTP requests handled by the Rust proxy.",
        );
        let _ = writeln!(out, "headroom_requests_total {requests_total}");

        push_metric_help(
            &mut out,
            "headroom_requests_in_flight",
            "gauge",
            "Current number of proxied HTTP requests in flight.",
        );
        let _ = writeln!(out, "headroom_requests_in_flight {requests_in_flight}");

        push_metric_help(
            &mut out,
            "headroom_errors_total",
            "counter",
            "Total proxied HTTP requests that failed before an upstream response.",
        );
        let _ = writeln!(out, "headroom_errors_total {request_errors_total}");

        push_metric_help(
            &mut out,
            "headroom_request_duration_seconds",
            "histogram",
            "Aggregate request duration for proxied HTTP requests.",
        );
        let _ = writeln!(
            out,
            "headroom_request_duration_seconds_sum {:.6}",
            request_duration_seconds_sum
        );
        let _ = writeln!(
            out,
            "headroom_request_duration_seconds_count {request_duration_count}"
        );

        push_metric_help(
            &mut out,
            "headroom_upstream_responses_total",
            "counter",
            "Total upstream responses observed by status class.",
        );
        let _ = writeln!(
            out,
            "headroom_upstream_responses_total{{status_class=\"2xx\"}} {}",
            self.upstream_2xx_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "headroom_upstream_responses_total{{status_class=\"4xx\"}} {}",
            self.upstream_4xx_total.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "headroom_upstream_responses_total{{status_class=\"5xx\"}} {}",
            self.upstream_5xx_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_openai_shadow_comparisons_total",
            "counter",
            "Total OpenAI native-route shadow comparisons executed.",
        );
        let _ = writeln!(
            out,
            "headroom_openai_shadow_comparisons_total {}",
            self.openai_shadow_comparisons_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_openai_shadow_matches_total",
            "counter",
            "Total OpenAI native-route shadow comparisons that matched the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_openai_shadow_matches_total {}",
            self.openai_shadow_matches_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_openai_shadow_mismatches_total",
            "counter",
            "Total OpenAI native-route shadow comparisons that diverged from the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_openai_shadow_mismatches_total {}",
            self.openai_shadow_mismatches_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_openai_shadow_skipped_total",
            "counter",
            "Total OpenAI native-route requests where shadow comparison was skipped.",
        );
        let _ = writeln!(
            out,
            "headroom_openai_shadow_skipped_total {}",
            self.openai_shadow_skipped_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_anthropic_shadow_comparisons_total",
            "counter",
            "Total Anthropic native-route shadow comparisons executed.",
        );
        let _ = writeln!(
            out,
            "headroom_anthropic_shadow_comparisons_total {}",
            self.anthropic_shadow_comparisons_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_anthropic_shadow_matches_total",
            "counter",
            "Total Anthropic native-route shadow comparisons that matched the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_anthropic_shadow_matches_total {}",
            self.anthropic_shadow_matches_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_anthropic_shadow_mismatches_total",
            "counter",
            "Total Anthropic native-route shadow comparisons that diverged from the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_anthropic_shadow_mismatches_total {}",
            self.anthropic_shadow_mismatches_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_anthropic_shadow_skipped_total",
            "counter",
            "Total Anthropic native-route requests where shadow comparison was skipped.",
        );
        let _ = writeln!(
            out,
            "headroom_anthropic_shadow_skipped_total {}",
            self.anthropic_shadow_skipped_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_shadow_comparisons_total",
            "counter",
            "Total Gemini native-route shadow comparisons executed.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_shadow_comparisons_total {}",
            self.gemini_shadow_comparisons_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_shadow_matches_total",
            "counter",
            "Total Gemini native-route shadow comparisons that matched the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_shadow_matches_total {}",
            self.gemini_shadow_matches_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_shadow_mismatches_total",
            "counter",
            "Total Gemini native-route shadow comparisons that diverged from the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_shadow_mismatches_total {}",
            self.gemini_shadow_mismatches_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_shadow_skipped_total",
            "counter",
            "Total Gemini native-route requests where shadow comparison was skipped.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_shadow_skipped_total {}",
            self.gemini_shadow_skipped_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_count_tokens_shadow_comparisons_total",
            "counter",
            "Total Gemini countTokens native-route shadow comparisons executed.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_count_tokens_shadow_comparisons_total {}",
            self.gemini_count_tokens_shadow_comparisons_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_count_tokens_shadow_matches_total",
            "counter",
            "Total Gemini countTokens native-route shadow comparisons that matched the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_count_tokens_shadow_matches_total {}",
            self.gemini_count_tokens_shadow_matches_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_gemini_count_tokens_shadow_mismatches_total",
            "counter",
            "Total Gemini countTokens native-route shadow comparisons that diverged from the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_gemini_count_tokens_shadow_mismatches_total {}",
            self.gemini_count_tokens_shadow_mismatches_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_google_cloudcode_stream_shadow_comparisons_total",
            "counter",
            "Total Google Cloud Code streaming alias shadow comparisons executed.",
        );
        let _ = writeln!(
            out,
            "headroom_google_cloudcode_stream_shadow_comparisons_total {}",
            self.google_cloudcode_stream_shadow_comparisons_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_google_cloudcode_stream_shadow_matches_total",
            "counter",
            "Total Google Cloud Code streaming alias shadow comparisons that matched the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_google_cloudcode_stream_shadow_matches_total {}",
            self.google_cloudcode_stream_shadow_matches_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_google_cloudcode_stream_shadow_mismatches_total",
            "counter",
            "Total Google Cloud Code streaming alias shadow comparisons that diverged from the passthrough response.",
        );
        let _ = writeln!(
            out,
            "headroom_google_cloudcode_stream_shadow_mismatches_total {}",
            self.google_cloudcode_stream_shadow_mismatches_total
                .load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_response_cache_hits_total",
            "counter",
            "Total native response-cache hits for buffered provider routes.",
        );
        let _ = writeln!(
            out,
            "headroom_response_cache_hits_total {}",
            self.response_cache_hits_total.load(Ordering::Relaxed)
        );

        push_metric_help(
            &mut out,
            "headroom_response_cache_misses_total",
            "counter",
            "Total native response-cache misses for buffered provider routes.",
        );
        let _ = writeln!(
            out,
            "headroom_response_cache_misses_total {}",
            self.response_cache_misses_total.load(Ordering::Relaxed)
        );

        out
    }
}

pub async fn metrics(State(state): State<AppState>) -> Response {
    let mut response = (StatusCode::OK, state.metrics.render_prometheus()).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

fn push_metric_help(target: &mut String, name: &str, metric_type: &str, help: &str) {
    let _ = writeln!(target, "# HELP {name} {help}");
    let _ = writeln!(target, "# TYPE {name} {metric_type}");
}

fn duration_millis(duration_seconds: f64) -> u64 {
    (duration_seconds.max(0.0) * 1000.0).round() as u64
}
