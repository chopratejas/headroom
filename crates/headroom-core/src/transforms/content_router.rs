//! Magika-primary content router — the layer one level above `content_detector`.
//!
//! `content_detector` is a parity-locked, regex-only port of the Python
//! detector. ML detection belongs HERE, mirroring the Python `ContentRouter`,
//! which classifies with Magika first and only falls back to regex for the
//! shapes Magika can't name.
//!
//! Routing:
//!   1. **Magika** (ML file-type classifier — the same chain the Python
//!      ContentRouter uses via the `detect_content_type` pyo3 binding) is
//!      trusted for the families it classifies reliably AND that don't overlap
//!      with our domain shapes: `JsonArray` (objects AND arrays — this fixes
//!      the regex tier's `starts_with('[')` blind spot that left JSON objects
//!      uncompressed), `GitDiff`, `Html`.
//!   2. When Magika returns `SourceCode` / `PlainText` (or fails): re-check the
//!      two DOMAIN shapes Magika has no label for — build logs and grep/search
//!      (`file:line:`) output (Magika tends to tag logs as source code). If
//!      neither matches, keep Magika's verdict (`SourceCode` / `PlainText` are
//!      no-op compressors).
//!
//! This is detection only; the live-zone dispatcher maps the returned
//! [`ContentType`] to a per-type compressor.

use super::content_detector::{detect_domain_content_type, ContentType};
use super::magika_detector::magika_detect;

/// Classify `content` into a [`ContentType`] for compressor dispatch.
///
/// Magika-primary with a log/search regex fallback — see module docs.
pub fn route_content_type(content: &str) -> ContentType {
    if content.trim().is_empty() {
        return ContentType::PlainText;
    }

    // Tier 1: Magika. On error, fall back to treating the content as text and
    // trying the domain heuristics — loud, never silent.
    let magika = match magika_detect(content) {
        Ok(ct) => ct,
        Err(e) => {
            tracing::warn!(
                event = "magika_detect_failed",
                error = %e,
                "magika failed in content_router; treating as text + trying domain heuristics"
            );
            ContentType::PlainText
        }
    };

    match magika {
        // Reliable and non-overlapping with logs/search → trust Magika directly.
        ContentType::JsonArray | ContentType::GitDiff | ContentType::Html => magika,
        // Magika said code/text (or failed). It can't name build logs or grep
        // output and tends to tag logs as code — re-check those two shapes;
        // otherwise keep Magika's verdict (SourceCode / PlainText → no-op).
        _ => detect_domain_content_type(content).unwrap_or(magika),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_object_routes_to_json_array() {
        // The fco shape: a JSON OBJECT with a nested array. Magika recognizes it
        // as JSON regardless of the leading `{` — the regex tier only saw `[`.
        let c = r#"{"status":"ok","count":3,"results":[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]}"#;
        assert_eq!(route_content_type(c), ContentType::JsonArray);
    }

    #[test]
    fn json_array_routes_to_json_array() {
        let c = r#"[{"id":1,"v":"x"},{"id":2,"v":"y"},{"id":3,"v":"z"}]"#;
        assert_eq!(route_content_type(c), ContentType::JsonArray);
    }

    #[test]
    fn build_log_routes_to_build_output_not_source_code() {
        // Magika tends to tag logs as SourceCode; the domain fallback must rescue
        // them to BuildOutput so LogCompressor runs.
        let c = "2026-06-04 10:00:01 INFO  starting build job\n\
                 2026-06-04 10:00:02 WARN  deprecated flag --foo\n\
                 2026-06-04 10:00:03 ERROR test failed: assertion mismatch\n\
                 2026-06-04 10:00:04 INFO  retrying shard 2\n\
                 2026-06-04 10:00:05 ERROR build aborted after 3 retries\n\
                 2026-06-04 10:00:06 INFO  cleanup complete\n";
        assert_eq!(route_content_type(c), ContentType::BuildOutput);
    }

    #[test]
    fn grep_output_routes_to_search_results() {
        let c = "src/main.rs:42:fn process() {\n\
                 src/util.rs:13:    return None;\n\
                 lib/x.rs:7:struct X;\n";
        assert_eq!(route_content_type(c), ContentType::SearchResults);
    }

    #[test]
    fn prose_routes_to_plain_text() {
        let c = "The quick brown fox jumps over the lazy dog. \
                 This is ordinary English prose with no particular structure to detect.";
        assert_eq!(route_content_type(c), ContentType::PlainText);
    }

    #[test]
    fn empty_routes_to_plain_text() {
        assert_eq!(route_content_type("   "), ContentType::PlainText);
    }
}
