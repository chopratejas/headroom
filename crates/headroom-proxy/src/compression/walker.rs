//! Per-blob compression: detect content type, run the
//! `CompressionPipeline`, return the result.
//!
//! All three provider arms call this on the **text content of a tool
//! output** in the new turn. Nothing else gets compressed (user
//! input, assistant text, system prompt, binary content, prefix
//! messages all bypass this entirely).

use std::sync::Arc;

use headroom_core::ccr::CcrStore;
use headroom_core::transforms::content_detector::detect_content_type;
use headroom_core::transforms::pipeline::{CompressionContext, CompressionPipeline};

/// Minimum blob length we even bother sending through the pipeline.
/// Below this, the orchestrator overhead (content detection,
/// reformat scan, bloat estimators) almost certainly dwarfs the
/// savings. Aligns with the Python `ContentRouter.min_tokens`
/// default of ~50 tokens × ~4 chars/token.
pub const MIN_BLOB_LEN: usize = 200;

/// Try to compress a single text blob. Returns:
///
/// - `Some(compressed)` when the pipeline produced a strictly smaller
///   output. The caller should swap this in.
/// - `None` when there's nothing to gain (blob too small, pipeline
///   skipped everything, or output >= input).
///
/// Never panics. Pipeline-internal errors degrade to "no change."
pub fn compress_blob(
    pipeline: &CompressionPipeline,
    store: &Arc<dyn CcrStore>,
    blob: &str,
) -> Option<String> {
    if blob.len() < MIN_BLOB_LEN {
        return None;
    }
    let detection = detect_content_type(blob);
    let ctx = CompressionContext::with_query(String::new());
    let result = pipeline.run(blob, detection.content_type, &ctx, store.as_ref());
    if result.bytes_saved == 0 || result.output.len() >= blob.len() {
        return None;
    }
    Some(result.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::pipeline::build_pipeline;

    #[test]
    fn small_blob_skipped() {
        let (pipeline, store) = build_pipeline();
        let out = compress_blob(&pipeline, &store, "tiny");
        assert!(out.is_none());
    }

    #[test]
    fn json_array_compresses() {
        let (pipeline, store) = build_pipeline();
        let row = r#"{"id":1,"name":"alice","email":"a@example.com","tags":["x","y","z"]}"#;
        let blob = format!("[{}]", vec![row; 1000].join(","));
        let out = compress_blob(&pipeline, &store, &blob);
        assert!(out.is_some());
        let compressed = out.unwrap();
        eprintln!(
            "blob len={} compressed len={} preview={:?}",
            blob.len(),
            compressed.len(),
            &compressed[..compressed.len().min(400)]
        );
        assert!(compressed.len() < blob.len());
    }

    #[test]
    fn plain_text_under_min_skipped() {
        let (pipeline, store) = build_pipeline();
        // Just under MIN_BLOB_LEN — pipeline shouldn't even be invoked.
        let s = "x".repeat(MIN_BLOB_LEN - 1);
        assert!(compress_blob(&pipeline, &store, &s).is_none());
    }
}
