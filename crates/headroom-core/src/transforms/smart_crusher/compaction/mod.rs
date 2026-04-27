//! Compaction subsystem — Stage 3c.2 PR2.
//!
//! Lossless-first compaction of JSON arrays. Pipeline:
//!
//! ```text
//! input array
//!    ↓
//! [TabularCompactor] → Compaction IR (recursive tree)
//!    ↓
//! [Formatter trait] → bytes
//! ```
//!
//! The IR ([`ir::Compaction`]) is a recursive tree so we can express
//! multi-level compression: nested-uniform objects flatten into dotted
//! columns, stringified-JSON cells become sub-tables, opaque blobs
//! become CCR pointers, heterogeneous arrays partition into buckets.
//!
//! Formatters consume the IR. [`JsonFormatter`] keeps byte-equal parity
//! with today's SmartCrusher output. [`CsvSchemaFormatter`] emits a
//! token-efficient `[N]{cols}:` declaration + JSON schema header + CSV
//! rows that LLMs read reliably.
//!
//! [`JsonFormatter`]: format_json::JsonFormatter
//! [`CsvSchemaFormatter`]: format_csv_schema::CsvSchemaFormatter

pub mod classifier;
pub mod compactor;
pub mod formatter;
pub mod ir;

pub use classifier::{classify_cell, CellClass, ClassifyConfig};
pub use compactor::{compact, CompactConfig};
pub use formatter::{CsvSchemaFormatter, Formatter, JsonFormatter};
pub use ir::{Bucket, CellValue, Compaction, FieldSpec, OpaqueKind, Row, Schema};

/// Composed compaction stage: a config + formatter pair.
///
/// Plug into [`SmartCrusher`] via the builder's `with_compaction(...)`.
/// When configured, `crush_array` runs compaction as an opt-in
/// lossless-first stage; when absent (default), behavior is byte-equal
/// with today's lossy-only path.
///
/// [`SmartCrusher`]: super::SmartCrusher
pub struct CompactionStage {
    pub config: CompactConfig,
    pub formatter: Box<dyn Formatter>,
}

impl CompactionStage {
    /// CSV+schema formatter, default config — the recommended OSS preset.
    pub fn default_csv_schema() -> Self {
        Self {
            config: CompactConfig::default(),
            formatter: Box::new(CsvSchemaFormatter::new()),
        }
    }

    /// JSON formatter, default config — useful for debugging or for
    /// downstream consumers that want structured rather than CSV-shaped
    /// output.
    pub fn default_json() -> Self {
        Self {
            config: CompactConfig::default(),
            formatter: Box::new(JsonFormatter::new()),
        }
    }

    /// Run the stage end-to-end: compact + format. Returns the
    /// [`Compaction`] tree (so callers can inspect kept/total row
    /// counts) alongside the rendered bytes.
    pub fn run(&self, items: &[serde_json::Value]) -> (Compaction, String) {
        let c = compact(items, &self.config);
        let rendered = self.formatter.format(&c);
        (c, rendered)
    }
}

impl std::fmt::Debug for CompactionStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionStage")
            .field("config", &self.config)
            .field("formatter", &self.formatter.name())
            .finish()
    }
}
