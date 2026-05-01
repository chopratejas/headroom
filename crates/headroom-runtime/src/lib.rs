//! Fixed execution lifecycle and plugin contracts for the Rust rewrite.
//!
//! This crate intentionally stays `std`-only so every other Headroom Rust crate
//! can share the same orchestration surface without inheriting framework,
//! telemetry, or serialization dependencies.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineStage {
    Setup,
    PreStart,
    PostStart,
    InputReceived,
    InputCached,
    InputRouted,
    InputCompressed,
    InputRemembered,
    PreSend,
    PostSend,
    ResponseReceived,
}

impl PipelineStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::PreStart => "pre_start",
            Self::PostStart => "post_start",
            Self::InputReceived => "input_received",
            Self::InputCached => "input_cached",
            Self::InputRouted => "input_routed",
            Self::InputCompressed => "input_compressed",
            Self::InputRemembered => "input_remembered",
            Self::PreSend => "pre_send",
            Self::PostSend => "post_send",
            Self::ResponseReceived => "response_received",
        }
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

pub const CANONICAL_PIPELINE_STAGES: [PipelineStage; 11] = [
    PipelineStage::Setup,
    PipelineStage::PreStart,
    PipelineStage::PostStart,
    PipelineStage::InputReceived,
    PipelineStage::InputCached,
    PipelineStage::InputRouted,
    PipelineStage::InputCompressed,
    PipelineStage::InputRemembered,
    PipelineStage::PreSend,
    PipelineStage::PostSend,
    PipelineStage::ResponseReceived,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    String(String),
}

impl From<bool> for MetadataValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for MetadataValue {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<u64> for MetadataValue {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<usize> for MetadataValue {
    fn from(value: usize) -> Self {
        Self::U64(value as u64)
    }
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineMetadata {
    entries: BTreeMap<String, MetadataValue>,
}

impl PipelineMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Option<MetadataValue> {
        self.entries.insert(key.into(), value.into())
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }
}

impl<K, V> FromIterator<(K, V)> for PipelineMetadata
where
    K: Into<String>,
    V: Into<MetadataValue>,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut metadata = Self::new();
        for (key, value) in iter {
            metadata.insert(key, value);
        }
        metadata
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionContext {
    pub component: String,
    pub operation: String,
    pub request_id: String,
    pub provider: String,
    pub model: String,
}

impl ExecutionContext {
    pub fn new(operation: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            request_id: request_id.into(),
            ..Self::default()
        }
    }

    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = component.into();
        self
    }

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineEvent {
    pub stage: PipelineStage,
    pub context: ExecutionContext,
    pub metadata: PipelineMetadata,
}

impl PipelineEvent {
    pub fn new(
        stage: PipelineStage,
        context: ExecutionContext,
        metadata: PipelineMetadata,
    ) -> Self {
        Self {
            stage,
            context,
            metadata,
        }
    }
}

pub trait PipelinePlugin: Send + Sync {
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    fn on_event(&self, event: &mut PipelineEvent);
}

#[derive(Default)]
pub struct PipelineDispatcher {
    plugins: Vec<Box<dyn PipelinePlugin>>,
}

impl PipelineDispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_plugin(mut self, plugin: impl PipelinePlugin + 'static) -> Self {
        self.push_plugin(plugin);
        self
    }

    pub fn push_plugin(&mut self, plugin: impl PipelinePlugin + 'static) {
        self.plugins.push(Box::new(plugin));
    }

    pub fn is_enabled(&self) -> bool {
        !self.plugins.is_empty()
    }

    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    pub fn emit(
        &self,
        stage: PipelineStage,
        context: ExecutionContext,
        metadata: PipelineMetadata,
    ) -> PipelineEvent {
        let mut event = PipelineEvent::new(stage, context, metadata);
        for plugin in &self.plugins {
            plugin.on_event(&mut event);
        }
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct AppendStageName;

    impl PipelinePlugin for AppendStageName {
        fn on_event(&self, event: &mut PipelineEvent) {
            event.metadata.insert("stage_name", event.stage.to_string());
        }
    }

    struct RecordStages {
        seen: Arc<Mutex<Vec<PipelineStage>>>,
    }

    impl PipelinePlugin for RecordStages {
        fn on_event(&self, event: &mut PipelineEvent) {
            self.seen.lock().expect("poisoned").push(event.stage);
        }
    }

    #[test]
    fn canonical_pipeline_stages_match_python_contract_order() {
        assert_eq!(
            CANONICAL_PIPELINE_STAGES,
            [
                PipelineStage::Setup,
                PipelineStage::PreStart,
                PipelineStage::PostStart,
                PipelineStage::InputReceived,
                PipelineStage::InputCached,
                PipelineStage::InputRouted,
                PipelineStage::InputCompressed,
                PipelineStage::InputRemembered,
                PipelineStage::PreSend,
                PipelineStage::PostSend,
                PipelineStage::ResponseReceived,
            ]
        );
    }

    #[test]
    fn dispatcher_allows_plugins_to_enrich_metadata() {
        let dispatcher = PipelineDispatcher::new().with_plugin(AppendStageName);
        let context = ExecutionContext::new("proxy.forward_http", "req-1")
            .with_component("headroom-proxy")
            .with_provider("openai");
        let event = dispatcher.emit(PipelineStage::PreSend, context, PipelineMetadata::new());

        assert_eq!(
            event.metadata.get("stage_name"),
            Some(&MetadataValue::String("pre_send".to_string()))
        );
    }

    #[test]
    fn dispatcher_runs_plugins_in_registration_order() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let dispatcher = PipelineDispatcher::new()
            .with_plugin(RecordStages { seen: seen.clone() })
            .with_plugin(RecordStages { seen: seen.clone() });
        let context = ExecutionContext::new("proxy.forward_http", "req-2");

        dispatcher.emit(
            PipelineStage::InputReceived,
            context,
            PipelineMetadata::new(),
        );

        assert_eq!(
            *seen.lock().expect("poisoned"),
            vec![PipelineStage::InputReceived, PipelineStage::InputReceived]
        );
    }
}
