//! Cache prefix alignment for provider-side prefix caching.

use std::sync::Mutex;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheAlignerConfig {
    pub enabled: bool,
    pub use_dynamic_detector: bool,
    pub detection_tiers: Vec<String>,
    pub extra_dynamic_labels: Vec<String>,
    pub entropy_threshold: f64,
    pub date_patterns: Vec<String>,
    pub normalize_whitespace: bool,
    pub collapse_blank_lines: bool,
    pub dynamic_tail_separator: String,
}

impl Default for CacheAlignerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            use_dynamic_detector: true,
            detection_tiers: vec!["regex".to_string()],
            extra_dynamic_labels: Vec::new(),
            entropy_threshold: 0.7,
            date_patterns: Self::default_date_patterns(),
            normalize_whitespace: true,
            collapse_blank_lines: true,
            dynamic_tail_separator: "\n\n---\n[Dynamic Context]\n".to_string(),
        }
    }
}

impl CacheAlignerConfig {
    pub fn default_date_patterns() -> Vec<String> {
        vec![
            r"Current [Dd]ate:?\s*\d{4}-\d{2}-\d{2}".to_string(),
            r"Today is \w+,?\s+\w+ \d+".to_string(),
            r"Today's date:?\s*\d{4}-\d{2}-\d{2}".to_string(),
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}".to_string(),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachePrefixMetrics {
    pub stable_prefix_bytes: usize,
    pub stable_prefix_tokens_est: usize,
    pub stable_prefix_hash: String,
    pub prefix_changed: bool,
    pub previous_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheAlignerResult {
    pub messages: Vec<Value>,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub transforms_applied: Vec<String>,
    pub markers_inserted: Vec<String>,
    pub warnings: Vec<String>,
    pub diff_artifact: Option<Value>,
    pub cache_metrics: Option<CachePrefixMetrics>,
    pub timing: serde_json::Map<String, Value>,
    pub waste_signals: Option<Value>,
}

pub struct CacheAligner {
    config: CacheAlignerConfig,
    date_patterns: Vec<Regex>,
    previous_prefix_hash: Mutex<Option<String>>,
    detector_patterns: Vec<Regex>,
}

impl CacheAligner {
    pub fn new(config: CacheAlignerConfig) -> Self {
        let date_patterns = config
            .date_patterns
            .iter()
            .map(|pattern| Regex::new(pattern).expect("cache aligner date pattern compiles"))
            .collect();
        let detector_patterns = vec![
            Regex::new(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
                .expect("iso datetime regex compiles"),
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("iso date regex compiles"),
        ];
        Self {
            config,
            date_patterns,
            previous_prefix_hash: Mutex::new(None),
            detector_patterns,
        }
    }

    pub fn apply(&self, messages: &[Value], tokenizer: &dyn Tokenizer) -> CacheAlignerResult {
        let tokens_before = count_message_tokens(messages, tokenizer);
        let mut result_messages = messages.to_vec();
        let mut transforms_applied = Vec::new();
        let warnings = Vec::new();
        let mut extracted_dynamic = Vec::new();

        for message in &mut result_messages {
            if message.get("role").and_then(Value::as_str) != Some("system") {
                continue;
            }
            let Some(content) = message
                .get("content")
                .and_then(Value::as_str)
                .map(ToString::to_string)
            else {
                continue;
            };
            let (new_content, extracted) = self.extract_dynamic_content(&content);
            if !extracted.is_empty() {
                extracted_dynamic.extend(extracted);
                if let Some(object) = message.as_object_mut() {
                    object.insert("content".to_string(), Value::String(new_content));
                }
            }
        }

        if self.config.normalize_whitespace {
            for message in &mut result_messages {
                let Some(content) = message
                    .get("content")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                else {
                    continue;
                };
                if let Some(object) = message.as_object_mut() {
                    object.insert(
                        "content".to_string(),
                        Value::String(self.normalize_whitespace(&content)),
                    );
                }
            }
        }

        let stable_prefix_content = self.get_stable_prefix_content(&result_messages);
        let stable_hash = compute_short_hash(&stable_prefix_content);
        let previous_hash = {
            let mut guard = self
                .previous_prefix_hash
                .lock()
                .expect("cache aligner previous hash mutex not poisoned");
            let previous = guard.clone();
            *guard = Some(stable_hash.clone());
            previous
        };

        let cache_metrics = CachePrefixMetrics {
            stable_prefix_bytes: stable_prefix_content.len(),
            stable_prefix_tokens_est: tokenizer.count_text(&stable_prefix_content),
            prefix_changed: previous_hash
                .as_ref()
                .is_some_and(|value| value != &stable_hash),
            previous_hash,
            stable_prefix_hash: stable_hash.clone(),
        };

        if !extracted_dynamic.is_empty() {
            self.reinsert_dynamic_content(&mut result_messages, &extracted_dynamic);
            transforms_applied.push("cache_align".to_string());
        }

        let tokens_after = count_message_tokens(&result_messages, tokenizer);
        CacheAlignerResult {
            messages: result_messages,
            tokens_before,
            tokens_after,
            transforms_applied,
            markers_inserted: vec![format!("stable_prefix_hash:{stable_hash}")],
            warnings,
            diff_artifact: None,
            cache_metrics: Some(cache_metrics),
            timing: serde_json::Map::new(),
            waste_signals: None,
        }
    }

    fn extract_dynamic_content(&self, content: &str) -> (String, Vec<String>) {
        let patterns = if self.config.use_dynamic_detector {
            &self.detector_patterns
        } else {
            &self.date_patterns
        };
        let mut extracted = Vec::new();
        let mut result = content.to_string();

        for pattern in patterns {
            extracted.extend(pattern.find_iter(&result).map(|m| m.as_str().to_string()));
            result = pattern.replace_all(&result, "").into_owned();
        }

        if extracted.is_empty() {
            (content.to_string(), extracted)
        } else {
            (self.cleanup_empty_lines(&result), extracted)
        }
    }

    fn normalize_whitespace(&self, content: &str) -> String {
        let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
        let mut lines = normalized
            .split('\n')
            .map(|line| line.trim_end().to_string())
            .collect::<Vec<_>>();

        if self.config.collapse_blank_lines {
            let mut collapsed = Vec::with_capacity(lines.len());
            let mut previous_blank = false;
            for line in lines {
                let is_blank = line.trim().is_empty();
                if is_blank && previous_blank {
                    continue;
                }
                previous_blank = is_blank;
                collapsed.push(line);
            }
            lines = collapsed;
        }

        lines.join("\n")
    }

    fn cleanup_empty_lines(&self, content: &str) -> String {
        let mut collapsed = Vec::new();
        let mut previous_empty = false;
        for line in content.split('\n') {
            let is_empty = line.trim().is_empty();
            if is_empty && previous_empty {
                continue;
            }
            previous_empty = is_empty;
            collapsed.push(line);
        }
        collapsed.join("\n").trim().to_string()
    }

    fn reinsert_dynamic_content(&self, messages: &mut [Value], dynamic_values: &[String]) {
        if dynamic_values.is_empty() {
            return;
        }
        let dynamic_note = if dynamic_values.len() <= 3 {
            dynamic_values.join(", ")
        } else {
            dynamic_values
                .iter()
                .map(|value| format!("- {value}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        for message in messages.iter_mut().rev() {
            if message.get("role").and_then(Value::as_str) != Some("system") {
                continue;
            }
            let Some(content) = message
                .get("content")
                .and_then(Value::as_str)
                .map(ToString::to_string)
            else {
                continue;
            };
            if let Some(object) = message.as_object_mut() {
                object.insert(
                    "content".to_string(),
                    Value::String(format!(
                        "{}{}{}",
                        content.trim(),
                        self.config.dynamic_tail_separator,
                        dynamic_note
                    )),
                );
            }
            break;
        }
    }

    fn get_stable_prefix_content(&self, messages: &[Value]) -> String {
        let mut prefix_parts = Vec::new();
        for message in messages {
            if message.get("role").and_then(Value::as_str) != Some("system") {
                break;
            }
            let Some(content) = message.get("content").and_then(Value::as_str) else {
                continue;
            };
            let content = content
                .split_once(&self.config.dynamic_tail_separator)
                .map(|(prefix, _)| prefix)
                .unwrap_or(content);
            prefix_parts.push(content.trim().to_string());
        }
        prefix_parts.join("\n---\n")
    }
}

fn compute_short_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

fn count_message_tokens(messages: &[Value], tokenizer: &dyn Tokenizer) -> usize {
    let mut total = 3;
    for message in messages {
        let Some(object) = message.as_object() else {
            continue;
        };
        total += 4;
        total += object
            .get("role")
            .and_then(Value::as_str)
            .map(|role| tokenizer.count_text(role))
            .unwrap_or(0);
        if let Some(content) = object.get("content") {
            total += count_content_tokens(content, tokenizer);
        }
        if let Some(name) = object.get("name").and_then(Value::as_str) {
            total += tokenizer.count_text(name) + 1;
        }
        if let Some(function_call) = object.get("function_call") {
            total += tokenizer.count_text(&function_call.to_string());
        }
        if let Some(tool_calls) = object.get("tool_calls") {
            total += tokenizer.count_text(&tool_calls.to_string());
        }
    }
    total
}

fn count_content_tokens(content: &Value, tokenizer: &dyn Tokenizer) -> usize {
    match content {
        Value::String(text) => tokenizer.count_text(text),
        Value::Array(parts) => parts
            .iter()
            .map(|part| count_content_part_tokens(part, tokenizer))
            .sum(),
        other => tokenizer.count_text(&other.to_string()),
    }
}

fn count_content_part_tokens(part: &Value, tokenizer: &dyn Tokenizer) -> usize {
    let Some(object) = part.as_object() else {
        return tokenizer.count_text(&part.to_string());
    };
    if object.get("type").and_then(Value::as_str) == Some("text") {
        return object
            .get("text")
            .and_then(Value::as_str)
            .map(|text| tokenizer.count_text(text))
            .unwrap_or(0);
    }
    if let Some(text) = object.get("text").and_then(Value::as_str) {
        return tokenizer.count_text(text);
    }
    tokenizer.count_text(&part.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::EstimatingCounter;
    use serde_json::json;

    #[test]
    fn extracts_iso_date_and_reinserts_dynamic_context() {
        let aligner = CacheAligner::new(CacheAlignerConfig::default());
        let result = aligner.apply(
            &[
                json!({"role": "system", "content": "You are a helpful assistant. The date is 2026-04-23. Request id 0."}),
                json!({"role": "user", "content": "Question number 0: what is 2+2?"}),
            ],
            &EstimatingCounter::default(),
        );

        assert_eq!(result.transforms_applied, vec!["cache_align"]);
        assert_eq!(
            result.messages[0]["content"],
            "You are a helpful assistant. The date is . Request id 0.\n\n---\n[Dynamic Context]\n2026-04-23"
        );
        assert_eq!(
            result.markers_inserted,
            vec!["stable_prefix_hash:a6f9a6e9b16b87f8"]
        );
    }

    #[test]
    fn tracks_previous_hash_across_requests() {
        let aligner = CacheAligner::new(CacheAlignerConfig::default());
        let tokenizer = EstimatingCounter::default();
        let first = aligner.apply(
            &[json!({"role": "system", "content": "You are a helpful assistant. The date is 2026-04-23. Request id 0."})],
            &tokenizer,
        );
        let second = aligner.apply(
            &[json!({"role": "system", "content": "You are a helpful assistant. The date is 2026-04-23. Request id 1."})],
            &tokenizer,
        );

        assert_eq!(first.cache_metrics.unwrap().previous_hash, None);
        let second_metrics = second.cache_metrics.unwrap();
        assert_eq!(
            second_metrics.previous_hash.as_deref(),
            Some("a6f9a6e9b16b87f8")
        );
        assert!(second_metrics.prefix_changed);
    }
}
