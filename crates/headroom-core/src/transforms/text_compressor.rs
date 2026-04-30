//! Generic plain-text compressor.

use std::collections::HashSet;
use std::sync::LazyLock;

use md5::{Digest, Md5};
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextCompressorConfig {
    pub keep_first_lines: usize,
    pub keep_last_lines: usize,
    pub max_total_lines: usize,
    pub sample_every_n_lines: usize,
    pub anchor_keywords: Vec<String>,
    pub boost_pattern_lines: bool,
    pub enable_ccr: bool,
    pub min_lines_for_ccr: usize,
}

impl Default for TextCompressorConfig {
    fn default() -> Self {
        Self {
            keep_first_lines: 10,
            keep_last_lines: 10,
            max_total_lines: 50,
            sample_every_n_lines: 10,
            anchor_keywords: Vec::new(),
            boost_pattern_lines: true,
            enable_ccr: true,
            min_lines_for_ccr: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextCompressionResult {
    pub compressed: String,
    pub original: String,
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub compression_ratio: f64,
    pub cache_key: Option<String>,
}

pub struct TextCompressor {
    config: TextCompressorConfig,
}

static ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(error|exception|fail(?:ed|ure)?|fatal|critical|crash|panic)\b")
        .expect("error regex compiles")
});
static IMPORTANCE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(important|note|todo|fixme|hack|xxx|bug|fix)\b")
        .expect("importance regex compiles")
});
static HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^#+\s").expect("header regex compiles"));
static BOLD_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\*\*").expect("bold regex compiles"));
static QUOTE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^>\s").expect("quote regex compiles"));

impl TextCompressor {
    pub fn new(config: TextCompressorConfig) -> Self {
        Self { config }
    }

    pub fn compress(&self, content: &str, context: &str) -> TextCompressionResult {
        let lines = content.lines().map(str::to_string).collect::<Vec<_>>();
        if lines.len() <= self.config.max_total_lines {
            return TextCompressionResult {
                compressed: content.to_string(),
                original: content.to_string(),
                original_line_count: lines.len(),
                compressed_line_count: lines.len(),
                compression_ratio: 1.0,
                cache_key: None,
            };
        }

        let scored_lines = self.score_lines(&lines, context);
        let selected = self.select_lines(&scored_lines, &lines);
        let mut compressed = self.format_output(&selected, lines.len());
        let mut ratio = compressed.len() as f64 / content.len().max(1) as f64;
        let cache_key = if self.config.enable_ccr
            && lines.len() >= self.config.min_lines_for_ccr
            && ratio < 0.7
        {
            let hash = hash_for_ccr(content);
            compressed.push_str(&format!(
                "\n[{} lines compressed to {}. Retrieve more: hash={hash}]",
                lines.len(),
                selected.len()
            ));
            ratio = compressed.len() as f64 / content.len().max(1) as f64;
            Some(hash)
        } else {
            None
        };

        TextCompressionResult {
            compressed,
            original: content.to_string(),
            original_line_count: lines.len(),
            compressed_line_count: selected.len(),
            compression_ratio: ratio,
            cache_key,
        }
    }

    fn score_lines(&self, lines: &[String], context: &str) -> Vec<(usize, String, f64)> {
        let context_words = context
            .to_ascii_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .map(str::to_string)
            .collect::<HashSet<_>>();
        let anchor_keywords = self
            .config
            .anchor_keywords
            .iter()
            .map(|keyword| keyword.to_ascii_lowercase())
            .collect::<HashSet<_>>();

        lines
            .iter()
            .enumerate()
            .map(|(index, line)| {
                let lower = line.to_ascii_lowercase();
                let mut score: f64 = 0.0;
                for word in &context_words {
                    if lower.contains(word) {
                        score += 0.3;
                    }
                }
                for keyword in &anchor_keywords {
                    if lower.contains(keyword) {
                        score += 0.4;
                    }
                }
                if self.config.boost_pattern_lines
                    && [
                        &*ERROR_PATTERN,
                        &*IMPORTANCE_PATTERN,
                        &*HEADER_PATTERN,
                        &*BOLD_PATTERN,
                        &*QUOTE_PATTERN,
                    ]
                    .iter()
                    .any(|pattern| pattern.is_match(line))
                {
                    score += 0.2;
                }
                if !line.trim().is_empty() {
                    score += 0.1;
                }
                (index, line.clone(), score.min(1.0))
            })
            .collect()
    }

    fn select_lines(
        &self,
        scored_lines: &[(usize, String, f64)],
        original_lines: &[String],
    ) -> Vec<(usize, String)> {
        let total = scored_lines.len();
        let mut selected_indices = HashSet::new();
        for index in 0..self.config.keep_first_lines.min(total) {
            selected_indices.insert(index);
        }
        for index in total.saturating_sub(self.config.keep_last_lines)..total {
            selected_indices.insert(index);
        }

        let mut high_score_lines = scored_lines
            .iter()
            .filter(|(index, _, score)| *score >= 0.3 && !selected_indices.contains(index))
            .cloned()
            .collect::<Vec<_>>();
        high_score_lines.sort_by(|left, right| right.2.total_cmp(&left.2));

        let mut remaining_slots = self
            .config
            .max_total_lines
            .saturating_sub(selected_indices.len());
        for (index, _, _) in high_score_lines {
            if remaining_slots == 0 {
                break;
            }
            selected_indices.insert(index);
            remaining_slots -= 1;
        }

        if remaining_slots > 0 {
            let middle_start = self.config.keep_first_lines.min(total);
            let middle_end = total.saturating_sub(self.config.keep_last_lines);
            let mut index = middle_start;
            while index < middle_end && remaining_slots > 0 {
                if selected_indices.insert(index) {
                    remaining_slots -= 1;
                }
                index += self.config.sample_every_n_lines.max(1);
            }
        }

        let mut selected = selected_indices.into_iter().collect::<Vec<_>>();
        selected.sort_unstable();
        selected
            .into_iter()
            .map(|index| (index, original_lines[index].clone()))
            .collect()
    }

    fn format_output(&self, selected: &[(usize, String)], total_lines: usize) -> String {
        if selected.is_empty() {
            return format!("[{total_lines} lines omitted]");
        }
        let mut output_lines = Vec::new();
        let mut previous = None;
        for (index, line) in selected {
            if let Some(previous_index) = previous {
                if index.saturating_sub(previous_index) > 1 {
                    output_lines.push(format!(
                        "[... {} lines omitted ...]",
                        index - previous_index - 1
                    ));
                }
            }
            output_lines.push(line.clone());
            previous = Some(*index);
        }
        if let Some((last_index, _)) = selected.last() {
            if *last_index < total_lines.saturating_sub(1) {
                output_lines.push(format!(
                    "[... {} lines omitted ...]",
                    total_lines - last_index - 1
                ));
            }
        }
        output_lines.join("\n")
    }
}

fn hash_for_ccr(content: &str) -> String {
    let digest = Md5::digest(content.as_bytes());
    let hex = format!("{digest:x}");
    hex[..24].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_passes_through() {
        let compressor = TextCompressor::new(TextCompressorConfig::default());
        let result = compressor.compress("one\ntwo", "");
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn compresses_with_ellipsis() {
        let compressor = TextCompressor::new(TextCompressorConfig {
            keep_first_lines: 2,
            keep_last_lines: 2,
            max_total_lines: 6,
            sample_every_n_lines: 3,
            ..TextCompressorConfig::default()
        });
        let content = (1..=20)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = compressor.compress(&content, "");
        assert!(result.compressed.contains("[..."));
        assert!(result.compression_ratio < 1.0);
    }

    #[test]
    fn keeps_contextual_lines() {
        let compressor = TextCompressor::new(TextCompressorConfig {
            keep_first_lines: 1,
            keep_last_lines: 1,
            max_total_lines: 4,
            ..TextCompressorConfig::default()
        });
        let content = "intro\nsomething boring\ncritical error path here\nending\nmore\ntrailer";
        let result = compressor.compress(content, "critical error");
        assert!(result.compressed.contains("critical error path here"));
    }
}
