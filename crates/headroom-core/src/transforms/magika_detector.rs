use std::sync::LazyLock;

use regex::Regex;
use thiserror::Error;

use crate::transforms::content_detector::{detect_content_type, ContentType};

static SOURCE_CODE_HINTS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?m)^\s*(def|class|import|from|async def)\s+\w+").unwrap(),
        Regex::new(r"(?m)^\s*(fn|struct|enum|impl|use|pub)\s+").unwrap(),
        Regex::new(r"(?m)^\s*(function|const|let|var|class|import|export)\s+").unwrap(),
        Regex::new(r"(?m)^\s*(func|type|package)\s+").unwrap(),
        Regex::new(r"(?m)^#!/(bin/)?(ba)?sh\b").unwrap(),
        Regex::new(r"(?i)\bselect\b.+\bfrom\b").unwrap(),
    ]
});

fn looks_like_yaml(content: &str) -> bool {
    let mut keyed_lines = 0usize;
    let mut nested_line = false;
    for line in content.lines() {
        let trimmed = line.trim_end();
        if trimmed.starts_with("  - ") || trimmed.starts_with('\t') {
            nested_line = true;
        }
        if let Some((key, _value)) = trimmed.split_once(':') {
            if !key.is_empty()
                && key
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
            {
                keyed_lines += 1;
            }
        }
    }

    keyed_lines >= 2 && nested_line
}

#[derive(Debug, Error)]
pub enum MagikaDetectorError {
    #[error("magika feature is disabled")]
    Disabled,
}

pub fn magika_detect(content: &str) -> Result<ContentType, MagikaDetectorError> {
    if content.is_empty() {
        return Ok(ContentType::PlainText);
    }

    let detected = detect_content_type(content).content_type;
    Ok(match detected {
        ContentType::SearchResults | ContentType::BuildOutput => ContentType::PlainText,
        ContentType::PlainText
            if SOURCE_CODE_HINTS
                .iter()
                .any(|pattern| pattern.is_match(content))
                || looks_like_yaml(content) =>
        {
            ContentType::SourceCode
        }
        other => other,
    })
}

pub fn map_magika_label(label: &str) -> ContentType {
    match label {
        "json" | "jsonl" => ContentType::JsonArray,
        "diff" => ContentType::GitDiff,
        "html" | "xml" => ContentType::Html,
        "rust" | "python" | "javascript" | "typescript" | "go" | "java" | "c" | "cpp" | "cs"
        | "php" | "ruby" | "swift" | "kotlin" | "scala" | "haskell" | "lua" | "dart" | "perl"
        | "shell" | "powershell" | "batch" | "sql" | "css" | "vue" | "groovy" | "clojure"
        | "asm" | "cmake" | "dockerfile" | "makefile" | "yaml" | "toml" | "ini" | "hcl"
        | "jinja" => ContentType::SourceCode,
        "markdown" | "rst" | "latex" | "txt" | "empty" | "unknown" | "undefined" => {
            ContentType::PlainText
        }
        _ => ContentType::PlainText,
    }
}
