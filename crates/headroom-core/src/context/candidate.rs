//! Build drop candidates from a scored message list.
//!
//! Three candidate shapes (matching Python):
//!
//! - **Tool unit** — atomic `(assistant_with_tool_calls, [tool_responses])`.
//!   Either the whole unit is dropped or none of it. Score = mean of
//!   member scores.
//! - **Turn** — paired `(user, assistant)` neighbours when both are
//!   unprotected and neither is in a tool unit. Score = mean of the
//!   two scores.
//! - **Single** — any other unprotected, non-tool-unit message that
//!   couldn't be paired. Score = the message's own score.
//!
//! Candidates sort by score ascending (lowest = drop first), with
//! position as a tiebreaker (older first).

use std::collections::HashSet;

use serde_json::Value;

use crate::scoring::MessageScore;

/// One unit the cascade may drop atomically.
#[derive(Debug, Clone)]
pub struct DropCandidate {
    pub kind: CandidateKind,
    /// Indices to remove together. May be 1 (single), 2 (turn), or
    /// `1 + N` (tool unit with N tool responses).
    pub indices: Vec<usize>,
    /// Aggregated score; lower = drop first.
    pub score: f32,
    /// Earliest index — used as the secondary sort key for stability
    /// (older candidates of equal score drop first).
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Single,
    Turn,
    ToolUnit,
}

/// Tool unit: `(assistant_index, [tool_response_indices])`.
pub type ToolUnit = (usize, Vec<usize>);

/// Find all tool units in the message list. Mirrors Python's
/// `headroom.parser.find_tool_units`. Handles OpenAI, Anthropic, and
/// Strands SDK shapes.
pub fn find_tool_units(messages: &[Value]) -> Vec<ToolUnit> {
    use std::collections::HashMap;

    // tool_call_id → response message index
    let mut response_map: HashMap<String, usize> = HashMap::new();
    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(Value::as_str);

        // OpenAI: role=tool, tool_call_id
        if role == Some("tool") {
            if let Some(tcid) = msg.get("tool_call_id").and_then(Value::as_str) {
                response_map.insert(tcid.to_string(), i);
            }
        }

        // Anthropic + Strands: role=user, content blocks
        if role == Some("user") {
            if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                for block in blocks {
                    // Anthropic: {type: tool_result, tool_use_id}
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(tcid) = block.get("tool_use_id").and_then(Value::as_str) {
                            response_map.insert(tcid.to_string(), i);
                        }
                    }
                    // Strands: {toolResult: {toolUseId}}
                    if let Some(tr) = block.get("toolResult") {
                        if let Some(tcid) = tr.get("toolUseId").and_then(Value::as_str) {
                            response_map.insert(tcid.to_string(), i);
                        }
                    }
                }
            }
        }
    }

    let mut units: Vec<ToolUnit> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }

        let mut response_indices: Vec<usize> = Vec::new();

        // OpenAI tool_calls
        if let Some(arr) = msg.get("tool_calls").and_then(Value::as_array) {
            for tc in arr {
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    if let Some(&idx) = response_map.get(id) {
                        response_indices.push(idx);
                    }
                }
            }
        }

        // Anthropic + Strands content blocks
        if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
            for block in blocks {
                // Anthropic: {type: tool_use, id}
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    if let Some(id) = block.get("id").and_then(Value::as_str) {
                        if let Some(&idx) = response_map.get(id) {
                            response_indices.push(idx);
                        }
                    }
                }
                // Strands: {toolUse: {toolUseId}}
                if let Some(tu) = block.get("toolUse") {
                    if let Some(id) = tu.get("toolUseId").and_then(Value::as_str) {
                        if let Some(&idx) = response_map.get(id) {
                            response_indices.push(idx);
                        }
                    }
                }
            }
        }

        if !response_indices.is_empty() {
            response_indices.sort_unstable();
            response_indices.dedup();
            units.push((i, response_indices));
        }
    }

    units
}

/// Aggregate `messages × scores × tool_units` into a sorted list of
/// drop candidates. Lowest score first; ties broken by position.
pub fn build_candidates(
    messages: &[Value],
    scores: &[MessageScore],
    protected: &HashSet<usize>,
    tool_units: &[ToolUnit],
) -> Vec<DropCandidate> {
    // Track every index that's part of some tool unit — we don't
    // want a tool-unit message to ALSO appear as a single or turn.
    let mut tool_unit_indices: HashSet<usize> = HashSet::new();
    for (asst_idx, responses) in tool_units {
        tool_unit_indices.insert(*asst_idx);
        for &r in responses {
            tool_unit_indices.insert(r);
        }
    }

    let mut candidates: Vec<DropCandidate> = Vec::new();

    // Tool unit candidates — drop atomically.
    for (asst_idx, responses) in tool_units {
        if protected.contains(asst_idx) {
            continue;
        }
        // If any response is protected, the unit can't drop atomically.
        // Skip it; its messages stay (mirrors Python behaviour).
        if responses.iter().any(|r| protected.contains(r)) {
            continue;
        }
        let mut indices: Vec<usize> = vec![*asst_idx];
        indices.extend(responses);
        let avg = mean_score(&indices, scores);
        candidates.push(DropCandidate {
            kind: CandidateKind::ToolUnit,
            indices,
            score: avg,
            position: *asst_idx,
        });
    }

    // User+assistant turn pairs and singles.
    let mut i: usize = 0;
    while i < messages.len() {
        if protected.contains(&i) || tool_unit_indices.contains(&i) {
            i += 1;
            continue;
        }
        let role = messages[i].get("role").and_then(Value::as_str);

        // Try to pair user with the next assistant.
        if role == Some("user") && i + 1 < messages.len() {
            let next_role = messages[i + 1].get("role").and_then(Value::as_str);
            let next_unprotected = !protected.contains(&(i + 1));
            let next_not_in_tool_unit = !tool_unit_indices.contains(&(i + 1));
            if next_role == Some("assistant") && next_unprotected && next_not_in_tool_unit {
                let avg = mean_score(&[i, i + 1], scores);
                candidates.push(DropCandidate {
                    kind: CandidateKind::Turn,
                    indices: vec![i, i + 1],
                    score: avg,
                    position: i,
                });
                i += 2;
                continue;
            }
        }

        // Single — any unprotected, non-tool-unit user/assistant message.
        if matches!(role, Some("user") | Some("assistant")) {
            let s = scores.get(i).map(|sc| sc.total_score).unwrap_or(0.5);
            candidates.push(DropCandidate {
                kind: CandidateKind::Single,
                indices: vec![i],
                score: s,
                position: i,
            });
        }

        i += 1;
    }

    // Sort by (score asc, position asc) — lowest score drops first;
    // older messages drop first as a tiebreaker.
    candidates.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.position.cmp(&b.position))
    });

    candidates
}

fn mean_score(indices: &[usize], scores: &[MessageScore]) -> f32 {
    if indices.is_empty() {
        return 0.5;
    }
    let total: f32 = indices
        .iter()
        .filter_map(|&i| scores.get(i).map(|s| s.total_score))
        .sum();
    let count = indices.iter().filter(|&&i| scores.get(i).is_some()).count() as f32;
    if count == 0.0 {
        0.5
    } else {
        total / count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn score(idx: usize, total: f32) -> MessageScore {
        MessageScore {
            message_index: idx,
            total_score: total,
            recency_score: 0.0,
            semantic_score: 0.0,
            toin_score: 0.0,
            error_score: 0.0,
            reference_score: 0.0,
            density_score: 0.0,
            tokens: 0,
            is_protected: false,
            drop_safe: true,
            score_breakdown: Default::default(),
        }
    }

    #[test]
    fn find_tool_units_openai_shape() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "result"}),
        ];
        let units = find_tool_units(&msgs);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].0, 1);
        assert_eq!(units[0].1, vec![2]);
    }

    #[test]
    fn find_tool_units_anthropic_shape() {
        let msgs = vec![
            json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}
                ]
            }),
            json!({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]
            }),
        ];
        let units = find_tool_units(&msgs);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].0, 0);
        assert_eq!(units[0].1, vec![1]);
    }

    #[test]
    fn find_tool_units_strands_sdk_shape() {
        let msgs = vec![
            json!({
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tu_x", "name": "f"}}]
            }),
            json!({
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "tu_x", "content": "ok"}}]
            }),
        ];
        let units = find_tool_units(&msgs);
        assert_eq!(units.len(), 1);
    }

    #[test]
    fn lowest_score_sorts_first() {
        let msgs = vec![
            json!({"role": "user", "content": "high"}),
            json!({"role": "assistant", "content": "high"}),
            json!({"role": "user", "content": "low"}),
            json!({"role": "assistant", "content": "low"}),
        ];
        let scores = vec![score(0, 0.9), score(1, 0.9), score(2, 0.1), score(3, 0.1)];
        let cands = build_candidates(&msgs, &scores, &HashSet::new(), &[]);
        // The low-score turn (idx 2,3) should come first.
        assert_eq!(cands[0].kind, CandidateKind::Turn);
        assert_eq!(cands[0].indices, vec![2, 3]);
    }

    #[test]
    fn protected_messages_excluded() {
        let msgs = vec![
            json!({"role": "user", "content": "u"}),
            json!({"role": "assistant", "content": "a"}),
        ];
        let scores = vec![score(0, 0.5), score(1, 0.5)];
        let mut protected = HashSet::new();
        protected.insert(1); // protect the assistant
        let cands = build_candidates(&msgs, &scores, &protected, &[]);
        // Pair impossible (asst protected); user becomes single; asst skipped.
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, CandidateKind::Single);
        assert_eq!(cands[0].indices, vec![0]);
    }

    #[test]
    fn tool_unit_takes_precedence_over_singles() {
        let msgs = vec![
            json!({"role": "user", "content": "go"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r"}),
        ];
        let scores = vec![score(0, 0.5), score(1, 0.2), score(2, 0.3)];
        let units = find_tool_units(&msgs);
        let cands = build_candidates(&msgs, &scores, &HashSet::new(), &units);
        // Two candidates: the user (single) and the tool unit. The
        // tool unit holds asst+tool atomically; neither asst nor tool
        // appears as its own candidate.
        let unit_cands: Vec<_> = cands
            .iter()
            .filter(|c| c.kind == CandidateKind::ToolUnit)
            .collect();
        assert_eq!(unit_cands.len(), 1);
        let single_cands: Vec<_> = cands
            .iter()
            .filter(|c| c.kind == CandidateKind::Single)
            .collect();
        assert_eq!(single_cands.len(), 1);
        assert_eq!(single_cands[0].indices, vec![0]);
    }

    #[test]
    fn tool_unit_with_protected_response_is_skipped() {
        let msgs = vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "f"}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r"}),
        ];
        let scores = vec![score(0, 0.2), score(1, 0.3)];
        let mut protected = HashSet::new();
        protected.insert(1); // protect the response
        let units = find_tool_units(&msgs);
        let cands = build_candidates(&msgs, &scores, &protected, &units);
        // Unit can't drop atomically when one half is protected.
        assert!(cands.iter().all(|c| c.kind != CandidateKind::ToolUnit));
    }

    #[test]
    fn position_tiebreaker_for_equal_scores() {
        let msgs = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "first"}),
            json!({"role": "user", "content": "second"}),
            json!({"role": "assistant", "content": "second"}),
        ];
        let scores = vec![score(0, 0.5), score(1, 0.5), score(2, 0.5), score(3, 0.5)];
        let cands = build_candidates(&msgs, &scores, &HashSet::new(), &[]);
        // Older turn drops first.
        assert_eq!(cands[0].position, 0);
    }
}
