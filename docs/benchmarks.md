# Accuracy Benchmarks

Headroom's core promise: **compress context without losing accuracy**. This page shows our latest benchmark results against established open-source datasets.

!!! success "Key Result"
    **98.2% recall** on article extraction with **94.9% compression** — we preserve nearly all information while dramatically reducing tokens.

---

## Summary

| Benchmark | Metric | Headroom | Baseline | Status |
|-----------|--------|----------|----------|--------|
| [Scrapinghub Article Extraction](#html-extraction) | F1 Score | **0.919** | 0.958 | :white_check_mark: |
| [Scrapinghub Article Extraction](#html-extraction) | Recall | **98.2%** | — | :white_check_mark: |
| [Scrapinghub Article Extraction](#html-extraction) | Compression | **94.9%** | — | :white_check_mark: |
| [SmartCrusher (JSON)](#json-compression) | Accuracy | **100%** | — | :white_check_mark: |
| [SmartCrusher (JSON)](#json-compression) | Compression | **87.6%** | — | :white_check_mark: |

---

## HTML Extraction

**Dataset**: [Scrapinghub Article Extraction Benchmark](https://huggingface.co/datasets/allenai/scrapinghub-article-extraction-benchmark)
**Samples**: 181 HTML pages with ground truth article bodies
**Baseline**: trafilatura (0.958 F1)

HTMLExtractor removes scripts, styles, navigation, ads, and boilerplate while preserving article content.

### Results

| Metric | Value | Description |
|--------|-------|-------------|
| **F1 Score** | 0.919 | Token-level overlap with ground truth |
| **Precision** | 0.879 | Proportion of extracted content that's relevant |
| **Recall** | 0.982 | Proportion of ground truth content captured |
| **Compression** | 94.9% | Average size reduction |

### Why Recall Matters Most

For LLM applications, **recall is critical** — we must capture all relevant information. A 98.2% recall means:

- Nearly all article content is preserved
- LLMs can answer questions accurately from extracted content
- The slight precision drop (some extra content) doesn't hurt LLM accuracy

### Run It Yourself

```bash
# Install dependencies
pip install "headroom-ai[html]" datasets

# Run the benchmark
pytest tests/test_evals/test_html_oss_benchmarks.py::TestExtractionBenchmark -v -s
```

---

## JSON Compression (SmartCrusher)

**Test**: 100 production log entries with critical error at position 67
**Task**: Find the error, error code, resolution, and affected count

### Results

| Metric | Baseline | Headroom |
|--------|----------|----------|
| Input tokens | 10,144 | 1,260 |
| Correct answers | 4/4 | **4/4** |
| Compression | — | **87.6%** |

SmartCrusher preserves:

- First N items (schema examples)
- Last N items (recency)
- All anomalies (errors, warnings, outliers)
- Statistical distribution

### Run It Yourself

```bash
python examples/needle_in_haystack_test.py
```

---

## QA Accuracy Preservation

We verify that LLMs can answer questions equally well from compressed content.

**Method**:
1. Take original HTML content
2. Extract with HTMLExtractor
3. Ask LLM same question on both
4. Compare answers against ground truth

**Datasets**: SQuAD v2, HotpotQA

### Results

| Metric | Original HTML | Extracted | Delta |
|--------|---------------|-----------|-------|
| F1 Score | 0.85 | 0.87 | +0.02 |
| Exact Match | 60% | 62% | +2% |

!!! note "Extraction Can Improve Accuracy"
    Removing HTML noise sometimes *helps* LLMs focus on relevant content.

### Run It Yourself

```bash
# Requires OPENAI_API_KEY
pytest tests/test_evals/test_html_oss_benchmarks.py::TestQAAccuracyPreservation -v -s
```

---

## Multi-Tool Agent Test

**Setup**: Agno agent with 4 tools investigating a memory leak
**Total tool output**: 62,323 chars (~15,580 tokens)

### Results

| Metric | Baseline | Headroom |
|--------|----------|----------|
| Tokens sent | 15,662 | 6,100 |
| Tool calls | 4 | 4 |
| Correct findings | All | **All** |
| Compression | — | **76.3%** |

Both found: Issue #42, `cleanup_worker()` fix, OutOfMemoryError logs, relevant papers.

### Run It Yourself

```bash
python examples/multi_tool_agent_test.py
```

---

## Methodology

### Token-Level F1

We use the standard NLP metric for text overlap:

```
Precision = |predicted ∩ ground_truth| / |predicted|
Recall = |predicted ∩ ground_truth| / |ground_truth|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### QA Accuracy

For question-answering, we measure:

- **Exact Match**: Normalized answer strings match exactly
- **F1 Score**: Token overlap between predicted and ground truth answers

### Compression Ratio

```
Compression = 1 - (compressed_size / original_size)
```

A 94.9% compression means the output is 5.1% of the original size.

---

## Reproducing Results

All benchmarks are reproducible:

```bash
# Clone the repo
git clone https://github.com/chopratejas/headroom.git
cd headroom

# Install with eval dependencies
pip install -e ".[evals,html]"

# Run all benchmarks
pytest tests/test_evals/ -v -s

# Run specific benchmark
pytest tests/test_evals/test_html_oss_benchmarks.py -v -s
```

### CI Integration

Benchmarks run on every PR. See [.github/workflows/ci.yml](https://github.com/chopratejas/headroom/blob/main/.github/workflows/ci.yml).

---

## Adding New Benchmarks

We welcome contributions! See [CONTRIBUTING.md](https://github.com/chopratejas/headroom/blob/main/CONTRIBUTING.md) for guidelines.

Benchmarks should:

1. Use established open-source datasets
2. Include reproducible evaluation code
3. Test accuracy preservation, not just compression
4. Run in CI without API keys (or skip gracefully)
