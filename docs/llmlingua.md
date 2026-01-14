# LLMLingua-2 Integration

For maximum compression, Headroom integrates with **LLMLingua-2**, Microsoft's BERT-based token classifier trained via GPT-4 distillation. It achieves **up to 20x compression** while preserving semantic meaning.

## When to Use LLMLingua-2

| Approach | Best For | Compression | Speed |
|----------|----------|-------------|-------|
| **SmartCrusher** | JSON tool outputs | 70-90% | ~1ms |
| **Text Utilities** | Search/logs | 50-90% | ~1ms |
| **LLMLingua-2** | Any text, max compression | 80-95% | ~50-200ms |

LLMLingua-2 is ideal when you need maximum compression and can tolerate slightly higher latency (e.g., compressing large tool outputs before storage, offline processing).

## Installation

```bash
# Adds ~2GB of model weights
pip install "headroom-ai[llmlingua]"
```

## Basic Usage

```python
from headroom.transforms import LLMLinguaCompressor

# Create compressor (model loaded lazily on first use)
compressor = LLMLinguaCompressor()

# Compress any text
long_output = "The function processUserData takes a user object and validates..."
result = compressor.compress(long_output)

print(f"Before: {result.original_tokens} tokens")
print(f"After: {result.compressed_tokens} tokens")
print(f"Saved: {result.savings_percentage:.1f}%")
print(result.compressed)
```

## Content-Aware Compression

LLMLingua-2 automatically adjusts compression based on content type:

```python
from headroom.transforms import LLMLinguaCompressor, LLMLinguaConfig

# Conservative for code (keep 40% of tokens)
config = LLMLinguaConfig(
    code_compression_rate=0.4,    # More conservative
    json_compression_rate=0.35,   # Moderate
    text_compression_rate=0.25,   # Aggressive
)

compressor = LLMLinguaCompressor(config)

# Auto-detects content type
code_result = compressor.compress("def calculate(x): return x * 2")
text_result = compressor.compress("This is a verbose explanation...")
```

## Memory Management

The model uses ~1GB RAM. Unload it when done:

```python
from headroom.transforms import (
    LLMLinguaCompressor,
    unload_llmlingua_model,
    is_llmlingua_model_loaded,
)

compressor = LLMLinguaCompressor()
result = compressor.compress(content)  # Model loaded here

# Check if loaded
print(is_llmlingua_model_loaded())  # True

# Free memory when done
unload_llmlingua_model()  # Frees ~1GB
print(is_llmlingua_model_loaded())  # False

# Next compression will reload automatically
```

## Device Configuration

```python
from headroom.transforms import LLMLinguaConfig, LLMLinguaCompressor

# Force CPU (slower but works everywhere)
config = LLMLinguaConfig(device="cpu")

# Force GPU (faster but needs CUDA)
config = LLMLinguaConfig(device="cuda")

# Auto-detect (default): uses CUDA > MPS > CPU
config = LLMLinguaConfig(device="auto")

compressor = LLMLinguaCompressor(config)
```

## Use in Pipeline

```python
from headroom.transforms import TransformPipeline, LLMLinguaCompressor, SmartCrusher

# Combine with other transforms
pipeline = TransformPipeline([
    SmartCrusher(),        # First: compress JSON
    LLMLinguaCompressor(), # Then: ML compression on remaining text
])

result = pipeline.apply(messages, tokenizer)
```

## Proxy Integration

Enable LLMLingua in the proxy server for automatic ML compression:

```bash
# Enable LLMLingua in proxy (requires: pip install headroom-ai[llmlingua,proxy])
headroom proxy --llmlingua

# With custom settings
headroom proxy --llmlingua --llmlingua-device cuda --llmlingua-rate 0.4

# The proxy shows LLMLingua status at startup:
#   LLMLingua: ENABLED  (device=cuda, rate=0.4)
#
# If llmlingua is installed but not enabled, you'll see a helpful hint:
#   LLMLingua: available (enable with --llmlingua for ML compression)
```

## Configuration Reference

| Option | Default | Description |
|--------|---------|-------------|
| `device` | `"auto"` | Device to run model on: auto, cpu, cuda, mps |
| `code_compression_rate` | `0.4` | Keep 40% of tokens for code |
| `json_compression_rate` | `0.35` | Keep 35% of tokens for JSON |
| `text_compression_rate` | `0.25` | Keep 25% of tokens for text |
| `force_tokens` | `[]` | Tokens to always preserve |
| `drop_consecutive` | `True` | Drop consecutive whitespace |

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Model size | ~500MB |
| Memory usage | ~1GB RAM |
| Cold start | 10-30s (first load) |
| Inference | 50-200ms per request |
| Compression | 80-95% |

## Why Opt-In?

LLMLingua adds significant dependencies and overhead:

| Aspect | Default Proxy | With LLMLingua |
|--------|--------------|----------------|
| Dependencies | ~50MB | ~2GB |
| Cold start | <1s | 10-30s |
| Per-request | ~1-5ms | ~50-200ms |
| Compression | 70-90% | 80-95% |

The default proxy is lightweight and fast. Enable LLMLingua when you need maximum compression and can accept the tradeoffs.

## Troubleshooting

### "Model not found"

```bash
# Ensure llmlingua extra is installed
pip install "headroom-ai[llmlingua]"
```

### "CUDA out of memory"

```python
# Force CPU mode
config = LLMLinguaConfig(device="cpu")
```

### "Slow compression"

- Use GPU if available: `device="cuda"`
- Batch multiple compressions
- Consider using SmartCrusher for JSON (faster, similar results)
