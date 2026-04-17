# 007. Behavior

**Status:** done

## Proxy Modes

### Passthrough Mode

Headroom forwards requests without modification.

**Behavior:**
- All requests pass through unchanged
- Response headers may be modified for telemetry
- No compression applied
- Useful for testing or debugging

**Configuration:** `HEADROOM_MODE=audit`

**Request Flow:**
```
Client → Proxy → Provider API → Response
```

---

### Optimize Mode

Headroom applies deterministic transforms to requests.

**Behavior:**
- SmartCrusher compresses JSON tool outputs
- CacheAligner stabilizes prefixes
- RollingWindow caps context tokens
- CCR caching enabled
- Token budget enforced

**Configuration:** `HEADROOM_MODE=optimize`

**Request Flow:**
```
Client → Proxy → [SmartCrusher] → [CacheAligner]
         → [RollingWindow] → [CCR Cache]
         → Provider API → Response
```

---

### Simulate Mode

Headroom returns transform plan without API call.

**Behavior:**
- Analyzes content for compression opportunity
- Returns TransformResult with planned transforms
- No actual compression or provider call
- Useful for debugging/optimization

**Configuration:** `HEADROOM_MODE=simulate`

---

## Session Modes

Session modes control how Headroom handles context windows.

| Mode | Description | Use Case |
|------|-------------|----------|
| `audit` | Observe only, no modifications | Monitoring |
| `optimize` | Apply deterministic transforms | Production |
| `simulate` | Return plan without API call | Debugging |

---

## Request Lifecycle

```
1. Request received at proxy endpoint
   │
   ▼
2. Session lookup/creation
   │  - Extract session ID from headers
   │  - Create new session if not found
   │
   ▼
3. Mode determination
   │  - Check HEADROOM_MODE
   │  - Check runtime headers
   │  - Determine active plugins
   │
   ▼
4. Compression pipeline execution
   │  a. Token counting
   │  b. Semantic cache check
   │  c. Content type detection
   │  d. Transform selection
   │  e. Summary compression (if eligible)
   │  f. Token budget enforcement
   │
   ▼
5. Forward to provider API
   │  - Route to correct provider
   │  - Apply API key from config
   │  - Handle timeouts
   │
   ▼
6. Response capture
   │  - Log request/response metadata
   │  - Calculate savings
   │
   ▼
7. Savings calculation
   │  - tokens_before - tokens_after
   │  - percentage = savings / tokens_before
   │
   ▼
8. Telemetry emission
   │  - Prometheus metrics
   │  - Optional tracing
   │
   ▼
9. Response returned to client
      - X-Headroom-Savings header
      - X-Headroom-Original-Tokens header
      - X-Headroom-Compressed-Tokens header
```

---

## Error Handling

| Error Type | HTTP Code | Behavior |
|------------|----------|----------|
| Provider timeout | 504 | Retry up to 3 times with exponential backoff |
| Invalid request | 400 | Return error details in body |
| Compression failure | 500 | Fall back to passthrough mode |
| Provider error | Provider code | Return provider error to client |
| Internal error | 500 | Return 500, log details |
| Rate limited | 429 | Return retry-after header |

**Retry Configuration:**
```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
```

---

## Response Headers

Headroom adds headers to all compressed responses:

```
X-Headroom-Savings: 0.35
X-Headroom-Original-Tokens: 8192
X-Headroom-Compressed-Tokens: 5325
X-Headroom-Compression-Type: semantic,summary
X-Headroom-Request-Id: abc123
X-Headroom-Cache-Hit: false
```

**Header Descriptions:**
- `X-Headroom-Savings` — Token savings percentage (0.35 = 35%)
- `X-Headroom-Original-Tokens` — Token count before compression
- `X-Headroom-Compressed-Tokens` — Token count after compression
- `X-Headroom-Compression-Type` — Types of compression applied
- `X-Headroom-Request-Id` — Unique request identifier
- `X-Headroom-Cache-Hit` — Whether result was from cache

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial behavior document |
