# PII / Secret Filter for Headroom Proxy — Design

**Date:** 2026-05-03
**Status:** Draft (awaiting user approval)
**Owner:** TBD
**Related:** `headroom/transforms/pipeline.py`, `headroom/proxy/handlers/{anthropic,openai,gemini}.py`, `headroom/config.py`

---

## 1. Goal

Detect and mask personally identifiable information (PII) and credentials/secrets in user-supplied request bodies **before** they are forwarded to upstream LLM providers (Anthropic, OpenAI, etc).

The filter is opt-in via three independent mechanisms:

1. **Environment variable** — `HEADROOM_PII_ENABLED=true` (default off).
2. **Per-request header** — `X-Headroom-PII: on|off` (overrides env).
3. **Dedicated mirrored route** — `POST /openai/pii/v1/chat/completions`, `POST /anthropic/pii/v1/messages` — same request/response shape as the regular routes but the filter is forced on regardless of env or header.

Confidence threshold for ML-based detection is configurable globally and per-label.

## 2. Non-goals

- Reverse-substitution / round-tripping the original PII back into the upstream response.
- Scanning assistant responses, tool results, system prompts, image/audio bytes, or tool-call argument JSON.
- Compliance certification (HIPAA / GDPR DPA). The filter is a defence-in-depth control, not a guarantee.
- Streaming-response inspection.
- Built-in pseudonymization / format-preserving fake-data generation.

## 3. Locked design decisions

| Axis | Decision |
|---|---|
| Detection backend | Hybrid: Microsoft **Presidio** + custom **regex** detector + **detect-secrets** |
| Masking strategy | **Tag redact** — replace with `[<LABEL>]` (e.g., `[PERSON]`, `[EMAIL]`, `[AWS_ACCESS_KEY]`) |
| Default behavior | **Off**; opt-in via env / header / mirrored route |
| Scan scope | **User messages only** (`role == "user"` text blocks). System / assistant / tool_result / tool_use untouched. Documented limitation: PII echoed in tool results is not caught. |
| Failure mode | **Fail-open** — request always proceeds; failed detector contributes no spans; logged and counted |
| Confidence | Global default `min_score=0.85`; per-label override map; per-request header override |
| Integration point | New first-stage transform in `headroom/transforms/pipeline.py` |
| Response path | Untouched (one-way mask) |

## 4. Architecture

```
                ┌─────────────────────────────────┐
client ────►    │  FastAPI route                  │
                │   /v1/messages                  │
                │   /v1/chat/completions          │
                │   /openai/pii/v1/chat/...   ──► force_pii=True
                │   /anthropic/pii/v1/messages──► force_pii=True
                └──────────────┬──────────────────┘
                               │ body dict + headers
                               ▼
                ┌─────────────────────────────────┐
                │  Handler mixin (anthropic/      │
                │  openai/gemini)                 │
                │  builds RequestContext.flags    │
                └──────────────┬──────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │  transforms/pipeline.py         │
                │  ┌───────────────────────────┐  │
                │  │ PIIRedactor   (NEW, 1st)  │  │ ◄── opt-in: env / hdr / route flag
                │  ├───────────────────────────┤  │
                │  │ existing compressors...   │  │
                │  └───────────────────────────┘  │
                └──────────────┬──────────────────┘
                               │ redacted body
                               ▼
                       upstream LLM
```

**Key invariants**

- `PIIRedactor` runs **first** in pipeline ordering — secrets never touch the compressor or `cache_aligner`.
- `PIIRedactor.run()` is a no-op unless `should_run(ctx)` returns true.
- Detector exceptions and timeouts are caught at the pipeline edge and converted to "skip stage, continue" + WARN log + counter increment. The request always proceeds.
- Only `role == "user"` text blocks are mutated.

## 5. Components

### 5.1 New module tree

```
headroom/transforms/pii/
├── __init__.py
├── redactor.py        # PIIRedactor(Transform) — pipeline stage entry
├── detectors/
│   ├── __init__.py
│   ├── base.py        # Detector ABC + Span dataclass
│   ├── presidio.py    # PresidioDetector — wraps presidio-analyzer
│   ├── regex.py       # RegexDetector — email/phone/SSN/IBAN/card/IP/MRN/...
│   └── secrets.py     # SecretDetector — wraps detect-secrets plugins
├── masker.py          # tag_redact(spans, text) -> str
└── span.py            # @dataclass Span(start, end, label, score, source)
```

### 5.2 Surgical edits to existing files

| File | Change |
|---|---|
| `headroom/transforms/pipeline.py` | Register `PIIRedactor` at index 0; thread `force_pii` flag through `RequestContext.flags`. |
| `headroom/config.py` | Add `PIIConfig` dataclass; embed in top-level `ProxyConfig`. |
| `headroom/proxy/server.py` | `_pii_config_from_env()` helper; register mirrored `/openai/pii/...` and `/anthropic/pii/...` routes that set `request.state.force_pii = True`. |
| `headroom/proxy/handlers/anthropic.py` | Read `request.state.force_pii` + `X-Headroom-PII` header; populate `RequestContext.flags`. |
| `headroom/proxy/handlers/openai.py` | Same as Anthropic. |
| `headroom/proxy/handlers/gemini.py` | Same; route group registered when Gemini handler used. |
| `pyproject.toml` | New optional dependency group `[pii]`. |

### 5.3 Detector contract

```python
@dataclass(frozen=True)
class Span:
    start: int          # char offset (NFC-normalized) in source text
    end: int            # exclusive
    label: str          # PERSON | EMAIL | AWS_ACCESS_KEY | ...
    score: float        # 0.0–1.0; regex/secrets fixed at 1.0
    source: str         # "presidio" | "regex" | "secrets"

class Detector(Protocol):
    def detect(self, text: str, lang: str = "en") -> list[Span]: ...
```

### 5.4 Composition order inside `PIIRedactor.run(text)`

1. `RegexDetector.detect()` (always; ~µs; covers structured PII).
2. `SecretDetector.detect()` (always; detect-secrets plugins).
3. `PresidioDetector.detect()` (if loaded; covers free-form PERSON/LOCATION/etc).
4. Drop spans below threshold (`min_score`, with per-label overrides).
5. Resolve overlaps (longest span wins; ties: `secrets > regex > presidio`).
6. `masker.tag_redact()` walks spans right-to-left, substituting `[<LABEL>]`.

### 5.5 Lazy loading

- `PresidioDetector` constructed lazily on first request (cold start ~1.5s for spaCy `en_core_web_sm`).
- `RegexDetector` and `SecretDetector` eager (cheap).

## 6. Data flow

### 6.1 Opt-in resolution (per request, in handler mixin)

```python
pii_config = proxy.config.pii  # ProxyConfig.pii: PIIConfig
hdr = _header_bool(request.headers, pii_config.header_name, default=None)
force_pii = (
    getattr(request.state, "force_pii", False)
    or (hdr is True)
    or (hdr is None and pii_config.enabled)
)
# header set to "off" must override env-on; only fall back to env when header is absent
```

Precedence: route flag > header > env. Header `X-Headroom-PII: off` wins over env-on (explicit per-request opt-out).

### 6.2 Anthropic body shape (`POST /v1/messages`)

```jsonc
{
  "model": "...",
  "system": "...",                          // SKIPPED
  "messages": [
    { "role": "user",     "content": "string OR [blocks]" },   // ← scan
    { "role": "assistant","content": "..." },                  // skip
    { "role": "user",     "content": [
        { "type": "text", "text": "scan THIS" },               // ← scan
        { "type": "image", "source": {...} },                  // skip
        { "type": "tool_result", "content": [...] }            // skip (per locked scope)
    ]}
  ]
}
```

### 6.3 OpenAI body shape (`POST /v1/chat/completions`)

```jsonc
{
  "model": "...",
  "messages": [
    { "role": "system",    "content": "..." },                 // skip
    { "role": "user",      "content": "string OR [parts]" },   // ← scan
    { "role": "assistant", "content": "..." },                 // skip
    { "role": "tool",      "content": "..." }                  // skip
  ]
}
```

User parts array: scan `{"type": "text", "text": ...}` only; skip `image_url` / `input_audio`.

### 6.4 Walk algorithm

```python
def run(self, body: dict) -> dict:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body
    for msg in msgs:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = self._redact_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = self._redact_text(block["text"])
    return body
```

### 6.5 Per-message redaction

```python
def _redact_text(self, text: str) -> str:
    if not text or not text.strip():
        return text
    if len(text.encode("utf-8")) > self.cfg.max_text_bytes:
        log.warning("pii: message exceeds %d bytes; skipping", self.cfg.max_text_bytes)
        return text
    text = unicodedata.normalize("NFC", text)
    spans: list[Span] = []
    for det in self._enabled_detectors():
        try:
            spans.extend(det.detect(text, lang=self.cfg.language))
        except Exception:
            log.warning("pii: %s detector failed", det.name, exc_info=True)
            self._counter_errors.labels(det.name).inc()
    spans = self._drop_below_threshold(spans)
    spans = self._resolve_overlaps(spans)
    masked = self._mask_right_to_left(text, spans)
    self._record_telemetry(spans)
    return masked
```

### 6.6 Response path

Untouched. Tag-redact is one-way; no reverse substitution. Upstream response returned to client unmodified.

### 6.7 Telemetry

- Counter: `headroom_pii_redactions_total{label, source, route}`
- Counter: `headroom_pii_detector_errors_total{detector}`
- Counter: `headroom_pii_detector_timeouts_total{detector}`
- Counter: `headroom_pii_total_failure_total` (all detectors failed)
- Histogram: `headroom_pii_latency_ms{stage}` (regex / secrets / presidio / total)
- INFO log per request: `pii: redacted=N labels=[...] dur=Xms` (counts only)

**Hard rule:** original PII values are never written to logs, never returned to the client, never persisted.

## 7. Config surface

### 7.1 `PIIConfig` dataclass

```python
@dataclass
class PIIConfig:
    enabled: bool = False                                  # master env switch
    regex_enabled: bool = True
    secrets_enabled: bool = True
    presidio_enabled: bool = True                          # auto-disabled if dep missing
    min_score: float = 0.85
    label_scores: dict[str, float] = field(default_factory=dict)
    enabled_labels: set[str] = field(default_factory=set)  # empty = all
    disabled_labels: set[str] = field(default_factory=set)
    mask_format: str = "[{label}]"
    presidio_timeout_ms: int = 200
    language: str = "en"
    header_name: str = "X-Headroom-PII"
    high_entropy_secrets: bool = False                     # noisy; opt-in
    max_text_bytes: int = 1_000_000                        # per-message cap
```

### 7.2 Environment variables

| Env var | Default | Notes |
|---|---|---|
| `HEADROOM_PII_ENABLED` | `false` | Master switch |
| `HEADROOM_PII_REGEX_ENABLED` | `true` | |
| `HEADROOM_PII_SECRETS_ENABLED` | `true` | |
| `HEADROOM_PII_PRESIDIO_ENABLED` | `true` | No-op if package missing |
| `HEADROOM_PII_MIN_SCORE` | `0.85` | Float 0–1 |
| `HEADROOM_PII_LABEL_SCORES` | `{}` | JSON, e.g. `{"PERSON":0.9}` |
| `HEADROOM_PII_ENABLED_LABELS` | `""` | CSV; empty = all |
| `HEADROOM_PII_DISABLED_LABELS` | `""` | CSV |
| `HEADROOM_PII_MASK_FORMAT` | `[{label}]` | Python format template |
| `HEADROOM_PII_PRESIDIO_TIMEOUT_MS` | `200` | Soft deadline |
| `HEADROOM_PII_LANGUAGE` | `en` | Presidio nlp lang |
| `HEADROOM_PII_HEADER_NAME` | `X-Headroom-PII` | Override if proxy strips |
| `HEADROOM_PII_HIGH_ENTROPY` | `false` | Generic high-entropy detector |
| `HEADROOM_PII_MAX_TEXT_BYTES` | `1000000` | Per-message cap |

### 7.3 Per-request headers

| Header | Values | Effect |
|---|---|---|
| `X-Headroom-PII` | `on` / `off` / `1` / `0` / `true` / `false` | Override env per-request |
| `X-Headroom-PII-Min-Score` | float `0`–`1` | Override `min_score` for this request |
| `X-Headroom-PII-Labels` | CSV | Restrict scan to these labels |

### 7.4 Mirrored routes (filter forced on)

| Method | Path | Forwards to |
|---|---|---|
| POST | `/anthropic/pii/v1/messages` | `/v1/messages` |
| POST | `/openai/pii/v1/chat/completions` | `/v1/chat/completions` |
| POST | `/openai/pii/v1/responses` | `/v1/responses` |

(Gemini route added when Gemini handler is in active use; same prefix pattern: `/gemini/pii/...`.)

**Implementation note:** mirrored routes are registered as additional FastAPI routes whose handler functions set `request.state.force_pii = True` and then delegate to the same handler function as the regular route. No internal path rewrite or HTTP redirect — the same code path runs with one extra flag set.

### 7.5 Optional dependency group

```toml
# pyproject.toml
[project.optional-dependencies]
pii = [
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
    "spacy>=3.7.0",
    "detect-secrets>=1.5.0",
]
```

Install: `pip install headroom-ai[pii]`. spaCy model `en_core_web_sm` must be downloaded separately (`python -m spacy download en_core_web_sm`); install hint logged at startup if presidio is enabled but the model is missing.

### 7.6 Default label sets

| Source | Labels |
|---|---|
| Regex | `EMAIL`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`, `URL`, `MAC_ADDRESS`, `MEDICAL_RECORD_NUMBER` |
| Secrets | `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `GITHUB_TOKEN`, `SLACK_TOKEN`, `STRIPE_KEY`, `JWT`, `PRIVATE_KEY`, `HIGH_ENTROPY_STRING` (opt-in) |
| Presidio | `PERSON`, `LOCATION`, `ORGANIZATION`, `DATE_TIME`, `NRP` |

## 8. Error handling + edge cases

### 8.1 Failure matrix (all paths fail-open)

| Failure | Behavior | Telemetry |
|---|---|---|
| `presidio-analyzer` not installed | WARN once at startup; `presidio_enabled=False` | startup log |
| spaCy model `en_core_web_sm` missing | WARN once; `presidio_enabled=False` | startup log |
| `PresidioDetector.detect()` raises | drop Presidio spans; keep regex+secrets; continue | `pii_detector_errors_total{detector="presidio"}` |
| `PresidioDetector.detect()` exceeds `presidio_timeout_ms` | `asyncio.wait_for` cancels; drop Presidio spans | `pii_detector_timeouts_total` |
| `RegexDetector` raises | drop regex spans; continue | counter |
| `SecretDetector` raises | drop secrets spans; continue | counter |
| All detectors fail | forward original unmasked body + WARN | `pii_total_failure_total` |
| Body JSON malformed | no-op; downstream returns its own error | none |
| `messages` field missing/invalid type | no-op; continue | none |
| `min_score` env un-parseable | use default; WARN at startup | startup log |

### 8.2 Edge cases

- **Overlapping spans.** Resolver picks longest; ties → `secrets > regex > presidio`. E.g., `john@acme.com` masked as `[EMAIL]`, not `[PERSON]@acme.com`.
- **Adjacent spans, same label.** Presidio model already aggregates BIO tags; no merge step needed.
- **Unicode / multi-byte.** Normalize to NFC at entry; offsets are Python str indices (code points); right-to-left substitution preserves earlier offsets.
- **Empty / whitespace-only text.** Skip detector calls.
- **Very large messages.** Per-message cap `max_text_bytes=1_000_000`; above cap → WARN, skip filter for that message (fail-open).
- **Streaming requests.** Filter runs on full request body before upstream call; chat-completion request bodies are non-streaming. No streaming concern.
- **Loopback.** `proxy/loopback_guard.py` runs after filter; unaffected.
- **Cache aligner / dedup.** Filter runs before cache_aligner — masked text is what gets cached; no leak via cache.
- **Tool args containing user PII.** Per scope (user msgs only), tool_use args **not** scanned. Documented limitation.
- **False positives mangling code.** A user asking about a literal AWS key gets it masked. Workaround: per-request `X-Headroom-PII-Labels` excluding `AWS_ACCESS_KEY`.
- **High-entropy detector.** Off by default. When enabled: `entropy >= 4.5` AND `length >= 20` AND not part of recognized URL/UUID/SHA.
- **Health / metrics endpoints.** Filter only runs inside provider handlers; `/health`, `/metrics`, `/docs` unaffected.
- **Mirrored route + opt-out header.** `/openai/pii/v1/...` with `X-Headroom-PII: off` → route flag wins (forced on). Documented behavior.

## 9. Testing

### 9.1 Test layout

```
tests/transforms/pii/
├── test_redactor.py
├── test_regex_detector.py
├── test_secrets_detector.py
├── test_presidio_detector.py        # @pytest.mark.optional_dep("presidio")
├── test_masker.py
├── test_span_resolver.py
└── fixtures/
    ├── anthropic_messages.json
    ├── openai_chat.json
    └── pii_corpus.txt
tests/proxy/
├── test_pii_routes.py
└── test_pii_header_opt_in.py
```

### 9.2 Coverage matrix

| Layer | What | How |
|---|---|---|
| Span | offsets, equality, ordering | unit |
| RegexDetector | each label hits its pattern; no FP on near-misses; Luhn on cards; IBAN checksum; AWS prefix | unit, table-driven |
| SecretDetector | each plugin (AWS / GH / Slack / JWT / private key); high-entropy gate; off-by-default | unit |
| PresidioDetector | mocked analyzer returns spans; timeout via `monkeypatch`; missing-dep import-skip | unit |
| Span resolver | overlap-longest-wins; tie-break by source priority; adjacent same-label | unit, hypothesis |
| Masker | right-to-left substitution; `mask_format` template; unicode NFC | unit |
| PIIRedactor | walks Anthropic body shape; walks OpenAI body shape; only role=user; only text blocks; no-op when disabled; fail-open on detector exception | integration |
| Pipeline ordering | PIIRedactor runs before cache_aligner (verified via spy) | integration |
| Routes | `/anthropic/pii/v1/messages` forces filter on even when env off; `/openai/pii/v1/chat/completions` same; regular routes untouched when env off | route-level (httpx AsyncClient) |
| Headers | `X-Headroom-PII: on/off` overrides env; `X-Headroom-PII-Min-Score=0.95` raises threshold; per-request `X-Headroom-PII-Labels=PERSON` restricts | route-level |
| Precedence | route > header > env (table-driven) | route-level |
| Failure | each detector raises → request still 200, no detector spans applied; all detectors fail → original body forwarded | route-level with monkey-patched detectors |
| Telemetry | counters increment per label/source; histogram observed; no PII values in log records | unit (caplog) |
| Loopback | masked body passes loopback_guard | integration |
| Optional dep absent | install without `[pii]` → routes still register; `presidio_enabled=False`; regex+secrets work | tox env |
| Large message | 2 MB text → skipped per cap, request succeeds | unit |
| Mass-leakage golden | 50 known-PII strings; assert ≥95% redacted with default config | golden test |

### 9.3 Eval / benchmark

`benchmarks/pii_benchmark.py` measures p50/p95/p99 added latency on 1000-message corpus; per-detector breakdown; comparison of regex-only vs regex+secrets vs full hybrid.

### 9.4 CI gate

- `pii` extra installed in one CI matrix cell (full coverage).
- Other cells run regex+secrets only (no Presidio).
- `pytest -k pii` part of default suite.

### 9.5 Manual smoke

```bash
curl -sX POST http://localhost:8000/openai/pii/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4","messages":[{"role":"user",
       "content":"My SSN is 123-45-6789 and key AKIAIOSFODNN7EXAMPLE"}]}'
# expect upstream sees: [US_SSN] and [AWS_ACCESS_KEY]
```

## 10. Open questions / follow-ups (out of scope for v1)

- Tool-result scanning (would require expanding scope decision).
- Reverse-substitution / round-trip preservation.
- Streaming response inspection.
- Per-tenant / per-API-key configuration overrides.
- Custom regex pattern injection via env / config file.
- Alternative ML backend (OpenMed, Nemotron) behind same `Detector` protocol.

## 11. Migration / rollout

- Default `enabled=False` → no behavior change for existing users.
- Add `[pii]` extra to release notes; document install command.
- New env vars and headers documented in `README.md` proxy section.
- Mirrored routes documented with curl example.
- Telemetry counters documented in metrics section.
