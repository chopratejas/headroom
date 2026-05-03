# PII / Secret Filter for Headroom Proxy — Design

**Date:** 2026-05-03
**Status:** Draft v2 (security-review pass applied; awaiting user approval)
**Owner:** TBD
**Related:** `headroom/proxy/handlers/{anthropic,openai,gemini}.py`, `headroom/proxy/server.py`, `headroom/config.py`

---

## 1. Goal

Detect and mask personally identifiable information (PII) and credentials/secrets in user-supplied request bodies **before** they are forwarded to upstream LLM providers (Anthropic, OpenAI, etc).

The filter is opt-in via three independent mechanisms:

1. **Environment variable** — `HEADROOM_PII_ENABLED=true` (default off).
2. **Per-request header** — `X-Headroom-PII: on` (escalate only — see §6.1).
3. **Dedicated mirrored route** — `POST /openai/pii/v1/chat/completions`, `POST /anthropic/pii/v1/messages`, `POST /openai/pii/v1/responses` — same request/response shape as the regular routes; filter is forced on; **forced routes fail-closed** (§8.1).

Confidence threshold for ML-based detection is configurable globally and per-label.

## 2. Non-goals

- Reverse-substitution / round-tripping the original PII back into the upstream response.
- Scanning assistant responses, tool results, system prompts, image/audio bytes, or tool-call argument JSON.
- Compliance certification (HIPAA / GDPR DPA). The filter is a defence-in-depth control, not a guarantee.
- Streaming-response inspection.
- Built-in pseudonymization / format-preserving fake-data generation.
- Hard wall-clock timeouts on detector execution (Python cannot kill threads). Soft budget only in v1; subprocess-based hard timeout deferred to v2.

## 3. Locked design decisions

| Axis | Decision |
|---|---|
| Detection backend | Hybrid: Microsoft **Presidio** + custom **regex** detector + **detect-secrets** |
| Masking strategy | **Tag redact** — replace with `[<LABEL>]` (e.g., `[PERSON]`, `[EMAIL]`, `[AWS_ACCESS_KEY]`) |
| Default behavior | **Off**; opt-in via env / header (escalate-only) / mirrored route |
| Scan scope | **User messages only** (`role == "user"` text blocks). System / assistant / tool_result / tool_use untouched. Documented limitation: PII echoed in tool results is not caught. |
| Failure mode | **Two-tier:** env/header opt-in → **fail-open**; forced `/pii/...` route → **fail-closed** (HTTP 503). Configurable via `failure_mode_default` and `failure_mode_forced`. |
| Confidence | Global default `min_score=0.85`; per-label override map; per-request header overrides gated behind `HEADROOM_PII_ALLOW_HEADER_OVERRIDES=false` (default off). |
| Integration point | **Pre-pipeline stage** invoked by handler immediately after JSON parse, **before** cache lookup, optimize gate, license gate, hooks, and `TransformPipeline`. PII redactor is **not** a `Transform` subclass. |
| Response path | Untouched (one-way mask) |
| Oversized text | **Chunk-scan** (256KB windows, 256-char overlap). No per-message skip. If chunk-scan itself fails on a forced route → fail-closed; on env/header path → fail-open with WARN. |

## 4. Architecture

```
                ┌─────────────────────────────────┐
client ────►    │  FastAPI route                  │
                │   /v1/messages                  │
                │   /v1/chat/completions          │
                │   /v1/responses                 │
                │   /openai/pii/v1/chat/...   ──► force_pii=True
                │   /anthropic/pii/v1/messages──► force_pii=True
                │   /openai/pii/v1/responses  ──► force_pii=True
                └──────────────┬──────────────────┘
                               │ raw bytes
                               ▼
                ┌─────────────────────────────────────────┐
                │  Handler entry (anthropic/openai/...)   │
                │   1. read body bytes                    │
                │   2. parse JSON                         │
                │   3. resolve_pii_decision(req,cfg)  ◄── escalate-only header
                │   4. await PIIRedactor.run(body, mode)  │ ◄── PRE-PIPELINE
                │      ├─ on success → continue           │
                │      ├─ on failure + forced → 503       │
                │      └─ on failure + opt-in → forward   │
                │   5. cache lookup (uses redacted body)  │
                │   6. optimize gate / license gate       │
                │   7. TransformPipeline.apply(...)       │
                │   8. forward to upstream                │
                └─────────────────────────────────────────┘
                               │
                               ▼
                       upstream LLM
```

**Key invariants**

- PII redactor runs **before** every other gate. Cache hits, `optimize=False`, `_bypass=True`, `_license_ok=False`, and `pre_compress` hooks all see the **already-redacted** body.
- Redactor is a no-op only if `PIIDecision.run == False` (env off, header absent, route not forced).
- Forced routes (`/{provider}/pii/...`) propagate `failure_mode="closed"` — any unrecoverable error returns HTTP 503 to the client. Env/header opt-in propagates `failure_mode="open"` — errors log WARN and forward unmasked.
- Only `role == "user"` text blocks are mutated.

## 5. Components

### 5.1 New module tree

```
headroom/pii/
├── __init__.py
├── redactor.py        # PIIRedactor — pre-pipeline entry; NOT a Transform
├── decision.py        # resolve_pii_decision(request, config) -> PIIDecision
├── walker.py          # body-shape walkers: anthropic / openai_chat / openai_responses
├── chunker.py         # 256KB windows w/ 256-char overlap; merge spans across chunks
├── async_runner.py    # bounded ThreadPoolExecutor + asyncio.wait_for harness
├── detectors/
│   ├── __init__.py
│   ├── base.py        # Detector ABC + Span dataclass
│   ├── presidio.py    # PresidioDetector — wraps presidio-analyzer
│   ├── regex.py       # RegexDetector — email/phone/SSN/IBAN/card/IP/MRN/...
│   └── secrets.py     # SecretDetector — wraps detect-secrets plugins
├── masker.py          # tag_redact(spans, text) -> str
└── span.py            # @dataclass Span(start, end, label, score, source)
```

Lives at `headroom/pii/` (top-level), **not** under `headroom/transforms/`, to make it clear PII is not part of the transform pipeline.

### 5.2 Surgical edits to existing files

| File | Change |
|---|---|
| `headroom/config.py` | Add `PIIConfig` dataclass; embed as `HeadroomConfig.pii: PIIConfig`. (Existing top-level config class is `HeadroomConfig`.) |
| `headroom/proxy/server.py` | `_pii_config_from_env()` helper; register mirrored `/openai/pii/...` and `/anthropic/pii/...` routes that set `request.state.force_pii = True` and delegate to the regular handler. |
| `headroom/proxy/handlers/anthropic.py` | At top of `handle_anthropic_messages` (after JSON parse, before cache lookup at ~line 596) call `await self._maybe_redact_pii(request, body)`. |
| `headroom/proxy/handlers/openai.py` | Same pattern in `handle_openai_chat_completions` (before cache return at ~line 314) and in `handle_openai_responses` (after `body` is parsed at ~line 1084). |
| `headroom/proxy/handlers/gemini.py` | Same pattern; route group registered when Gemini handler used. |
| `pyproject.toml` | New optional dependency group `[pii]`. |

The redactor is invoked from a small mixin method on each handler:

```python
async def _maybe_redact_pii(self, request: Request, body: dict) -> None:
    decision = resolve_pii_decision(request, self.config.pii)
    if not decision.run:
        return
    try:
        await self._pii_redactor.run(body, decision)
    except PIIBlocked as exc:        # forced + chunk-scan failed
        raise HTTPException(503, detail=exc.reason) from exc
    except Exception:
        if decision.failure_mode == "closed":
            log.exception("pii: forced route failed; refusing")
            raise HTTPException(503, detail="pii redaction failed")
        log.warning("pii: detector pipeline failed; forwarding unmasked", exc_info=True)
        self._pii_total_failure_counter.inc()
```

### 5.3 Detector contract

Detectors are **synchronous** by design (Presidio's `AnalyzerEngine.analyze`, regex, detect-secrets are all sync). The redactor wraps them in `asyncio.to_thread` + bounded `ThreadPoolExecutor` for the async interface.

```python
@dataclass(frozen=True)
class Span:
    start: int          # char offset (NFC-normalized) in source text
    end: int            # exclusive
    label: str          # PERSON | EMAIL | AWS_ACCESS_KEY | ...
    score: float        # 0.0–1.0; regex/secrets fixed at 1.0
    source: str         # "presidio" | "regex" | "secrets"

class Detector(Protocol):
    name: str
    def detect(self, text: str, lang: str = "en") -> list[Span]: ...
```

### 5.4 Async runner (`async_runner.py`)

```python
class DetectorRunner:
    """Bounded thread pool harness for sync detectors.

    asyncio.wait_for cancels the *await*, but the underlying thread
    keeps running until detect() returns naturally. The bounded pool
    bounds pileup; per-detector concurrency is capped.
    """
    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pii")

    async def run(self, det: Detector, text: str, lang: str, timeout_s: float) -> list[Span]:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._pool, det.detect, text, lang)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            # Future is still running in the pool. Caller proceeds without
            # this detector's spans. The pool's max_workers prevents pileup.
            return []
```

Documented caveat (in code + spec §10): a runaway sync detector occupies a worker until it returns; once `max_workers` is saturated, subsequent requests skip the affected detector immediately.

### 5.5 Composition order inside `PIIRedactor.run(body, decision)`

1. Pick body walker based on route shape (`anthropic_messages` / `openai_chat` / `openai_responses`).
2. For each user-text segment found by walker:
   - If `len(text.encode("utf-8")) > chunk_size`: split into 256KB chunks with 256-char overlap; each chunk processed independently.
   - For each chunk, run detectors in parallel:
     a. `RegexDetector.detect()` (sync, ~µs; bypasses runner).
     b. `SecretDetector.detect()` (sync, ~µs; bypasses runner).
     c. `PresidioDetector.detect()` via `DetectorRunner` (timeout `presidio_timeout_ms`).
   - Drop spans below threshold (`min_score`, with per-label overrides).
   - Resolve overlaps (longest span wins; ties: `secrets > regex > presidio`).
   - For chunked text, translate chunk-local offsets to global, drop spans wholly inside the overlap region of the **next** chunk (deduped by exact `(start, end, label)`).
3. `masker.tag_redact()` walks merged spans right-to-left, substituting `[<LABEL>]`.
4. Mutate the original body in place via the walker's setter.

### 5.6 Lazy loading

- `PresidioDetector` constructed lazily on first request (cold start ~1.5s for spaCy `en_core_web_sm`).
- `RegexDetector` and `SecretDetector` eager (cheap).
- `DetectorRunner` thread pool created at app startup.

## 6. Data flow

### 6.1 Opt-in resolution (escalate-only header)

```python
def resolve_pii_decision(request: Request, cfg: PIIConfig) -> PIIDecision:
    forced = bool(getattr(request.state, "force_pii", False))
    hdr_on = _header_truthy(request.headers, cfg.header_name)  # only True/False/None
    env_on = cfg.enabled

    # ESCALATE ONLY: header `off` is ignored. Header `on` enables when env off.
    run = forced or env_on or hdr_on

    failure_mode = (
        cfg.failure_mode_forced if forced
        else cfg.failure_mode_default
    )

    # Per-request overrides only when ops trusts headers.
    min_score = cfg.min_score
    labels = cfg.enabled_labels
    if cfg.allow_header_overrides:
        ms_hdr = _header_float(request.headers, f"{cfg.header_name}-Min-Score")
        if ms_hdr is not None:
            min_score = max(0.0, min(1.0, ms_hdr))
        lbl_hdr = _header_csv(request.headers, f"{cfg.header_name}-Labels")
        if lbl_hdr is not None:
            labels = set(lbl_hdr)

    return PIIDecision(run=run, forced=forced, failure_mode=failure_mode,
                       min_score=min_score, enabled_labels=labels)
```

**Precedence** (higher = stronger):
- Forced route always wins (forces `run=True`, `failure_mode="closed"`).
- Env-on always wins over header-off (header-off has **no effect**).
- Header-on enables when env is off.

This closes the auth-bypass: an untrusted client cannot disable filtering that ops has enabled by env.

### 6.2 Anthropic body shape (`POST /v1/messages`) — `walker.anthropic_messages`

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

### 6.3 OpenAI Chat Completions (`POST /v1/chat/completions`) — `walker.openai_chat`

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

### 6.4 OpenAI Responses API (`POST /v1/responses`) — `walker.openai_responses`

The Responses API uses `input` and `instructions` instead of `messages`. Walker handles three sub-shapes (matches existing `responses_items_to_messages` helper at `openai.py:1099`):

```jsonc
{
  "model": "...",
  "instructions": "...",                    // SKIPPED (system per scope)
  "input": "string"                         // ← scan (treated as user text)
}

// OR

{
  "input": [
    { "role": "user",      "content": "string OR [parts]" },           // ← scan
    { "role": "assistant", "content": "..." },                         // skip
    { "type": "input_text", "text": "scan THIS" },                     // ← scan (Responses item form)
    { "type": "input_image", "image_url": "..." },                     // skip
    { "type": "function_call_output", "output": "..." }                // skip (per scope)
  ]
}
```

Walker rules:
- `input` is `str` → scan as one user text segment.
- `input` is `list` → for each item:
  - `dict` with `role == "user"` → scan its `content` (str or parts list, same as Chat Completions).
  - `dict` with `type == "input_text"` and `text` is `str` → scan `text`.
  - All other items skipped.

### 6.5 Walk algorithm (defensive, with type guards)

```python
def walk_chat_messages(body: dict) -> Iterator[TextRef]:
    """Yield mutable refs to user-text segments."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    for msg in msgs:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            yield TextRef(get=lambda m=msg: m["content"],
                          set=lambda v, m=msg: m.__setitem__("content", v))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                yield TextRef(get=lambda b=block: b["text"],
                              set=lambda v, b=block: b.__setitem__("text", v))
```

`TextRef` is a tiny `(get, set)` pair so the redactor can mutate values in place after redaction. All walker callers use `.get()` + `isinstance()` — no key-error or attribute-error path.

### 6.6 Per-segment redaction (with chunking)

```python
async def redact_segment(self, text: str, decision: PIIDecision) -> str:
    text = unicodedata.normalize("NFC", text)
    if not text or not text.strip():
        return text

    chunks = list(chunker.split(text, size=self.cfg.chunk_size_bytes,
                                overlap=self.cfg.chunk_overlap_chars))
    all_spans: list[Span] = []
    for chunk in chunks:
        spans = await self._detect_chunk(chunk.text, decision)
        for s in spans:
            all_spans.append(Span(s.start + chunk.offset, s.end + chunk.offset,
                                  s.label, s.score, s.source))

    all_spans = self._dedupe_overlap_region(all_spans, chunks)
    all_spans = self._drop_below_threshold(all_spans, decision)
    all_spans = self._resolve_overlaps(all_spans)
    masked = self._mask_right_to_left(text, all_spans)
    self._record_telemetry(all_spans)
    return masked
```

**Single-detector errors are absorbed inside `_detect_chunk`** — each detector call is wrapped in try/except (timeout returns `[]` from `DetectorRunner`; raises are caught and counted via `pii_detector_errors_total`). `_detect_chunk` only re-raises (as `PIIBlocked`) when **every** enabled detector for that chunk failed (raised or returned no spans due to error, not zero-find). On `decision.failure_mode == "closed"`: `PIIBlocked` propagates → handler returns HTTP 503. On `failure_mode == "open"`: caller logs WARN, returns original text, increments `pii_total_failure_total`.

### 6.7 Response path

Untouched. Tag-redact is one-way; no reverse substitution. Upstream response returned to client unmodified.

### 6.8 Telemetry

- Counter: `headroom_pii_redactions_total{label, source, route}`
- Counter: `headroom_pii_detector_errors_total{detector}`
- Counter: `headroom_pii_detector_timeouts_total{detector}`
- Counter: `headroom_pii_total_failure_total{forced}` — split forced-vs-opt-in
- Counter: `headroom_pii_blocked_total{route}` — forced route 503s
- Histogram: `headroom_pii_latency_ms{stage}` (regex / secrets / presidio / total)
- Gauge: `headroom_pii_pool_active{}` — DetectorRunner busy workers
- INFO log per request: `pii: redacted=N labels=[...] dur=Xms forced=bool` (counts only)

**Hard rule:** original PII values are never written to logs, never returned to the client, never persisted.

## 7. Config surface

### 7.1 `PIIConfig` dataclass

```python
@dataclass
class PIIConfig:
    # Master env switch (env-on == default-on for all non-/pii routes)
    enabled: bool = False

    # Detectors (independently toggleable)
    regex_enabled: bool = True
    secrets_enabled: bool = True
    presidio_enabled: bool = True            # auto-disabled if dep missing

    # Confidence
    min_score: float = 0.85
    label_scores: dict[str, float] = field(default_factory=dict)

    # Label allow/deny
    enabled_labels: set[str] = field(default_factory=set)   # empty = all
    disabled_labels: set[str] = field(default_factory=set)

    # Mask format
    mask_format: str = "[{label}]"

    # Async runner
    presidio_timeout_ms: int = 200
    detector_pool_workers: int = 4

    # Language
    language: str = "en"

    # Header opt-in
    header_name: str = "X-Headroom-PII"
    allow_header_overrides: bool = False     # min_score / labels via header — TRUSTED ENVS ONLY

    # Failure modes
    failure_mode_default: Literal["open", "closed"] = "open"   # env/header opt-in
    failure_mode_forced: Literal["open", "closed"] = "closed"  # /pii/* routes

    # Chunking
    chunk_size_bytes: int = 262_144          # 256 KB per chunk
    chunk_overlap_chars: int = 256           # boundary overlap

    # Optional noisy detectors
    high_entropy_secrets: bool = False
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
| `HEADROOM_PII_PRESIDIO_TIMEOUT_MS` | `200` | Soft async deadline |
| `HEADROOM_PII_DETECTOR_POOL_WORKERS` | `4` | ThreadPoolExecutor cap |
| `HEADROOM_PII_LANGUAGE` | `en` | Presidio nlp lang |
| `HEADROOM_PII_HEADER_NAME` | `X-Headroom-PII` | Override if proxy strips |
| `HEADROOM_PII_ALLOW_HEADER_OVERRIDES` | `false` | Trust client `-Min-Score` / `-Labels` headers |
| `HEADROOM_PII_FAILURE_MODE_DEFAULT` | `open` | env/header opt-in failure mode |
| `HEADROOM_PII_FAILURE_MODE_FORCED` | `closed` | `/pii/*` route failure mode |
| `HEADROOM_PII_CHUNK_SIZE_BYTES` | `262144` | 256 KB |
| `HEADROOM_PII_CHUNK_OVERLAP_CHARS` | `256` | |
| `HEADROOM_PII_HIGH_ENTROPY` | `false` | Generic high-entropy detector |

### 7.3 Per-request headers

| Header | Values | Effect | Trust |
|---|---|---|---|
| `X-Headroom-PII` | `on` / `1` / `true` | **Escalate** — enable when env off | Always honored |
| `X-Headroom-PII` | `off` / `0` / `false` | **Ignored** — cannot disable env-on filtering | Never honored |
| `X-Headroom-PII-Min-Score` | float `0`–`1` | Override `min_score` for this request | Only when `allow_header_overrides=true` |
| `X-Headroom-PII-Labels` | CSV | Restrict scan to these labels | Only when `allow_header_overrides=true` |

**Security note:** `allow_header_overrides` defaults to `false` because client-controlled threshold/label restrictions are an attack vector — an untrusted client could send `X-Headroom-PII-Min-Score: 1.0` and effectively disable detection. Operators must explicitly opt in for trusted clients (e.g., internal services behind authenticated edge).

### 7.4 Mirrored routes (filter forced on, fail-closed)

| Method | Path | Forwards to |
|---|---|---|
| POST | `/anthropic/pii/v1/messages` | `/v1/messages` |
| POST | `/openai/pii/v1/chat/completions` | `/v1/chat/completions` |
| POST | `/openai/pii/v1/responses` | `/v1/responses` (Responses API walker) |

(Gemini route added when Gemini handler is in active use; same prefix pattern: `/gemini/pii/...`.)

**Implementation:** mirrored routes are registered as additional FastAPI routes whose handler functions set `request.state.force_pii = True` and then delegate to the same handler function as the regular route. No internal path rewrite or HTTP redirect — the same code path runs with one extra flag set. Forced routes pin `failure_mode="closed"` regardless of `failure_mode_default`.

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

### 8.1 Failure matrix

`forced` column = request hit a `/{provider}/pii/...` route. `Response` reflects `failure_mode_default=open` and `failure_mode_forced=closed` defaults; both configurable.

| Failure | Forced? | Behavior | Telemetry |
|---|---|---|---|
| `presidio-analyzer` not installed | — | WARN once at startup; `presidio_enabled=False` | startup log |
| spaCy model `en_core_web_sm` missing | — | WARN once; `presidio_enabled=False` | startup log |
| `PresidioDetector.detect()` raises | no | drop Presidio spans; keep regex+secrets; continue | `pii_detector_errors_total{detector="presidio"}` |
| `PresidioDetector.detect()` raises | yes | as above (single detector failure does not abort if other detectors succeed) | counter |
| `PresidioDetector.detect()` exceeds `presidio_timeout_ms` | — | `asyncio.wait_for` cancels the wait; thread keeps running until detect() returns; drop Presidio spans; continue | `pii_detector_timeouts_total` |
| `RegexDetector` raises | — | drop regex spans; continue | counter |
| `SecretDetector` raises | — | drop secrets spans; continue | counter |
| **All** detectors fail on a chunk | no | log WARN; forward original chunk text + increment counter | `pii_total_failure_total{forced="false"}` |
| **All** detectors fail on a chunk | yes | raise `PIIBlocked` → HTTP 503 | `pii_total_failure_total{forced="true"}` + `pii_blocked_total{route}` |
| Walker raises (malformed body shape) | — | log WARN; no-op for unknown segments; downstream returns its own validation error | none |
| `messages` / `input` field missing or wrong type | — | walker yields nothing; redactor is a no-op | none |
| `min_score` env un-parseable | — | use default; WARN at startup | startup log |
| ThreadPoolExecutor saturated | — | new requests skip the saturated detector immediately (its `wait_for` returns `[]` after timeout, but with all workers busy the `asyncio.wait_for` clock starts only once a worker frees) — to bound this, `detector_pool_workers` should be ≥ expected concurrent request count for ML detector | gauge `pii_pool_active` |

### 8.2 Edge cases

- **Overlapping spans.** Resolver picks longest; ties → `secrets > regex > presidio`. E.g., `john@acme.com` masked as `[EMAIL]`, not `[PERSON]@acme.com`.
- **Adjacent spans, same label.** Presidio model already aggregates BIO tags; no merge step needed.
- **Unicode / multi-byte.** Normalize to NFC at entry; offsets are Python str indices (code points); right-to-left substitution preserves earlier offsets. Chunker splits on code-point boundaries (never mid-grapheme).
- **Empty / whitespace-only text.** Skip detector calls.
- **Oversized message — chunk-scan, no skip.** Default 256KB chunk, 256-char overlap. Span entirely inside the overlap region of the *next* chunk is deduped. Spans straddling a chunk boundary land in both chunks; dedupe by `(start_global, end_global, label)`. Worst case: a span longer than `chunk_overlap_chars` may be split — for v1, masker still tags both halves (e.g., two adjacent `[PERSON]` runs) which is acceptable degradation. Document increasing `chunk_overlap_chars` for workloads with very long entities.
- **Streaming requests.** Filter runs on full request body before upstream call; chat-completion/responses request bodies are non-streaming. No streaming concern.
- **Loopback.** `proxy/loopback_guard.py` runs after filter; unaffected.
- **Cache aligner / dedup.** Filter runs before all caching (semantic_cache lookup, cache_aligner, prefix freeze). Cache hits store and read **redacted** content; no leak via cache.
- **Tool args containing user PII.** Per scope (user msgs only), tool_use args **not** scanned. Documented limitation.
- **False positives mangling code.** A user asking about a literal AWS key gets it masked. Workaround: `disabled_labels=AWS_ACCESS_KEY` in env (per-request label restriction is gated behind `allow_header_overrides`).
- **High-entropy detector.** Off by default. When enabled: `entropy >= 4.5` AND `length >= 20` AND not part of recognized URL/UUID/SHA.
- **Health / metrics endpoints.** Filter only runs inside provider handlers; `/health`, `/metrics`, `/docs` unaffected.
- **Mirrored route + opt-out header.** `/openai/pii/v1/...` with `X-Headroom-PII: off` → route flag wins (forced on, header `off` is always ignored). Documented behavior.
- **Cache-hit replay attack defence.** Cache key is computed over redacted body; an attacker cannot prime cache with PII-bearing input and then read it back, because the stored key is the masked form.
- **Pipeline bypass paths covered.** Because PII runs before `if self.config.optimize and messages and not _bypass and _license_ok:`, all of: `optimize=False`, `_bypass=True`, `_license_ok=False`, cache-mode skip, `pre_compress` hook bypass, and early 4xx returns still see redacted bodies.

## 9. Testing

### 9.1 Test layout

```
tests/pii/
├── test_redactor.py
├── test_decision.py                 # opt-in resolution; escalate-only header
├── test_walker_anthropic.py
├── test_walker_openai_chat.py
├── test_walker_openai_responses.py
├── test_chunker.py                  # boundary + overlap + dedupe
├── test_async_runner.py             # timeout, pool saturation
├── test_regex_detector.py
├── test_secrets_detector.py
├── test_presidio_detector.py        # @pytest.mark.optional_dep("presidio")
├── test_masker.py
├── test_span_resolver.py
└── fixtures/
    ├── anthropic_messages.json
    ├── openai_chat.json
    ├── openai_responses.json
    └── pii_corpus.txt
tests/proxy/
├── test_pii_routes.py               # mirrored /{provider}/pii/* routes
├── test_pii_header_opt_in.py        # escalate-only behavior
├── test_pii_failure_modes.py        # forced=closed, opt-in=open
└── test_pii_pre_pipeline_ordering.py # cache hit / optimize=False / bypass / license=False all see redacted body
```

### 9.2 Coverage matrix

| Layer | What | How |
|---|---|---|
| Span | offsets, equality, ordering | unit |
| Decision | escalate-only header (`off` ignored when env on); forced wins over env-off; `allow_header_overrides=false` rejects `-Min-Score` / `-Labels`; `=true` accepts and clamps | unit, table-driven |
| Walker (Anthropic) | string content; blocks list; non-dict block; missing `text` key; non-string `text` value; non-list `messages` | unit |
| Walker (OpenAI Chat) | string content; parts list; non-text parts skipped; non-dict parts skipped | unit |
| Walker (Responses) | `input` as str; `input` as list with role=user; `input` as list with `type=input_text`; `instructions` skipped | unit |
| Chunker | segment <chunk_size: one chunk; ≥chunk_size: N chunks with overlap; spans dedupe across overlap; spans crossing boundary appear in both chunks | unit, hypothesis |
| AsyncRunner | timeout returns `[]`, thread keeps running but is reaped on completion; pool saturation falls through to `[]`; `max_workers` honored | unit |
| RegexDetector | each label hits its pattern; no FP on near-misses; Luhn on cards; IBAN checksum; AWS prefix | unit, table-driven |
| SecretDetector | each plugin (AWS / GH / Slack / JWT / private key); high-entropy gate; off-by-default | unit |
| PresidioDetector | mocked analyzer returns spans; timeout via `monkeypatch`; missing-dep import-skip | unit |
| Span resolver | overlap-longest-wins; tie-break by source priority; adjacent same-label | unit, hypothesis |
| Masker | right-to-left substitution; `mask_format` template; unicode NFC | unit |
| PIIRedactor | walks all 3 body shapes; only role=user; only text blocks; no-op when decision.run=False | integration |
| Pre-pipeline ordering | cache-hit path: redacted body is what cache stores/reads; `optimize=False`: still redacted; `_bypass=True`: still redacted; `_license_ok=False`: still redacted | integration (monkey-patch each gate) |
| Routes | `/{provider}/pii/...` forces filter on even when env off; regular routes untouched when env off; Responses route walks `input` not `messages` | route-level (httpx AsyncClient) |
| Headers (escalate-only) | `on` enables when env off; `off` is ignored when env on; `off` is no-op when env off | route-level |
| Header overrides gate | `allow_header_overrides=false` ignores `-Min-Score`; `=true` honors and clamps to [0,1] | route-level |
| Failure modes | env-on, all detectors patched to raise → 200 + unmasked + counter; forced route, all detectors patched to raise → 503 + counter | route-level |
| Padding-attack regression | request with 5 MB body containing AWS key at offset 4_500_000 → key is masked (chunker covers it); no skip | route-level |
| Telemetry | counters increment per label/source/forced; histogram observed; `pii_blocked_total` increments on forced-route 503; **no PII values in log records** | unit (caplog) |
| Loopback | masked body passes loopback_guard | integration |
| Optional dep absent | install without `[pii]` → routes still register; `presidio_enabled=False`; regex+secrets work | tox env |
| Mass-leakage golden | 50 known-PII strings; assert ≥95% redacted with default config | golden test |

### 9.3 Eval / benchmark

`benchmarks/pii_benchmark.py` measures p50/p95/p99 added latency on a 1000-message corpus; per-detector breakdown; comparison of regex-only vs regex+secrets vs full hybrid; chunk-scan overhead at 256KB / 1MB / 5MB segment sizes.

### 9.4 CI gate

- `pii` extra installed in one CI matrix cell (full coverage).
- Other cells run regex+secrets only (no Presidio).
- `pytest -k pii` part of default suite.
- New job: `pytest tests/proxy/test_pii_pre_pipeline_ordering.py` — gates merging by verifying redaction precedes every pipeline-bypass path.

### 9.5 Manual smoke

```bash
# Forced route, fail-closed
curl -sX POST http://localhost:8000/openai/pii/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4","messages":[{"role":"user",
       "content":"My SSN is 123-45-6789 and key AKIAIOSFODNN7EXAMPLE"}]}'
# expect upstream sees: [US_SSN] and [AWS_ACCESS_KEY]

# Responses API forced route
curl -sX POST http://localhost:8000/openai/pii/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4.1","input":"Email me at jane@acme.com"}'
# expect upstream sees: input="Email me at [EMAIL]"

# Padding attack regression
python - <<'PY'
import json, requests
pad = "x" * 5_000_000
body = {"model":"gpt-4","messages":[{"role":"user",
        "content": pad + " AKIAIOSFODNN7EXAMPLE"}]}
r = requests.post("http://localhost:8000/openai/pii/v1/chat/completions",
                  json=body)
print(r.status_code, r.text[:200])
PY
# AWS key MUST be masked in upstream view
```

## 10. Open questions / follow-ups (out of scope for v1)

- Tool-result scanning (would require expanding scope decision).
- Reverse-substitution / round-trip preservation.
- Streaming response inspection.
- Per-tenant / per-API-key configuration overrides.
- Custom regex pattern injection via env / config file.
- Alternative ML backend (OpenMed, Nemotron) behind same `Detector` protocol.
- **Hard wall-clock detector timeouts** via `multiprocessing.Process` (currently soft only; thread keeps running until detector returns).
- Sub-second authenticated rate-limit on forced `/pii/*` routes (NER is expensive — rate-limit per API key to deter abuse).

## 11. Migration / rollout

- Default `enabled=False` and `failure_mode_default="open"` → no behavior change for existing users.
- Add `[pii]` extra to release notes; document install command.
- New env vars and headers documented in `README.md` proxy section.
- Mirrored routes documented with curl example **and explicit fail-closed warning**.
- Telemetry counters (including `pii_blocked_total`) documented in metrics section.
- Operators enabling env-default redaction must size `detector_pool_workers` to expected concurrent ML-detection load (default 4 is fine for ≤4 concurrent requests; bump for higher).

## 12. Security review trail (v1 → v2)

This v2 spec applies fixes for the following findings raised during security review:

| ID | Finding | Resolution |
|---|---|---|
| C1 | PII as first pipeline transform skipped on cache-hit / `optimize=False` / `_bypass` / `_license_ok=False` | Moved out of `TransformPipeline`. PII now runs in handler entry, before all gates. (§4, §5.2) |
| C2 | Client header `off` could disable env-enabled filtering = auth bypass | Header is **escalate-only**. Header `off` is always ignored. Score/label header overrides gated behind `allow_header_overrides=false` (default). (§6.1, §7.3) |
| C3 | Per-message size cap allowed padding attack to skip filter on >1MB messages | Cap removed. Replaced with chunk-scan (256KB windows, 256-char overlap). On forced route + chunk-scan failure → fail-closed. (§5.5, §6.6, §7.1) |
| C4 | `/openai/pii/v1/responses` listed but walker only handled `messages` | Added `walker.openai_responses` covering `input` (str / list, including Responses API item shapes) and `instructions` (skipped per scope). (§5.1, §6.4) |
| W5 | Sync detector + `asyncio.wait_for` doesn't actually cancel sync work | New `DetectorRunner` with bounded `ThreadPoolExecutor`. `wait_for` cancels the await; thread runs to completion. Saturation bounded by `max_workers`. Documented caveat; hard timeout deferred. (§5.4, §10) |
| W6 | `block["text"]` had no key / type guard | Walker uses `.get()` + `isinstance()` for every level. (§6.5) |
| W7 | Forced `/pii/*` routes still fail open on total detector failure | Forced routes default to `failure_mode_forced="closed"` → HTTP 503. Configurable. (§7.1, §8.1) |
| I8 | Spec referenced non-existent `Transform.run(text)`, `RequestContext.flags`, `ProxyConfig` | Aligned to actual `Transform.apply(messages, tokenizer, **kwargs) -> TransformResult` and `HeadroomConfig`. PII is no longer a `Transform` subclass at all. (§3, §5.2) |
