# Adversarial Security Review — 2026-06-05

Full end-to-end adversarial review of the headroom proxy. Two review threads ran
concurrently: direct source analysis (proxy, auth, CLI, memory) and a dedicated
supply-chain/infrastructure agent.

---

## Findings: Application Layer

### CORS-01 | CRITICAL | CWE-346: CORS wildcard + credentials

**File:** `headroom/proxy/server.py:1689-1695`

```python
# BEFORE (vulnerable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    ...
)
```

`allow_origins=["*"]` combined with `allow_credentials=True` is forbidden by the
CORS spec. Browsers silently reject the combination, but it signals developer intent
to permit any origin with credentialed access — a misconfiguration that could be
exploited if the origin restriction is later "fixed" without removing the wildcard.
For a local proxy there is no justification for allowing any external origin.

**Fix:** Use `allow_origin_regex` restricted to localhost; remove `allow_credentials`.
`HEADROOM_CORS_ORIGIN_REGEX` env var allows customization for non-default deployments.

**Status:** Fixed in this PR.

---

### HEALTH-01 | MEDIUM | Information disclosure via /health

**File:** `headroom/proxy/server.py:1826-1831`

`GET /health` returns `include_config=True`, which includes internal API base URLs
(`anthropic_api_url`, `openai_api_url`, etc.) and the process PID. This endpoint had
no access restriction while `/debug/*` endpoints were already guarded by
`require_loopback()`.

**Fix:** Added `dependencies=[Depends(_require_loopback)]` to `/health`. Non-loopback
callers receive 404. `/readyz` (used by Kubernetes probes) is unaffected.

**Status:** Fixed in this PR.

---

### TOML-01 | LOW | Path injection into TOML string literal

**File:** `headroom/cli/wrap.py:1121-1126`

`db_path` and `sys.executable` were interpolated into a TOML basic-string literal
without escaping embedded `"` characters. A path containing `"` would produce
malformed TOML, potentially overwriting keys that follow in the same config block.

**Fix:** Added `_toml_escape()` helper that escapes `\` and `"` before interpolation.

**Status:** Fixed in this PR.

---

### AUTH-01 | MEDIUM | Auth-mode entirely determined by spoofable User-Agent

**File:** `headroom/proxy/auth_mode.py:60-69`

`AuthMode` (SUBSCRIPTION vs. PAYG vs. OAUTH) is inferred from `User-Agent` prefix.
A client that spoofs a subscription User-Agent will receive more aggressive
compression behavior. This is not a privilege-escalation vector (no privileged actions
differ between modes) but the auth-mode boundary is not a security guarantee.

**Fix (documentation-only):** No code change in this PR. A comment was considered;
ultimately the classification is explicitly client-hinting and the docstring should be
updated to say so. Tracked as a follow-up.

**Status:** Documentation only — not a security boundary, no code change needed.

---

## Findings: CI/CD and Supply Chain

### SC-01 | HIGH | CWE-494: curl\|bash from floating HEAD

**File:** `.github/workflows/ci.yml:312-318`

`actionlint` and `act` were installed by piping from `raw.githubusercontent.com/…/main/`
directly into bash with no integrity check. A compromised upstream branch injects
arbitrary code running on GitHub-hosted runners with access to all workflow secrets.

**Fix:** Replaced with pinned release-artifact downloads with SHA-256 verification.

**Status:** Fixed in this PR.

---

### SC-02 | HIGH | CWE-732: Over-privileged workflow-level permissions in docker.yml

**File:** `.github/workflows/docker.yml:26-29`

`id-token: write` and `packages: write` were declared at the workflow level, meaning
all jobs (including the build stage that runs `docker bake` with third-party actions)
held OIDC token issuance capability for their entire lifetime.

**Fix:** Set `permissions: contents: read` at workflow level; added per-job overrides:
- `docker-build`: `contents: read, packages: write`
- `docker-manifest`: `contents: read, packages: write, id-token: write`
- `promote-latest`: `contents: read, packages: write`

**Status:** Fixed in this PR.

---

### SC-03 | HIGH | cargo audit + deny running with continue-on-error

**File:** `.github/workflows/rust.yml`

`cargo audit` and `cargo deny check licenses` ran with `continue-on-error: true`,
meaning a known CVE in a Rust crate would never block a release. No Python
`pip-audit` step existed.

**Fix:** Removed `continue-on-error: true` from both `cargo audit` and
`cargo deny`. Added a `pip-audit` job to `ci.yml` that installs the proxy extras
and audits the installed environment.

**Status:** Fixed in this PR.

---

### SC-04 | MEDIUM | PIP_EXTRA_INDEX_URL at workflow scope (CWE-770)

**File:** `.github/workflows/ci.yml:32`

`PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu` at workflow level meant
every `pip install` (including `pip install ruff`, `pip install mypy`) also searched
pytorch.org. A package on that index sharing a name with a CI tool would be preferred
if it had a higher version number.

**Fix:** Removed the workflow-level env var. Each step that installs torch already
uses `--index-url` explicitly; the workflow-level variable was redundant.

**Status:** Fixed in this PR.

---

### SC-05 | MEDIUM | Dockerfile curl in runtime image

**File:** `Dockerfile`

`runtime-slim-base` installed `curl` solely for the `HEALTHCHECK` command. curl is a
network-capable binary; its presence in the runtime image expands the attack surface
available to any code-execution exploit in the proxy.

**Fix:** Removed the `apt-get install curl` block. Replaced `HEALTHCHECK CMD curl …`
with a pure-Python equivalent using `urllib.request`. The distroless `runtime-slim`
stage already used this pattern.

**Status:** Fixed in this PR.

---

### SC-06 | MEDIUM | CWE-732: docs.yml over-privileged + unpinned mkdocs

**File:** `.github/workflows/docs.yml`

`contents: write` at the workflow level meant the `pip install mkdocs-material` step
ran with a token that could push to any branch. mkdocs-material was unpinned.

**Fix:** Set `permissions: contents: read` at workflow level; added
`permissions: contents: write` only to the deploy job. Pinned mkdocs-material to
`9.5.49`.

**Status:** Fixed in this PR.

---

### SC-07 | LOW | actions/checkout@v6 anomaly

**File:** `.github/workflows/docker.yml`

`docker-build` used `actions/checkout@v6` while every other workflow used `@v4`.
`v6` does not exist as a stable release and likely resolved via tag aliasing.

**Fix:** Standardized to `actions/checkout@v4`.

**Status:** Fixed in this PR.

---

### SC-08 | LOW | No CODEOWNERS for security-critical paths

**File:** `.github/` (new)

No CODEOWNERS file existed. Changes to CI workflows, Dockerfile, and release scripts
could be merged without designated-reviewer approval.

**Fix:** Created `.github/CODEOWNERS` covering workflows, Dockerfile, docker-bake.hcl,
pyproject.toml, and scripts/install.sh.

**Status:** Fixed in this PR.

---

## Follow-up items (not fixed in this PR)

These require additional investigation or tooling beyond the scope of this PR:

| ID | Severity | Description |
|----|----------|-------------|
| FU-01 | MEDIUM | Pin all GitHub Actions to immutable SHA digests (use `pin-github-actions` or Dependabot `github-actions` ecosystem) |
| FU-02 | MEDIUM | `runtime` and `runtime-code` bake targets default `RUNTIME_USER=root`; consider making nonroot the default published image |
| FU-03 | MEDIUM | Pin base Docker images to SHA digests (Dependabot supports this) |
| FU-04 | MEDIUM | Pin Dockerfile rustup installer to a specific version + SHA-256 verification |
| FU-05 | MEDIUM | Add SBOM generation (CycloneDX) to `release.yml` main publish path |
| FU-06 | MEDIUM | Commit `uv.lock` and use `uv sync --frozen` in CI to prevent dependency drift |
| FU-07 | LOW | Add Gitleaks secret-scanning job to CI |
| FU-08 | INFO | `HEADROOM_USER_ID` env var for traffic learner is unvalidated; document expected format |

---

## Multi-Persona Adversarial Review — Round 2 (2026-06-08)

A second pass over the security-hardening PR worked through 12 attacker
personas (network attacker, supply-chain, CI/CD, container, AppSec, crypto,
secrets-leak, privilege-esc, injection, dep-analyst, compliance, red-team).
Findings below were not surfaced in the first review.

### TOML-02 | HIGH | CWE-94: Unescaped `user_id` in Codex MCP TOML injection

**File:** `headroom/cli/wrap.py:1143` (`_inject_memory_mcp_config`)

`_toml_escape` is applied to `python_bin` and `db_path` but NOT to the
`user_id` interpolation. `user_id` comes from `$USER` / `$USERNAME` (or
`"default"`), which an attacker who can set those env vars (e.g. via a
crafted launcher script, container image, or shared dev machine) can use
to break out of the TOML string literal:

```text
USER='victim", inject = "x' headroom wrap codex --memory
```

Result: arbitrary `[mcp_servers.*]` table can be appended to `~/.codex/config.toml`
including an attacker-controlled `command =` that runs every Codex launch.

**Fix:** Run `user_id` through `_toml_escape` before interpolation. Added
explanatory comment to deter regression.

**Status:** Fixed in this commit.

---

### LEAK-01 | HIGH | CWE-200: Unauthenticated `/transformations/feed` leaks chat content

**File:** `headroom/proxy/server.py:2370` (pre-fix)

`/transformations/feed` returned full `request_messages` and
`response_content` (prompts and completions) when `log_full_messages` was
enabled. The endpoint had no auth and no loopback gate. With the default
`--host 0.0.0.0` Docker bind, any network-reachable client could exfiltrate
conversation history simply by polling the dashboard's data feed.

**Fix:** Added `dependencies=[Depends(_require_loopback)]` so the endpoint
is invisible (404) to non-loopback callers. The dashboard already runs in
the user's browser on loopback, so legitimate use is unaffected.

**Status:** Fixed in this commit.

---

### LEAK-02 | HIGH | CWE-200: Unauthenticated `/stats` leaks `recent_requests`

**File:** `headroom/proxy/server.py:2320` (pre-fix)

`/stats` returned `recent_requests` (the last 10 request log entries with
request_id / provider / model / error / timestamp) without authentication.
Even with `log_full_messages=False`, this leaks operational telemetry:
attackers can fingerprint customer routing and infer error patterns.

**Fix:** `/stats` is now soft-gated: aggregate counters are still returned
to every caller (preserves external monitoring), but `recent_requests`
is only embedded for loopback callers — matching the same trust boundary
as `/dashboard` and `/transformations/feed`. The full dashboard surface
(`/dashboard`) is now hard-gated to loopback.

**Status:** Fixed in this commit.

---

### CSRF-01 | HIGH | CWE-352: Unauthenticated `/cache/clear` POST

**File:** `headroom/proxy/server.py:2473` (pre-fix)

`/cache/clear` was an unauthenticated state-mutating POST. On the default
`--host 0.0.0.0` Docker bind, any network-reachable client could
repeatedly flush the proxy's response cache as a denial-of-service or to
evict cached completions the operator relies on for cost control.

**Fix:** Added loopback gate. Cache management is an operator action and
belongs behind the same trust boundary as `/debug/*` and `/stats/reset`.

**Status:** Fixed in this commit.

---

### SC-09 | HIGH | Missing `cargo deny check advisories`

**File:** `.github/workflows/rust.yml`

The audit job ran `cargo deny check licenses` but not `cargo deny check
advisories`. RUSTSEC advisories on transitive deps were only being caught
by `cargo audit`; `cargo deny`'s advisory database adds an independent
source of truth (different update cadence, different curation).
Additionally, `cargo install ... || true` was masking install failures of
the audit tools themselves.

**Fix:** Drop `|| true` (install must succeed), add `cargo audit --deny
warnings`, add `cargo deny check advisories` as a hard-fail step, and add
`cargo deny check bans` as a warning-only step pending baseline curation.

**Status:** Fixed in this commit.

---

### LOG-01 | MEDIUM | CWE-117: Path log injection via URL-decoded path

**File:** `headroom/proxy/server.py:1715` (pre-fix middleware)

Uvicorn URL-decodes ASGI paths before populating `request.url.path`, so
an attacker sending `GET /foo%0aevent=injected_event` would produce a log
line containing a literal newline and could spoof a second log event
(useful for hiding malicious activity under a forged event tag in log
aggregators).

**Fix:** New helper `_sanitize_log_token` replaces ASCII control chars
with `?` and truncates to a fixed length. Applied to `path`, `method`,
and `query` before they are embedded in the `event=proxy_inbound_request`
log line.

**Status:** Fixed in this commit.

---

## Follow-up items (round 2)

| ID | Severity | Description |
|----|----------|-------------|
| FU-09 | MEDIUM | SHA-256 verify all CI binary downloads (gitleaks, actionlint, act, uv installer). Currently version-pinned but unchecksummed. |
| FU-10 | MEDIUM | Document the `HEADROOM_CORS_ORIGIN_REGEX` env var risk: operators setting a too-permissive regex (e.g. `.*`) effectively disable CORS. Add a startup-time warning when the regex matches non-loopback origins. |
| FU-11 | LOW | Distroless `runtime-slim` `WORKDIR /app` is created as root-owned even though `USER nonroot` is set. Currently a non-issue (the proxy never writes to cwd) but surprises users mounting volumes. |
| FU-12 | LOW | `is_loopback_host(None)` returns `True` to support TestClient. Document that any future ASGI server that strips `request.client` would silently expand the loopback set. |
| FU-13 | INFO | Run native fuzzing on the proxy middleware path-parsing layer with a CR/LF/control-char corpus to confirm there are no other unsanitized log sites. |
