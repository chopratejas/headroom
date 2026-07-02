# Headroom × GitHub Copilot — Enterprise Test Runbook (Copilot Business)

Goal: prove that **policy-gated premium models flow through Headroom, to the
business host, with compression** — and cleanly tell a Headroom bug apart from a
GitHub entitlement/policy limit.

Everything below is **read-only / secret-free** except the live proxy run. Run on
the machine where Copilot is logged in, from the headroom repo root, in the venv.

---

## Step 0 — Provision (you own the org)

1. Org → **Settings → Copilot → Access**: enable **Copilot Business**, assign yourself a **seat**.
2. Org → **Settings → Copilot → Policies → Models**: **enable every model** (Claude, GPT‑5, Gemini).
   *(This is the #1 gate — premium 400s until these are on.)*
3. On the test machine, log in the **Copilot CLI** (`copilot` → `/login`) so the token lands in the OS store.
4. Host will be `https://api.business.githubcopilot.com` (Step A confirms it).

> **Pass-through note:** Headroom forwards the token your client already holds. SSO/SAML
> stays the client's job, so Business (no org SSO) is enough to prove the core. Add a GHEC
> trial later only if a deal demands SSO validation beyond pass-through.

---

## Step A — Doctor (read-only: where's the key + native entitlement)

```bash
GITHUB_COPILOT_API_URL=https://api.business.githubcopilot.com \
  .venv/bin/python tools/copilot-test/copilot_doctor.py
```

Look for:
- **[4] host type** = `ENTERPRISE / data-residency` *(business host)*; **[5] exchange** should now return a `tid_` (it 404s on unentitled seats).
- **[6] catalog** lists premium models; **[7]** shows `gpt-5.5` / `claude` as `✅` (via `chat` or `responses`).
- If `[7]` is `🔒 403` → SSO authorization needed (re-login via IdP).
- If `[7]` premium is `❌ 400` → models **not enabled in org policy** (Step 0.2).

---

## Step B — Live proxy (premium models THROUGH Headroom + outbound capture)

```bash
GITHUB_COPILOT_API_URL=https://api.business.githubcopilot.com \
  .venv/bin/python tools/copilot-test/enterprise_proxy_test.py
```

Look for:
- `premium through proxy: ✅ PASS`
- Outbound capture lines showing `host=api.business.githubcopilot.com` (Headroom routed correctly).
- If a model worked **only via `/responses`**, the harness flags the **wire-API split** — confirm `headroom wrap` auto-selects it (#644/#647).

---

## Step C — Compression proof (the actual value-add)

Steps A/B use tiny prompts that won't compress. Prove savings on real volume:

```bash
GITHUB_COPILOT_API_URL=https://api.business.githubcopilot.com \
  headroom wrap copilot --subscription -- \
  --model claude-sonnet-4.6 -p "Summarize this file: $(cat <a-large-source-file>)"
# then open the dashboard:
open http://localhost:8787/dashboard
```

Look for token savings attributed to the `copilot` provider on the dashboard.

---

## Decision matrix

| Doctor (native) | Harness (through proxy) | Conclusion | Owner |
|---|---|---|---|
| premium ✅ | premium ✅ | **Headroom supports Copilot Business** — ship it | — |
| premium ✅ | premium ❌ | **Headroom bug** — capture shows host/header/wire-API cause | us |
| premium ❌ (400) | premium ❌ | **Org policy/entitlement** — enable models (Step 0.2) | you |
| `🔒 403` anywhere | — | **SSO** not authorized — re-auth via IdP | you |
| `Credential found: NONE` | — | discovery gap — run where the token lives, or use pass-through | us/you |

---

## What to send back (no secrets)

The full stdout of Step A and Step B (tokens are already redacted to prefixes/kind),
plus `~/.headroom/copilot_outbound.jsonl` (host + headers + token *kind* only).

## Cleanup

```bash
lsof -ti:8911 | xargs kill 2>/dev/null        # stop any test proxy
unset HEADROOM_COPILOT_DEBUG_OUTBOUND          # stop outbound capture
```

---

### Kit contents (currently uncommitted)
- `tools/copilot-test/copilot_doctor.py` — Phase A, read-only diagnostic
- `tools/copilot-test/enterprise_proxy_test.py` — Phase B, live-proxy through-path
- Outbound capture hook lives in `headroom/copilot_auth.py` (env-gated `HEADROOM_COPILOT_DEBUG_OUTBOUND`)
