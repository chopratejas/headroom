#!/usr/bin/env bash
#
# Refresh the two vendored price/limit snapshots used by the Rust proxy:
#
#   1. LiteLLM model_prices_and_context_window.json
#      → context-window limits  (compression/model_limits.rs)
#   2. models.dev api.json
#      → per-model USD prices    (observability/pricing.rs)
#
# Both are vendored rather than fetched at build/runtime so the proxy binary
# ships with no network dependency at startup. Operators tracking new model
# releases run this script and commit the diff.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/crates/headroom-proxy/data"

LIMITS_DEST="$DATA_DIR/model_prices_and_context_window.json"
LIMITS_URL="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PRICES_DEST="$DATA_DIR/models_dev.json"
PRICES_URL="https://models.dev/api.json"

# Fetch + validate BOTH into tempfiles first, then move both into place only
# after both pass. This keeps the refresh atomic: a failure in step 2 must not
# leave an updated limits file beside a stale prices file (a partial, committable
# diff that defeats "refresh both together"). The tempfile path is passed via
# argv (sys.argv[1]), never interpolated into the Python source, so a TMPDIR with
# special chars can't break the validator.
TMP_LIMITS="$(mktemp)"
TMP_PRICES="$(mktemp)"
trap 'rm -f "${TMP_LIMITS:-}" "${TMP_PRICES:-}"' EXIT

fetch() { echo "Fetching $1"; curl -fsSL "$1" -o "$2"; }

# --- 1. LiteLLM context-window limits ------------------------------------- #
# Validation: JSON parses, has the sample_spec sentinel, and known-stable
# Claude + GPT entries — guards against vendoring an empty / schema-changed file.
fetch "$LIMITS_URL" "$TMP_LIMITS"
python3 - "$TMP_LIMITS" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if not isinstance(data, dict):
    sys.exit('top-level not an object')
if 'sample_spec' not in data:
    sys.exit('missing sample_spec entry — schema may have changed')
required = ['claude-sonnet-4-5-20250929', 'gpt-4o-mini', 'gpt-4-turbo']
missing = [k for k in required if k not in data]
if missing:
    sys.exit(f'missing required entries: {missing!r}')
print(f'OK (limits): {len(data)} entries, including {required}')
PY

# --- 2. models.dev USD prices --------------------------------------------- #
# Validation: JSON parses, nested provider→models→cost shape, and *some* Anthropic
# haiku model carries a positive input price (the field pricing.rs reads). Checks
# any haiku, not the first by dict order, so a zero-priced preview variant doesn't
# spuriously fail the guard.
fetch "$PRICES_URL" "$TMP_PRICES"
python3 - "$TMP_PRICES" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if not isinstance(data, dict):
    sys.exit('top-level not an object')
anthropic = data.get('anthropic', {}).get('models', {})
priced = [
    m for k, m in anthropic.items()
    if 'haiku' in k
    and isinstance((m.get('cost') or {}).get('input'), (int, float))
    and m['cost']['input'] > 0
]
if not priced:
    sys.exit('no Anthropic haiku model with a positive input price — schema may have changed')
print(f'OK (prices): {len(data)} providers, {len(priced)} priced haiku model(s)')
PY

# Both validated — commit the swap.
mv "$TMP_LIMITS" "$LIMITS_DEST"
echo "Updated $LIMITS_DEST"
mv "$TMP_PRICES" "$PRICES_DEST"
echo "Updated $PRICES_DEST"

trap - EXIT
echo "Run 'cargo test -p headroom-proxy --lib observability::pricing compression::model_limits' to verify."
