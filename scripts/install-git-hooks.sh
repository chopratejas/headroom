#!/usr/bin/env bash
# Install a pre-push git hook that runs `make ci-precheck` before every push.
#
# Why: the 2026-04-27 push hit five CI failures that could all have been
# caught locally — cargo fmt drift, an x86_64-apple-darwin wheel that the
# project doesn't actually need, missing Rust extension in two CI lanes,
# and a commitlint warning treated as an error. The fixes are committed;
# this hook ensures we don't repeat the same dance.
#
# Idempotent. Re-running is safe — it overwrites the hook file with the
# current desired contents. Skips installation if `.git/hooks/` is missing
# (e.g. running outside a git checkout).

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .git/hooks ]]; then
    echo "error: .git/hooks/ not found — run from a git checkout root" >&2
    exit 1
fi

HOOK_PATH=".git/hooks/pre-push"

cat > "$HOOK_PATH" <<'HOOK_EOF'
#!/usr/bin/env bash
# Headroom pre-push hook — runs `make ci-precheck` so CI never finds a
# bug a local check could have caught.
#
# Skip with: `git push --no-verify`. Use sparingly — every skip is a roll
# of the dice on a CI break.

set -euo pipefail

# Skip the hook entirely when push goes to a ref that is not on the main
# tracking branches we gate. Adjust the pattern below if more branches
# need gating.
remote="$1"
url="$2"

while IFS=' ' read -r local_ref local_sha remote_ref remote_sha; do
    # Empty local_sha means a delete; nothing to verify.
    if [[ "$local_sha" == "0000000000000000000000000000000000000000" ]]; then
        continue
    fi
    echo "── pre-push: running 'make ci-precheck' before pushing $local_ref → $remote_ref"
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "warn: no VIRTUAL_ENV set and no .venv/ found — python checks may use the wrong interpreter" >&2
    fi
fi

if make ci-precheck; then
    exit 0
else
    echo ""
    echo "❌ pre-push: 'make ci-precheck' failed. Fix the issues above before pushing."
    echo "   To bypass (NOT recommended): git push --no-verify"
    exit 1
fi
HOOK_EOF

chmod +x "$HOOK_PATH"

echo "✅ installed: $HOOK_PATH"
echo "   Runs 'make ci-precheck' before every git push."
echo "   Bypass (use sparingly): git push --no-verify"
