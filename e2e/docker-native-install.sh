#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HEADROOM_DOCKER_IMAGE:?set HEADROOM_DOCKER_IMAGE to a built test image}"
PROFILE="ci-smoke"
TMP_HOME="$(mktemp -d)"
PORT="$(python3 - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"

cleanup() {
  docker rm -f "headroom-${PROFILE}" >/dev/null 2>&1 || true
  rm -rf "${TMP_HOME}"
}
trap cleanup EXIT

mkdir -p "${TMP_HOME}/.local"
export HOME="${TMP_HOME}"
export PATH="${HOME}/.local/bin:${PATH}"
export HEADROOM_DOCKER_IMAGE="${IMAGE}"

bash "${ROOT_DIR}/scripts/install.sh"

WRAPPER="${HOME}/.local/bin/headroom"
[[ -x "${WRAPPER}" ]]

"${WRAPPER}" install -? | grep -Fq "persistent-docker preset only"

"${WRAPPER}" install apply \
  --profile "${PROFILE}" \
  --port "${PORT}" \
  --image "${IMAGE}" \
  --no-telemetry

status_output="$("${WRAPPER}" install status --profile "${PROFILE}")"
printf '%s\n' "${status_output}"
grep -Fq "Status:     running" <<<"${status_output}"
curl --fail --silent "http://127.0.0.1:${PORT}/readyz" >/dev/null
health_output="$(curl --fail --silent "http://127.0.0.1:${PORT}/health")"

python3 - <<'PY' "${HOME}" "${PROFILE}" "${PORT}" "${health_output}"
import json
import sys
from pathlib import Path

home = Path(sys.argv[1])
profile = sys.argv[2]
port = int(sys.argv[3])
health = json.loads(sys.argv[4])
manifest = json.loads((home / ".headroom" / "deploy" / profile / "manifest.json").read_text())
assert manifest["preset"] == "persistent-docker"
assert manifest["port"] == port
assert manifest["telemetry_enabled"] is False
assert health["deployment"]["profile"] == profile
assert health["deployment"]["preset"] == "persistent-docker"
assert health["deployment"]["runtime"] == "docker"
PY

if apply_error="$("${WRAPPER}" install apply --scope user 2>&1)"; then
  echo "expected docker-native install apply --scope user to fail" >&2
  exit 1
fi
grep -Fq "does not support provider/user/system mutation flags" <<<"${apply_error}"

"${WRAPPER}" install stop --profile "${PROFILE}"
stopped_output="$("${WRAPPER}" install status --profile "${PROFILE}")"
printf '%s\n' "${stopped_output}"
grep -Fq "Status:     stopped" <<<"${stopped_output}"

"${WRAPPER}" install start --profile "${PROFILE}"
started_output="$("${WRAPPER}" install status --profile "${PROFILE}")"
printf '%s\n' "${started_output}"
grep -Fq "Status:     running" <<<"${started_output}"
curl --fail --silent "http://127.0.0.1:${PORT}/readyz" >/dev/null

"${WRAPPER}" install restart --profile "${PROFILE}"
curl --fail --silent "http://127.0.0.1:${PORT}/readyz" >/dev/null

"${WRAPPER}" install remove --profile "${PROFILE}"
[[ ! -e "${HOME}/.headroom/deploy/${PROFILE}" ]]
