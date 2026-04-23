#!/usr/bin/env bash
set -euo pipefail

actionlint .github/workflows/*.yml

run_act() {
  local attempt=1
  local max_attempts=3
  local delay_seconds=5

  while true; do
    if "$@"; then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      return 1
    fi

    echo "act dry-run failed on attempt ${attempt}/${max_attempts}; retrying in ${delay_seconds}s..." >&2
    sleep "${delay_seconds}"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

run_act act workflow_dispatch -W .github/workflows/release.yml -e .github/act/dry-run.json -n
run_act act push -W .github/workflows/release.yml -e .github/act/push-feat.json -n
run_act act workflow_dispatch -W .github/workflows/docker.yml -e .github/act/docker-version.json -n
