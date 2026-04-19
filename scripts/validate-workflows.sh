#!/usr/bin/env bash
set -euo pipefail

actionlint .github/workflows/*.yml

act workflow_dispatch -W .github/workflows/release.yml -e .github/act/dry-run.json -n
act push -W .github/workflows/release.yml -e .github/act/push-feat.json -n
act workflow_dispatch -W .github/workflows/docker.yml -e .github/act/docker-version.json -n
