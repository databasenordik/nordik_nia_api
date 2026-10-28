#!/usr/bin/env bash
# Run the regression suite inside the backend image, which carries the reasoning SDK
# and the API key. backend/app is already bind-mounted by compose; benchmarks and docs
# are mounted here so the harness and the question file are visible too.
set -euo pipefail
ROOT=${ROOT:-/mnt/c/Users/gad/Desktop/nia-standalone}
cd "$ROOT"
exec docker compose -f infra/docker-compose.yml run --rm --no-deps \
  -v "$ROOT/backend/benchmarks:/app/benchmarks" \
  -v "$ROOT/docs:/app/docs" \
  backend python -m benchmarks.regression "$@"
