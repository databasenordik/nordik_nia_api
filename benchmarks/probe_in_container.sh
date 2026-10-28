#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/c/Users/gad/Desktop/nia-standalone}
cd "$ROOT"
exec docker compose -f infra/docker-compose.yml run --rm --no-deps \
  -v "$ROOT/backend/benchmarks:/app/benchmarks" \
  -v "$ROOT/docs:/app/docs" \
  backend python -m benchmarks.probe "$@"
