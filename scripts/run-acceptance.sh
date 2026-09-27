#!/usr/bin/env bash
# Start a JEF server, run the Robot Framework acceptance suites against its real
# HTTP surface, then shut it down. Used by CI and by `make acceptance`.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${JEF_PORT:-8080}"
BASE="http://127.0.0.1:${PORT}"
PY="${PY:-.venv/bin/python}"

"$PY" -m jef_server >/tmp/jef-acceptance.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
  echo "server failed to become healthy; log follows:" >&2
  cat /tmp/jef-acceptance.log >&2
  exit 1
fi

"$PY" -m robot \
  --outputdir test/results \
  --variable "JEF_BASE_URL:${BASE}" \
  "$@" \
  test/acceptance
