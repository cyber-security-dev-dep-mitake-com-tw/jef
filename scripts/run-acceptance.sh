#!/usr/bin/env bash
# Start a JEF server, run the Robot Framework acceptance suites against its real
# HTTP surface, then shut it down.
#
#   ./scripts/run-acceptance.sh                 # Python server, every suite
#   JEF_RUNTIME=go ./scripts/run-acceptance.sh  # Go server, contract suites only
#
# The Go runtime does not implement scenes yet, so its run excludes them. What
# it does cover, it must cover identically: the hashing backbone is bit-for-bit
# the same in both languages, so the same assertions hold against either server.
set -euo pipefail

cd "$(dirname "$0")/.."

RUNTIME="${JEF_RUNTIME:-python}"
PORT="${JEF_PORT:-8080}"
BASE="http://127.0.0.1:${PORT}"
PY="${PY:-.venv/bin/python}"
OUTDIR="${JEF_OUTDIR:-test/results}"

export JEF_SCENES_DIR="${JEF_SCENES_DIR:-scenes}"

case "$RUNTIME" in
  python)
    "$PY" -m jef_server >/tmp/jef-acceptance.log 2>&1 &
    # Left unset rather than empty: under `set -u`, bash 3.2 (still the system
    # bash on macOS) treats "${EMPTY[@]}" as an unbound variable and aborts.
    ;;
  go)
    # Built from inside packages/jef-go: the Go module lives there, and the
    # repository root has none, so building by relative path from here fails
    # with "cannot find main module".
    (cd packages/jef-go && go build -o /tmp/jef-server-go ./cmd/jef-server)
    JEF_ADDR=":${PORT}" /tmp/jef-server-go >/tmp/jef-acceptance.log 2>&1 &
    # Scenes live in jef_scene, which has no Go port yet.
    EXCLUDE=(--exclude scene)
    OUTDIR="${JEF_OUTDIR:-test/results-go}"
    ;;
  *)
    echo "unknown JEF_RUNTIME '$RUNTIME' (expected 'python' or 'go')" >&2
    exit 2
    ;;
esac

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
  echo "${RUNTIME} server failed to become healthy; log follows:" >&2
  cat /tmp/jef-acceptance.log >&2
  exit 1
fi

"$PY" -m robot \
  --outputdir "$OUTDIR" \
  --variable "JEF_BASE_URL:${BASE}" \
  ${EXCLUDE[@]+"${EXCLUDE[@]}"} \
  "$@" \
  test/acceptance
