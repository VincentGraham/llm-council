#!/bin/bash

set -euo pipefail

BACKEND_PID=""

cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup SIGINT SIGTERM EXIT

echo "Starting LLM Council (local NIM + backend)..."

echo "Bringing up NIM containers with docker compose..."
docker compose up -d

echo "Starting backend on http://localhost:8001..."
uv run python -m backend.main &
BACKEND_PID=$!

echo "Waiting for backend and model endpoints to become ready..."
READY=0
for attempt in $(seq 1 120); do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend process exited unexpectedly."
    wait "${BACKEND_PID}" || true
    exit 1
  fi

  response="$(curl -fsS http://localhost:8001/api/health 2>/dev/null || true)"
  if [[ -n "${response}" ]] && echo "${response}" | grep -q '"ready"[[:space:]]*:[[:space:]]*true'; then
    READY=1
    break
  fi

  echo "  [${attempt}/120] waiting..."
  sleep 5
done

if [[ "${READY}" -eq 1 ]]; then
  echo "All model endpoints are healthy."
else
  echo "Timeout waiting for all model endpoints. Backend remains running."
fi

echo "LLM Council backend running at http://localhost:8001"
echo "Press Ctrl+C to stop backend (containers keep running)."

wait "${BACKEND_PID}"
