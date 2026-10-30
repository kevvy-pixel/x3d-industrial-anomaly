#!/usr/bin/env bash
set -euo pipefail
: "${WORKER_RTSP_URL:?Set WORKER_RTSP_URL}"
args=(--source "$WORKER_RTSP_URL")
[[ -n "${WORKER_DEVICE_ID:-}" ]] && args+=(--device-id "$WORKER_DEVICE_ID")
[[ -n "${WORKER_CALLBACK_URL:-}" ]] && args+=(--callback-url "$WORKER_CALLBACK_URL")
exec python -m worker_behavior.delivery "${args[@]}"
