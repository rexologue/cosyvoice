#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO_PATH=${MODEL_REPO_PATH:-/models}
SERVICE_PORT=${SERVICE_PORT:-9000}
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
LOG_LEVEL=${LOG_LEVEL:-INFO}
export PYTHONPATH=/workspace:${PYTHONPATH}

# start triton server
tritonserver --model-repository "$MODEL_REPO_PATH" --http-port "$TRITON_HTTP_PORT" --grpc-port "$TRITON_GRPC_PORT" &
TRITON_PID=$!

# start orchestrator service
python3 -m triton_trtllm_bistream.runtime.service.server --port "$SERVICE_PORT" --triton-http-port "$TRITON_HTTP_PORT" --triton-grpc-port "$TRITON_GRPC_PORT" --log-level "$LOG_LEVEL" --startup-reference "${STARTUP_REFERENCE:-}" &
SERVICE_PID=$!

trap "kill $SERVICE_PID $TRITON_PID" SIGINT SIGTERM
wait $SERVICE_PID
