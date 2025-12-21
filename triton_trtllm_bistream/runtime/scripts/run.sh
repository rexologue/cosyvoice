#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RUNTIME_ROOT/../.." && pwd)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

MODEL_PATH=""
TRITON_HTTP_PORT=8000
TRITON_GRPC_PORT=8001
SERVICE_PORT=9000
INSTANCES=1
DEVICES="0"
STARTUP_REFERENCE=""
NO_BUILD=0
NO_CONVERT=0
LOG_LEVEL=INFO
WORKDIR="$RUNTIME_ROOT/.workdir"
TEMPLATE_ROOT="$RUNTIME_ROOT/model_repo_templates"
MODEL_REPO="$WORKDIR/model_repo"
ENGINE_DIR=""
TOKENIZER_DIR=""
MODEL_DIR=""
DECOUPLED=1

usage() {
  cat <<EOF
Usage: bash run.sh --model <path> [options]
  --model PATH                 HF model directory or TensorRT-LLM engine directory
  --triton-http-port PORT      Triton HTTP port (default: 8000)
  --triton-grpc-port PORT      Triton gRPC port (default: 8001)
  --service-port PORT          Service HTTP/WebSocket port (default: 9000)
  --instances N                Number of Triton llm_bistream instances (default: 1)
  --device IDS                 CUDA visible devices (default: 0)
  --startup-reference PATH     Optional reference wav to encode at startup
  --no-build                   Skip Docker build
  --no-convert                 Do not attempt checkpoint conversion
  --log-level LEVEL            Service and backend log level
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2;;
    --triton-http-port) TRITON_HTTP_PORT="$2"; shift 2;;
    --triton-grpc-port) TRITON_GRPC_PORT="$2"; shift 2;;
    --service-port) SERVICE_PORT="$2"; shift 2;;
    --instances) INSTANCES="$2"; shift 2;;
    --device) DEVICES="$2"; shift 2;;
    --startup-reference) STARTUP_REFERENCE="$2"; shift 2;;
    --no-build) NO_BUILD=1; shift;;
    --no-convert) NO_CONVERT=1; shift;;
    --log-level) LOG_LEVEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model is required" >&2
  exit 1
fi

mkdir -p "$WORKDIR"

# Determine model layout
if [[ -f "$MODEL_PATH/config.json" ]]; then
  echo "Detected HuggingFace-style model directory"
  TOKENIZER_DIR="$MODEL_PATH"
  MODEL_DIR="$MODEL_PATH"
  ENGINE_DIR="$WORKDIR/engines"
  if [[ $NO_CONVERT -eq 1 ]]; then
    echo "Conversion required but --no-convert set" >&2
    exit 1
  fi
  python3 "$SCRIPT_DIR/convert_checkpoint.py" --model_dir "$MODEL_PATH" --output_dir "$WORKDIR/trt_weights" --dtype bfloat16
  trtllm-build --checkpoint_dir "$WORKDIR/trt_weights" --output_dir "$ENGINE_DIR" --max_batch_size 16 --max_num_tokens 32768 --gemm_plugin bfloat16
else
  echo "Assuming TensorRT-LLM engine directory"
  ENGINE_DIR="$MODEL_PATH"
  TOKENIZER_DIR="$MODEL_PATH"
  MODEL_DIR="$MODEL_PATH"
fi

# Prepare model repository
rm -rf "$MODEL_REPO"
python3 -m triton_trtllm_bistream.runtime.scripts.prepare_model_repo \
  --template-root "$TEMPLATE_ROOT" \
  --output-root "$MODEL_REPO" \
  --engine-dir "$ENGINE_DIR" \
  --llm-tokenizer-dir "$TOKENIZER_DIR" \
  --model-dir "$MODEL_DIR" \
  --instances "$INSTANCES" \
  --decoupled \
  --log-level "$LOG_LEVEL"

# Build docker image
IMAGE_NAME=cosyvoice-bistream:latest
if [[ $NO_BUILD -eq 0 ]]; then
  docker build -t $IMAGE_NAME -f "$RUNTIME_ROOT/docker/Dockerfile" "$REPO_ROOT"
fi

# Launch container
CONTAINER_NAME=cosyvoice-bistream-runtime
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  docker rm -f "$CONTAINER_NAME"
fi

docker run -it --gpus "device=${DEVICES}" --net host --shm-size=2g --name "$CONTAINER_NAME" \
  -e STARTUP_REFERENCE="$STARTUP_REFERENCE" \
  -e MODEL_REPO_PATH="$MODEL_REPO" \
  -e SERVICE_PORT="$SERVICE_PORT" \
  -e TRITON_HTTP_PORT="$TRITON_HTTP_PORT" \
  -e TRITON_GRPC_PORT="$TRITON_GRPC_PORT" \
  -v "$MODEL_REPO":"/models" \
  -v "$ENGINE_DIR":"/engines" \
  -v "$MODEL_DIR":"/model_dir" \
  -v "$TOKENIZER_DIR":"/tokenizer" \
  $IMAGE_NAME
