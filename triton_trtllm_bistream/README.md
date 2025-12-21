# Triton TensorRT-LLM Bi-Streaming Runtime

This folder contains an end-to-end, Dockerized streaming TTS runtime that mirrors `runtime/triton_trtllm` but swaps the acoustic LLM for the custom **bi-streaming** backend defined in `triton_trtllm_bistream/runtime`. The backend preserves KV cache between increments, enabling the CosyVoice2 "fill-token + append" protocol with low-latency, real-time PCM output.

## Quickstart

```bash
bash triton_trtllm_bistream/runtime/scripts/run.sh --model /path/to/CosyVoice2-0.5B
```

After the container starts, the orchestration service exposes:
- HTTP health: `curl http://localhost:9000/health`
- HTTP streaming TTS (chunked PCM):
  ```bash
  curl -X POST http://localhost:9000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "你好，欢迎体验双流式TTS"}' --output output.pcm
  ```
- WebSocket streaming (duplex text/audio):
  ```bash
  websocat ws://localhost:9000/stream
  # send: {"session_id":"demo","text":"Streaming TTS is live"}
  ```

The response stream is 24 kHz, mono, little-endian PCM (`int16`). Chunks are emitted as soon as the LLM produces new speech tokens; the final frame is followed by the sentinel text frame `END` on the WebSocket endpoint.

## What the runtime does

1. **Model intake** – Accepts either a Hugging Face-style checkpoint or a directory containing prebuilt TensorRT-LLM engines. If a checkpoint is provided, `run.sh` converts it to TensorRT weights and builds engines via `trtllm-build`.
2. **Repository templating** – Fills `model_repo_templates/` into a concrete Triton model repository (under `.workdir/model_repo`) and injects paths for engines, tokenizer, and speaker assets.
3. **Triton launch** – Starts Triton with decoupled transaction policy for `llm_bistream`, plus helper python backends for reference encoding and `token2wav` vocoding.
4. **Streaming orchestrator** – A FastAPI service (`runtime/service/server.py`) consumes streaming token responses from `llm_bistream`, incrementally calls the `token2wav` vocoder, and pushes PCM chunks to clients. It supports optional startup reference caching and per-request reference audio.

## run.sh flags (single-line examples)

- Hugging Face checkpoint with automatic conversion:
  ```bash
  bash triton_trtllm_bistream/runtime/scripts/run.sh --model /models/CosyVoice2-0.5B --device 0 --instances 2
  ```
- Prebuilt TensorRT engines:
  ```bash
  bash triton_trtllm_bistream/runtime/scripts/run.sh --model /engines/cosy2_trt --no-convert
  ```
- Custom ports and startup reference:
  ```bash
  bash triton_trtllm_bistream/runtime/scripts/run.sh --model /models/CosyVoice2-0.5B \
    --triton-http-port 8100 --triton-grpc-port 8101 --service-port 9100 \
    --startup-reference /audio/ref.wav --instances 4
  ```
- Reuse existing image:
  ```bash
  bash triton_trtllm_bistream/runtime/scripts/run.sh --model /engines/cosy2_trt --no-build
  ```

## API details

### HTTP `/tts`
- **Request body**: JSON with fields:
  - `text` (string, required): text to synthesize.
  - `session_id` (string, optional): stable identifier for multi-turn sessions.
  - `reference` (base64 wav, optional): custom speaker reference; defaults to startup reference or the built-in speaker profile.
- **Response**: `audio/raw` streaming body, 24 kHz, mono, signed int16 PCM. Headers `X-Sample-Rate` and `X-Format` describe the payload.

### WebSocket `/stream`
Send JSON frames shaped like `{ "session_id": "uuid", "text": "..." }`. Binary frames carry PCM audio chunks; a final text frame `END` marks completion. The same connection can be reused for multiple turns.

## Repository layout

- `runtime/scripts/run.sh` – one-command launcher: conversion (if needed), template fill, Docker build/run.
- `runtime/scripts/prepare_model_repo.py` – fills `model_repo_templates` with concrete paths (engines, tokenizer, model dir) and concurrency settings.
- `runtime/service/server.py` – FastAPI orchestrator that bridges `llm_bistream`, `audio_tokenizer`, `speaker_embedding`, and `token2wav` to produce streaming PCM.
- `runtime/docker/Dockerfile` – builds a Triton + service image; `entrypoint.sh` starts Triton and the orchestrator.
- `runtime/model_repo_templates/` – pbtxt templates for all models (bi-stream LLM, vocoder, and reference encoders).

## Audio chunking semantics

- **Sample rate**: 24000 Hz
- **Format**: mono, `int16`, little-endian
- **Chunk cadence**: each chunk corresponds to the latest speech-token batch from `llm_bistream`; chunk size varies with backend scheduling and `max_batch_tokens`.
- **Finalization**: a trailing chunk is emitted after `is_final=true` to flush the vocoder cache.

## Reference handling

- **Startup reference** (`--startup-reference`): encoded once at service start and cached.
- **Per-request reference**: send `reference` as base64 WAV in `/tts`; the service encodes tokens, mel features, and embeddings on the fly via Triton helper models. Encoded prompts are cached per `session_id` for reuse within the connection.

## Scaling and performance knobs

Key environment variables and script parameters:
- `--instances`: number of `llm_bistream` model instances in Triton.
- `--device`: comma-separated GPU IDs exposed to the container.
- `--log-level`: controls orchestrator verbosity.
- Template parameters `max_sessions`, `max_batch_tokens`, `mix_ratio` are exposed in `prepare_model_repo.py` for scheduler tuning.

## Troubleshooting

- **Triton model load errors**: ensure `ENGINE_DIR` contains valid TensorRT plans and the tokenizer directory is mounted inside the container (`/tokenizer`).
- **Audio drift or silence**: verify reference sampling rate is 16 kHz; mismatched rates lead to invalid mel features.
- **Streaming stalls**: check GPU memory availability and reduce `--instances` or `max_batch_tokens` via template substitution.
- **Port conflicts**: override `--triton-http-port`, `--triton-grpc-port`, or `--service-port` in `run.sh`.

## Development notes

- Generated model repositories live under `triton_trtllm_bistream/runtime/.workdir/model_repo`.
- The Docker image is rebuilt automatically unless `--no-build` is provided.
- To tweak pbtxt templates or add new parameters, edit files under `runtime/model_repo_templates/` and rerun `prepare_model_repo.py` via `run.sh`.
