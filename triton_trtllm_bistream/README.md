# Triton TensorRT-LLM Bi-Streaming Backend

This module provides a custom Triton backend that embeds TensorRT-LLM runtime to serve CosyVoice2's LLM acoustic model with true bi-streaming semantics. Unlike the generic `tensorrt_llm` backend, the implementation here owns the session state, KV-cache, and scheduling logic so that text can be appended mid-session without resetting generation.

## Repository layout

- `model_repository/llm_bistream/` – Triton model repository entry for the acoustic LM backend.
- `runtime/` – Backend runtime implementation (TRT-LLM wrapper, sessions, scheduler, Triton glue).
- `scripts/` – Helper scripts for exporting engines and running local smoke tests.
- `tests/` – Lightweight client example used for CI smoke validation.

## Engine preparation

1. Export a TensorRT-LLM engine for the CosyVoice2 LLM acoustic model. The backend expects a standard TensorRT-LLM engine folder containing `config.json` and serialized engine plans. The folder path is passed via the `engine_dir` model parameter (see `config.pbtxt`).
2. Ensure the tokenizer directory is available. The backend uses Hugging Face `AutoTokenizer` for text tokenization and to obtain special token IDs.

## Running Triton

```bash
cd triton_trtllm_bistream
# populate the config template if needed
tritonserver --model-repository ./model_repository
```

The backend operates in decoupled mode; the `execute` entry point returns immediately and streaming responses are emitted through `InferenceResponseSender` objects.

## Client example

A minimal asynchronous client is available at `tests/client_example.py`. It demonstrates how to:

1. Start a session with `is_start=true` and the initial text.
2. Append additional text while generation is ongoing.
3. Receive partial speech-token chunks until EOS.

## Notes

- The scheduler performs micro-batching across sessions and keeps KV-caches resident on GPU memory.
- Session eviction, TTL, and backpressure are configurable via model parameters. See inline documentation in `model.py` and `runtime/session.py`.

