import argparse
import asyncio
import base64
import io
import json
import logging
import os
import queue
import sys
from typing import AsyncGenerator, Dict, Optional

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../../..")
sys.path.append(f"{ROOT_DIR}/../../../third_party/Matcha-TTS")

from matcha.utils.audio import mel_spectrogram

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

_LOGGER = logging.getLogger("bistream_service")


class TritonLLMClient:
    def __init__(self, url: str, model_name: str = "llm_bistream"):
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self.model_name = model_name

    def stream_tokens(self, *, session_id: str, text: str, is_start: bool, is_end: bool, max_tokens: int,
                      temperature: float, top_p: float, top_k: int) -> AsyncGenerator[np.ndarray, None]:
        response_queue: "queue.Queue[grpcclient.InferResult]" = queue.Queue()
        error_queue: "queue.Queue[Exception]" = queue.Queue()

        def _callback(result, error):
            if error:
                error_queue.put(error)
            else:
                response_queue.put(result)

        self.client.start_stream(_callback, stream_timeout=None)
        inputs = [
            grpcclient.InferInput("session_id", [1], "BYTES"),
            grpcclient.InferInput("text", [len(text)], "BYTES"),
            grpcclient.InferInput("is_start", [1], "BOOL"),
            grpcclient.InferInput("is_end", [1], "BOOL"),
            grpcclient.InferInput("streaming", [1], "BOOL"),
            grpcclient.InferInput("max_tokens", [1], "INT32"),
            grpcclient.InferInput("temperature", [1], "FP32"),
            grpcclient.InferInput("top_p", [1], "FP32"),
            grpcclient.InferInput("top_k", [1], "INT32"),
        ]
        inputs[0].set_data_from_numpy(np.array([session_id.encode("utf-8")], dtype=object))
        inputs[1].set_data_from_numpy(np.array(list(text.encode("utf-8")), dtype=np.bytes_))
        inputs[2].set_data_from_numpy(np.array([is_start], dtype=np.bool_))
        inputs[3].set_data_from_numpy(np.array([is_end], dtype=np.bool_))
        inputs[4].set_data_from_numpy(np.array([True], dtype=np.bool_))
        inputs[5].set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
        inputs[6].set_data_from_numpy(np.array([temperature], dtype=np.float32))
        inputs[7].set_data_from_numpy(np.array([top_p], dtype=np.float32))
        inputs[8].set_data_from_numpy(np.array([top_k], dtype=np.int32))

        self.client.async_stream_infer(self.model_name, inputs=inputs, request_id=session_id)

        while True:
            if not error_queue.empty():
                self.client.stop_stream()
                raise error_queue.get()
            result = response_queue.get()
            speech_tokens = result.as_numpy("speech_token_ids")
            is_final = bool(result.as_numpy("is_final")[0])
            yield speech_tokens
            if is_final:
                break
        self.client.stop_stream()


class TritonHelper:
    def __init__(self, url: str):
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def encode_reference(self, wav: np.ndarray) -> Dict[str, np.ndarray]:
        wav = wav.astype(np.float32)
        wav_len = np.array([wav.shape[0]], dtype=np.int32)
        inputs = [
            grpcclient.InferInput("reference_wav", wav.shape, "FP32"),
            grpcclient.InferInput("reference_wav_len", [1], "INT32"),
        ]
        inputs[0].set_data_from_numpy(wav)
        inputs[1].set_data_from_numpy(wav_len)
        audio_tokenizer = self.client.infer(
            model_name="audio_tokenizer",
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("prompt_speech_tokens")],
        )
        prompt_tokens = audio_tokenizer.as_numpy("prompt_speech_tokens")

        spk_inputs = [grpcclient.InferInput("reference_wav", wav.shape, "FP32")]
        spk_inputs[0].set_data_from_numpy(wav)
        spk_embedding_resp = self.client.infer(
            model_name="speaker_embedding",
            inputs=spk_inputs,
            outputs=[grpcclient.InferRequestedOutput("prompt_spk_embedding")],
        )
        spk_embedding = spk_embedding_resp.as_numpy("prompt_spk_embedding")

        mel_tensor = torch.from_numpy(np.expand_dims(wav, 0))
        mel = mel_spectrogram(mel_tensor, 16000)
        mel = mel.squeeze(0).T.cpu().numpy().astype(np.float16)

        return {
            "prompt_speech_tokens": prompt_tokens,
            "prompt_spk_embedding": spk_embedding,
            "prompt_speech_feat": mel,
        }

    def token2wav(self, tokens: np.ndarray, prompt: Optional[Dict[str, np.ndarray]], token_offset: int,
                  finalize: bool) -> np.ndarray:
        inputs = [
            grpcclient.InferInput("target_speech_tokens", tokens.shape, "INT32"),
        ]
        inputs[0].set_data_from_numpy(tokens)
        if prompt is not None:
            ps = prompt["prompt_speech_tokens"]
            pf = prompt["prompt_speech_feat"]
            pe = prompt["prompt_spk_embedding"]
            ps_input = grpcclient.InferInput("prompt_speech_tokens", ps.shape, "INT32")
            ps_input.set_data_from_numpy(ps)
            pf_input = grpcclient.InferInput("prompt_speech_feat", pf.shape, "FP16")
            pf_input.set_data_from_numpy(pf)
            pe_input = grpcclient.InferInput("prompt_spk_embedding", pe.shape, "FP16")
            pe_input.set_data_from_numpy(pe)
            inputs.extend([ps_input, pf_input, pe_input])

        offset_tensor = grpcclient.InferInput("token_offset", [1], "INT32")
        offset_tensor.set_data_from_numpy(np.array([token_offset], dtype=np.int32))
        finalize_tensor = grpcclient.InferInput("finalize", [1], "BOOL")
        finalize_tensor.set_data_from_numpy(np.array([finalize], dtype=np.bool_))
        inputs.extend([offset_tensor, finalize_tensor])

        response = self.client.infer(
            model_name="token2wav",
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("waveform")],
        )
        return response.as_numpy("waveform")


class PCMStreamer:
    def __init__(self, llm: TritonLLMClient, helper: TritonHelper, default_prompt: Optional[Dict[str, np.ndarray]] = None):
        self.llm = llm
        self.helper = helper
        self.default_prompt = default_prompt

    async def stream(self, *, session_id: str, text: str, prompt: Optional[Dict[str, np.ndarray]]) -> AsyncGenerator[bytes, None]:
        prompt = prompt or self.default_prompt
        offset = 0
        loop = asyncio.get_event_loop()
        for token_chunk in self.llm.stream_tokens(
            session_id=session_id,
            text=text,
            is_start=True,
            is_end=True,
            max_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
        ):
            flat_chunk = token_chunk.reshape(-1)
            wav = await loop.run_in_executor(
                None,
                self.helper.token2wav,
                flat_chunk.astype(np.int32),
                prompt,
                offset,
                False,
            )
            offset += flat_chunk.shape[0]
            pcm = np.clip(wav.flatten(), -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16).tobytes()
            yield pcm
        # finalize
        final_wav = await loop.run_in_executor(
            None,
            self.helper.token2wav,
            np.zeros((1,), dtype=np.int32),
            prompt,
            offset,
            True,
        )
        pcm = np.clip(final_wav.flatten(), -1.0, 1.0)
        yield (pcm * 32767.0).astype(np.int16).tobytes()


app = FastAPI()


def create_app(triton_grpc_port: int, startup_reference: Optional[str]):
    triton_url = f"localhost:{triton_grpc_port}"
    llm_client = TritonLLMClient(triton_url)
    helper = TritonHelper(triton_url)
    default_prompt = None
    prompt_cache: Dict[str, Dict[str, np.ndarray]] = {}
    if startup_reference:
        wav, sr = sf.read(startup_reference)
        if sr != 16000:
            raise ValueError("Startup reference must be 16kHz")
        default_prompt = helper.encode_reference(wav.astype(np.float32))
        _LOGGER.info("Loaded startup reference %s", startup_reference)
    streamer = PCMStreamer(llm_client, helper, default_prompt)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/tts")
    async def tts(payload: Dict[str, str]):
        text = payload.get("text")
        if not text:
            return JSONResponse(status_code=400, content={"error": "text is required"})
        session_id = payload.get("session_id", "default")
        reference_b64 = payload.get("reference")
        prompt = prompt_cache.get(session_id, default_prompt)
        if reference_b64:
            audio_bytes = base64.b64decode(reference_b64)
            wav, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != 16000:
                return JSONResponse(status_code=400, content={"error": "reference must be 16kHz"})
            prompt = helper.encode_reference(wav.astype(np.float32))
            prompt_cache[session_id] = prompt

        async def generate():
            async for chunk in streamer.stream(session_id=session_id, text=text, prompt=prompt):
                yield chunk

        headers = {"X-Sample-Rate": "24000", "X-Format": "s16le"}
        return StreamingResponse(generate(), media_type="audio/raw", headers=headers)

    @app.websocket("/stream")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                text = data.get("text", "")
                session_id = data.get("session_id", "default")
                prompt = prompt_cache.get(session_id, default_prompt)
                async for chunk in streamer.stream(session_id=session_id, text=text, prompt=prompt):
                    await websocket.send_bytes(chunk)
                await websocket.send_text("END")
        except WebSocketDisconnect:
            return

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--triton-grpc-port", type=int, default=8001)
    parser.add_argument("--triton-http-port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--startup-reference", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    import uvicorn

    app = create_app(args.triton_grpc_port, args.startup_reference)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
