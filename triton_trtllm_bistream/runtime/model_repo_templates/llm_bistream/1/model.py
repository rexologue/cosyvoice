# Copyright (c) 2025, CosyVoice contributors
"""Custom Triton Python backend for CosyVoice2 bi-streaming acoustic LM."""

import json
import logging
import os
import sys
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from runtime.engine import EngineConfig, TRTLLMEngine
from runtime.scheduler import Scheduler
from runtime.session import SessionManager


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in self.model_config.get("parameters", {}).items()}
        logging.basicConfig(level=getattr(logging, params.get("log_level", "INFO")))
        self.logger = logging.getLogger("llm_bistream")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)
        if not self.decoupled:
            raise pb_utils.TritonModelException("llm_bistream must run in decoupled mode")

        engine_dir = params["engine_dir"]
        tokenizer_dir = params["llm_tokenizer_dir"]
        mix_ratio = [int(x) for x in params.get("mix_ratio", "5,15").split(",")]
        session_timeout = int(params.get("session_timeout_s", "180"))
        max_sessions = int(params.get("max_sessions", "64"))
        max_batch_tokens = int(params.get("max_batch_tokens", "512"))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.sos_token_id = self.tokenizer.convert_tokens_to_ids("<|sos|>")
        self.task_id = self.tokenizer.convert_tokens_to_ids("<|task_id|>")
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>")
        self.fill_token_id = self.tokenizer.vocab.get("<|fill|>") if hasattr(self.tokenizer, "vocab") else None
        if self.fill_token_id is None:
            speech_vocab = self.tokenizer.vocab_size
            self.fill_token_id = speech_vocab + 2
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

        engine_config = EngineConfig(engine_dir=engine_dir, max_batch_tokens=max_batch_tokens)
        self.engine = TRTLLMEngine(engine_config)
        self.sessions = SessionManager(ttl_seconds=session_timeout, max_sessions=max_sessions)
        self.scheduler = Scheduler(
            engine=self.engine,
            session_manager=self.sessions,
            fill_token_id=self.fill_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            mix_ratio=mix_ratio,
            max_batch_tokens=max_batch_tokens,
        )
        self.scheduler.start()
        self.mix_ratio = mix_ratio
        self.logger.info("llm_bistream initialized with engine_dir=%s", engine_dir)

    def _tokenize_text(self, text: bytes) -> List[int]:
        decoded = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
        return self.tokenizer.encode(decoded, add_special_tokens=False)

    def _prime_prompt(self, text_tokens: List[int]) -> List[int]:
        return [self.sos_token_id] + text_tokens + [self.task_id]

    def execute(self, requests):
        for request in requests:
            sender = pb_utils.InferenceResponseSender(request)
            try:
                self._handle_request(request, sender)
            except Exception as exc:  # pragma: no cover - triton error path
                self.logger.exception("failed to handle request")
                err = pb_utils.TritonError(str(exc))
                sender.send(pb_utils.InferenceResponse(error=err))
                sender.finish()
        return None

    def finalize(self):
        if hasattr(self, "scheduler"):
            self.scheduler.stop()
            self.scheduler.join(timeout=1.0)

    def _as_bytes(self, tensor: pb_utils.Tensor) -> bytes:
        if tensor is None:
            return b""
        arr = tensor.as_numpy().reshape(-1)
        if arr.dtype.type is np.bytes_:
            return b"".join(arr.tolist())
        return bytes(arr)

    def _handle_request(self, request, sender: pb_utils.InferenceResponseSender):
        session_raw = pb_utils.get_input_tensor_by_name(request, "session_id").as_numpy()[0]
        session_id = session_raw.tobytes().decode("utf-8")
        text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
        text = self._as_bytes(text_tensor)
        is_start = bool(pb_utils.get_input_tensor_by_name(request, "is_start").as_numpy()[0])
        is_end = bool(pb_utils.get_input_tensor_by_name(request, "is_end").as_numpy()[0])
        streaming = bool(pb_utils.get_input_tensor_by_name(request, "streaming").as_numpy()[0])

        if not streaming:
            self.logger.warning("non-streaming mode requested; continuing with streaming semantics")

        text_tokens = self._tokenize_text(text) if len(text) > 0 else []

        session = self.sessions.get(session_id)
        if is_start:
            if session is not None:
                raise pb_utils.TritonModelException(f"session {session_id} already exists")
            prompt = self._prime_prompt(text_tokens)
            trt_session = self.engine.create_session(session_id, prompt)
            session = self.sessions.create(
                session_id=session_id,
                trt_session=trt_session,
                fill_token_id=self.fill_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                mix_ratio=self.mix_ratio,
            )
            session.pending_requests.append(sender)
            self.scheduler.submit(session_id)
            return

        if session is None:
            raise pb_utils.TritonModelException(f"unknown session {session_id}")

        session.pending_requests.append(sender)
        if text_tokens:
            session.buffer_text_tokens(text_tokens)
        if is_end:
            session.mark_closed()
        self.scheduler.submit(session_id)

