# Copyright (c) 2025, CosyVoice contributors
# Licensed under the MIT license.
"""TensorRT-LLM runtime wrapper for bi-streaming acoustic LM."""

import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig
    from tensorrt_llm.runtime import LLM
except Exception as exc:  # pragma: no cover - environment guard
    GenerationSession = None
    ModelConfig = None
    SamplingConfig = None
    LLM = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class EngineConfig:
    engine_dir: str
    max_batch_tokens: int
    log_level: str = "INFO"


class TRTLLMEngine:
    """Owning TensorRT-LLM runtime objects and session creation."""

    def __init__(self, config: EngineConfig):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "TensorRT-LLM runtime is required for triton_trtllm_bistream"  # noqa: TRY003
            ) from _IMPORT_ERROR

        if not os.path.isdir(config.engine_dir):
            raise FileNotFoundError(f"engine_dir {config.engine_dir} does not exist")

        self._config = config
        self._model_config = ModelConfig.from_json(os.path.join(config.engine_dir, "config.json"))
        self._llm = LLM(config.engine_dir)
        self._lock = threading.Lock()

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    def create_session(self, session_id: str, prompt_token_ids: Sequence[int]) -> GenerationSession:
        """Create a GenerationSession with primed prompt tokens."""
        with self._lock:
            session = GenerationSession(self._llm, self._model_config)
        # set input ids on session; GenerationSession accepts numpy array
        input_ids = np.array(prompt_token_ids, dtype=np.int32)[None, :]
        session.setup(input_ids)
        return session

    def append_tokens(self, session: GenerationSession, token_ids: Sequence[int]):
        """Append tokens to the session's context (used for fill-token text injection)."""
        if len(token_ids) == 0:
            return
        step_input = np.array(token_ids, dtype=np.int32)[None, :]
        session.append_token_ids(step_input)

    def decode_step(
        self,
        sessions: List[GenerationSession],
        max_new_tokens: int,
        sampling: SamplingConfig,
    ) -> List[np.ndarray]:
        """Run one decode step for a micro-batch of sessions.

        Each session is expected to have already been primed with the proper
        input IDs. TensorRT-LLM internally handles KV-cache reuse.
        """
        if len(sessions) == 0:
            return []

        with self._lock:
            outputs: List[np.ndarray] = []
            # Micro-batching: execute sessions sequentially to guarantee KV reuse
            for sess in sessions:
                result = sess.generate(
                    sampling_config=sampling,
                    max_new_tokens=max_new_tokens,
                    streaming=True,
                )
                outputs.append(result.output_ids[0])
        return outputs

    def build_sampling_config(
        self, temperature: float, top_p: float, top_k: int, end_id: int, pad_id: int
    ) -> SamplingConfig:
        return SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            end_id=end_id,
            pad_id=pad_id,
            random_seed=0,
            do_sample=True,
        )

