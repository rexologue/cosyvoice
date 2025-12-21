# Copyright (c) 2025, CosyVoice contributors
"""Scheduling and micro-batching for bi-streaming sessions."""

import logging
import queue
import threading
import time
from typing import List, Optional

import numpy as np
import triton_python_backend_utils as pb_utils

from .engine import TRTLLMEngine
from .session import SessionManager, SessionState


class Scheduler(threading.Thread):
    def __init__(
        self,
        engine: TRTLLMEngine,
        session_manager: SessionManager,
        fill_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        mix_ratio: List[int],
        max_batch_tokens: int,
    ):
        super().__init__(daemon=True)
        self._engine = engine
        self._session_manager = session_manager
        self._fill_token_id = fill_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._mix_ratio = mix_ratio
        self._max_batch_tokens = max_batch_tokens
        self._stop = threading.Event()
        self._pending = queue.Queue()
        self._logger = logging.getLogger("llm_bistream.scheduler")

    def submit(self, session_id: str):
        self._pending.put(session_id)

    def stop(self):
        self._stop.set()

    def _build_ready_batches(self) -> List[List[SessionState]]:
        ready_sessions: List[SessionState] = []
        seen = set()
        while not self._pending.empty():
            try:
                sid = self._pending.get_nowait()
            except queue.Empty:
                break
            if sid in seen:
                continue
            seen.add(sid)
            session = self._session_manager.get(sid)
            if session is None or session.closed:
                continue
            ready_sessions.append(session)
        batches: List[List[SessionState]] = []
        current: List[SessionState] = []
        current_tokens = 0
        for sess in ready_sessions:
            est = self._mix_ratio[1]
            if current and current_tokens + est > self._max_batch_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(sess)
            current_tokens += est
        if current:
            batches.append(current)
        return batches

    def run(self):
        self._logger.info("scheduler started")
        while not self._stop.is_set():
            batches = self._build_ready_batches()
            if not batches:
                time.sleep(0.001)
                continue
            for batch in batches:
                self._process_batch(batch)
        self._logger.info("scheduler stopped")

    def _process_batch(self, batch: List[SessionState]):
        sampling = self._engine.build_sampling_config(
            temperature=1.0, top_p=0.9, top_k=50, end_id=self._eos_token_id, pad_id=self._pad_token_id
        )
        outputs = self._engine.decode_step([s.trt_session for s in batch], self._mix_ratio[1], sampling)
        for session, out in zip(batch, outputs):
            tokens = out.tolist()
            session.step_index += 1
            new_tokens = tokens[len(session.emitted_tokens) :]
            session.emitted_tokens.extend(new_tokens)
            self._emit_tokens(session, new_tokens)
            if new_tokens and new_tokens[-1] == self._fill_token_id:
                if session.ready_for_fill(len(session.emitted_tokens)):
                    append = session.pop_fill_tokens()
                    self._engine.append_tokens(session.trt_session, append)
                    self.submit(session.session_id)
                else:
                    self._logger.debug("session %s waiting for text", session.session_id)
            elif new_tokens and new_tokens[-1] == self._eos_token_id:
                session.mark_closed()
                self._finalize(session, finish_reason=0)
            else:
                self.submit(session.session_id)

    def _emit_tokens(self, session: SessionState, tokens: List[int]):
        if not tokens:
            return
        sender: Optional[pb_utils.InferenceResponseSender] = None
        if session.pending_requests:
            sender = session.pending_requests[-1]
        if sender is None:
            return
        tensor = pb_utils.Tensor("speech_token_ids", np.array(tokens, dtype=np.int32))
        final_tensor = pb_utils.Tensor("is_final", np.array([False], dtype=np.bool_))
        finish_tensor = pb_utils.Tensor("finish_reason", np.array([-1], dtype=np.int32))
        sender.send(pb_utils.InferenceResponse(output_tensors=[tensor, final_tensor, finish_tensor]))

    def _finalize(self, session: SessionState, finish_reason: int):
        if session.pending_requests:
            sender = session.pending_requests.popleft()
            final_tensor = pb_utils.Tensor("is_final", np.array([True], dtype=np.bool_))
            finish_tensor = pb_utils.Tensor("finish_reason", np.array([finish_reason], dtype=np.int32))
            empty = pb_utils.Tensor("speech_token_ids", np.zeros((0,), dtype=np.int32))
            sender.send(pb_utils.InferenceResponse(output_tensors=[empty, final_tensor, finish_tensor]))
            sender.finish()  # type: ignore[attr-defined]
        session.mark_closed()
        self._session_manager.drop(session.session_id)

