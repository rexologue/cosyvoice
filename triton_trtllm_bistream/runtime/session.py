# Copyright (c) 2025, CosyVoice contributors
"""Session state tracking for bi-streaming."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class SessionState:
    session_id: str
    trt_session: object
    fill_token_id: int
    eos_token_id: int
    pad_token_id: int
    mix_ratio: List[int]
    token_buffer: Deque[int] = field(default_factory=deque)
    emitted_tokens: List[int] = field(default_factory=list)
    last_access: float = field(default_factory=time.time)
    step_index: int = 0
    closed: bool = False
    pending_requests: Deque[object] = field(default_factory=deque)

    def buffer_text_tokens(self, token_ids: List[int]):
        self.token_buffer.extend(token_ids)
        self.last_access = time.time()

    def mark_closed(self):
        self.closed = True
        self.last_access = time.time()

    def ready_for_fill(self, out_tokens_len: int) -> bool:
        """Check whether enough buffered text tokens exist for the next fill slot."""
        need = self.mix_ratio[0]
        return len(self.token_buffer) >= need

    def pop_fill_tokens(self) -> List[int]:
        need = self.mix_ratio[0]
        tokens = [self.token_buffer.popleft() for _ in range(min(need, len(self.token_buffer)))]
        self.last_access = time.time()
        return tokens


class SessionManager:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        session = self._sessions.get(session_id)
        if session:
            session.last_access = time.time()
        return session

    def create(self, session_id: str, **kwargs) -> SessionState:
        if session_id in self._sessions:
            raise ValueError(f"session {session_id} already exists")
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError("maximum sessions reached")
        session = SessionState(session_id=session_id, **kwargs)
        self._sessions[session_id] = session
        return session

    def drop(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def evict_expired(self):
        now = time.time()
        expired = [sid for sid, state in self._sessions.items() if now - state.last_access > self._ttl]
        for sid in expired:
            self.drop(sid)

    def active_sessions(self) -> List[SessionState]:
        return list(self._sessions.values())

