from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HarnessProvider(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    GEMINI = "gemini"


class HarnessJobStatus(str, Enum):
    ACCEPTED = "accepted"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class HarnessJobRequest(BaseModel):
    provider: HarnessProvider
    prompt: str = Field(min_length=1)
    cwd: Optional[str] = None
    additional_args: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=7200)
    env: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("additional_args")
    @classmethod
    def _validate_additional_args(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            text = (item or "").strip()
            if text:
                cleaned.append(text)
        if len(cleaned) > 24:
            raise ValueError("additional_args exceeds maximum of 24 arguments")
        return cleaned


class HarnessOutputChunk(BaseModel):
    ts: str
    stream: str
    text: str


class HarnessJobState(BaseModel):
    id: str
    request_id: str
    provider: HarnessProvider
    prompt: str
    cwd: str
    additional_args: List[str] = Field(default_factory=list)
    timeout_seconds: int = 900
    env: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: HarnessJobStatus = HarnessJobStatus.ACCEPTED
    pid: Optional[int] = None
    return_code: Optional[int] = None
    command: List[str] = Field(default_factory=list)
    timed_out: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    output_tail: List[HarnessOutputChunk] = Field(default_factory=list)
    output_truncated: bool = False

    def append_output(self, stream: str, text: str, max_chunks: int) -> None:
        self.output_tail.append(
            HarnessOutputChunk(ts=datetime.now(UTC).isoformat(), stream=stream, text=text)
        )
        if len(self.output_tail) > max_chunks:
            overflow = len(self.output_tail) - max_chunks
            self.output_tail = self.output_tail[overflow:]
            self.output_truncated = True

