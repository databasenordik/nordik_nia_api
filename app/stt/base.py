from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel


class TranscriptEvent(BaseModel):
    type: Literal["partial", "final"]
    text: str
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0
    speech: bool = True


class STTProvider(Protocol):
    """Replaceable local STT. Retrieval and reasoning must not import a vendor."""

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]: ...

    async def reset_session(self) -> None: ...

    async def finalize_turn(self) -> TranscriptEvent | None: ...
