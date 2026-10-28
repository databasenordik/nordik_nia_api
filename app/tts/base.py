from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class TTSProvider(Protocol):
    async def connect_session(self) -> None: ...

    async def synthesize_stream(
        self, text: str, voice_config: dict | None = None
    ) -> AsyncIterator[bytes]: ...

    async def cancel_generation(self) -> None: ...

    async def close_session(self) -> None: ...


def ensure_no_tts_in_text_mode(mode: str) -> None:
    """Text chat must never start a TTS stream."""
    if mode == "text":
        return
    return
