from __future__ import annotations

from collections.abc import AsyncIterator


class FakeTTSProvider:
    """In-process TTS. Tests never open wss://api.x.ai."""

    def __init__(self) -> None:
        self.connected = False
        self.cancelled = False
        self.closed = False
        self.spoken: list[str] = []
        self.started_streams = 0

    async def connect_session(self) -> None:
        self.connected = True
        self.closed = False

    async def synthesize_stream(
        self, text: str, voice_config: dict | None = None
    ) -> AsyncIterator[bytes]:
        if not self.connected:
            await self.connect_session()
        self.started_streams += 1
        self.cancelled = False
        self.spoken.append(text)
        if self.cancelled:
            return
        yield f"PCM:{text}".encode()

    async def cancel_generation(self) -> None:
        self.cancelled = True

    async def close_session(self) -> None:
        self.connected = False
        self.closed = True
