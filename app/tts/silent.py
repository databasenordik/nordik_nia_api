from __future__ import annotations

from collections.abc import AsyncIterator


class SilentTTSProvider:
    """Keyless fallback TTS that emits PCM silence.

    Used when TTS_PROVIDER=xai but no XAI_API_KEY is configured. A voice session
    still connects, transcribes with whisper.cpp, and streams the assistant's
    text answer to the UI — it just produces no spoken audio. Set XAI_API_KEY,
    or TTS_PROVIDER=kokoro for local speech, to hear answers.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self._sample_rate = sample_rate
        self.connected = False
        self._cancel = False

    async def connect_session(self) -> None:
        self.connected = True
        self._cancel = False

    async def synthesize_stream(
        self, text: str, voice_config: dict | None = None
    ) -> AsyncIterator[bytes]:
        if not self.connected:
            await self.connect_session()
        self._cancel = False
        if not text.strip() or self._cancel:
            return
        # ~40ms of 16-bit mono PCM silence per word keeps timing roughly natural.
        words = max(1, len(text.split()))
        samples = int(self._sample_rate * 0.04 * words)
        if self._cancel:
            return
        yield b"\x00\x00" * samples

    async def cancel_generation(self) -> None:
        self._cancel = True

    async def close_session(self) -> None:
        self.connected = False
