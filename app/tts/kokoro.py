from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings


class KokoroTTSProvider:
    """Self-hosted Kokoro-FastAPI client. Same TTSProvider surface as xAI."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        voice: str | None = None,
        audio_format: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.kokoro_url).rstrip("/")
        self._voice = voice or getattr(settings, "kokoro_voice", "af_heart")
        self._format = audio_format or getattr(settings, "kokoro_format", "pcm")
        self._client = client
        self._owns_client = client is None
        self._cancel = asyncio.Event()
        self.connected = False
        self.cancelled = False

    async def connect_session(self) -> None:
        client = await self._http()
        response = await client.get("/health")
        response.raise_for_status()
        self.connected = True
        self.cancelled = False

    @property
    def voice(self) -> str:
        return self._voice

    async def set_voice(self, voice: str) -> None:
        selected = voice.strip()
        if selected:
            self._voice = selected

    async def synthesize_stream(
        self, text: str, voice_config: dict | None = None
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        from app.observability.tracing import get_tracer

        with get_tracer().span("tts", provider="kokoro"):
            async for chunk in self._synthesize_inner(text, voice_config):
                yield chunk

    async def _synthesize_inner(
        self, text: str, voice_config: dict | None
    ) -> AsyncIterator[bytes]:
        self._cancel.clear()
        self.cancelled = False
        if not self.connected:
            await self.connect_session()
        voice = (voice_config or {}).get("voice") or self._voice
        client = await self._http()
        async with client.stream(
            "POST",
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": self._format,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if self._cancel.is_set():
                    await response.aclose()
                    return
                if chunk:
                    yield chunk

    async def cancel_generation(self) -> None:
        self.cancelled = True
        self._cancel.set()

    async def close_session(self) -> None:
        self.connected = False
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client
