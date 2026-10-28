from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.stt.base import TranscriptEvent


class WhisperCppSTTProvider:
    """HTTP/WS client for the independent whisper service. No subprocess here."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.whisper_service_url).rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._session_id: str | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client

    async def _ensure_session(self) -> str:
        if self._session_id:
            return self._session_id
        client = await self._http()
        response = await client.post("/v1/session")
        response.raise_for_status()
        self._session_id = response.json()["session_id"]
        return self._session_id

    async def push_audio(self, chunk: bytes) -> list[TranscriptEvent]:
        session_id = await self._ensure_session()
        client = await self._http()
        response = await client.post(f"/v1/session/{session_id}/audio", content=chunk)
        response.raise_for_status()
        return [TranscriptEvent.model_validate(event) for event in response.json().get("events", [])]

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16000,
    ) -> AsyncIterator[TranscriptEvent]:
        from app.observability.tracing import get_tracer

        with get_tracer().span("stt", provider="whisper.cpp"):
            async for chunk in audio:
                for event in await self.push_audio(chunk):
                    yield event
            final = await self.finalize_turn()
            if final:
                yield final

    async def reset_session(self) -> None:
        if not self._session_id:
            return
        client = await self._http()
        await client.post(f"/v1/session/{self._session_id}/reset")

    async def finalize_turn(self) -> TranscriptEvent | None:
        if not self._session_id:
            return None
        client = await self._http()
        response = await client.post(f"/v1/session/{self._session_id}/finalize")
        response.raise_for_status()
        return TranscriptEvent.model_validate(response.json())

    async def aclose(self) -> None:
        client = self._client
        session_id = self._session_id
        self._session_id = None
        try:
            if client is not None and session_id is not None:
                response = await client.delete(f"/v1/session/{session_id}")
                response.raise_for_status()
        finally:
            if self._owns_client and client is not None:
                await client.aclose()
                self._client = None
