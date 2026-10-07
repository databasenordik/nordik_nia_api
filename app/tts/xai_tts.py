from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

from app.config import get_settings

OpenSocket = Callable[[str, dict[str, str], float], Awaitable[Any]]


class XAITTSProvider:
    """Streaming xAI TTS over WebSocket. Retrieval never sees this client."""

    supports_expressive_tags = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
        open_socket: OpenSocket | None = None,
        first_audio_timeout_ms: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.xai_api_key
        self._voice = voice or settings.xai_tts_voice
        self._language = language or settings.xai_tts_language
        self._format = audio_format or settings.xai_tts_format
        self._sample_rate = sample_rate or getattr(settings, "xai_tts_sample_rate", 24000)
        self._url = url or settings.xai_tts_url
        self._open_socket = open_socket or _open_websockets
        self._first_audio_timeout_s = (
            (first_audio_timeout_ms or getattr(settings, "xai_tts_first_audio_timeout_ms", 4000))
            / 1000
        )
        self._ws: Any = None
        self._cancel = asyncio.Event()
        self.connected = False
        self.first_audio_ms: float | None = None

    @property
    def voice(self) -> str:
        return self._voice

    async def set_voice(self, voice: str) -> None:
        """Change voice for this session and reconnect before the next phrase."""
        selected = voice.strip().lower()
        if not selected or selected == self._voice.lower():
            return
        await self.close_session()
        self._voice = selected

    async def connect_session(self) -> None:
        if self._ws is not None and self.connected:
            return
        if not self._api_key and self._open_socket is _open_websockets:
            raise RuntimeError("XAI_API_KEY is not configured")
        params = {
            "voice": self._voice,
            "language": self._language,
            "codec": self._format or "pcm",
            "sample_rate": self._sample_rate,
        }
        url = f"{self._url}?{urlencode(params)}"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._ws = await self._open_socket(url, headers, self._first_audio_timeout_s)
        self.connected = True

    async def synthesize_stream(
        self, text: str, voice_config: dict | None = None
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        requested_voice = str((voice_config or {}).get("voice") or "").strip()
        if requested_voice:
            await self.set_voice(requested_voice)
        from app.observability.tracing import get_tracer

        with get_tracer().span("tts", provider="xai"):
            async for chunk in self._synthesize_inner(text):
                yield chunk

    async def _synthesize_inner(self, text: str) -> AsyncIterator[bytes]:
        self._cancel.clear()
        self.first_audio_ms = None
        if not self.connected or self._ws is None:
            await self.connect_session()
        started = time.perf_counter()
        await self._send({"type": "text.delta", "delta": text})
        await self._send({"type": "text.done"})
        while not self._cancel.is_set():
            raw = await self._recv()
            if raw is None:
                break
            payload = json.loads(raw) if isinstance(raw, str | bytes | bytearray) else raw
            kind = payload.get("type")
            if kind == "audio.delta":
                if self.first_audio_ms is None:
                    self.first_audio_ms = (time.perf_counter() - started) * 1000
                yield base64.b64decode(payload["delta"])
            elif kind == "audio.done":
                break
            elif kind == "error":
                raise RuntimeError(payload.get("message") or "xAI TTS error")

    async def cancel_generation(self) -> None:
        self._cancel.set()
        if self._ws is not None:
            closer = getattr(self._ws, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    pass
            self._ws = None
            self.connected = False

    async def close_session(self) -> None:
        self._cancel.set()
        if self._ws is not None:
            closer = getattr(self._ws, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    pass
        self._ws = None
        self.connected = False

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._ws is not None
        message = json.dumps(payload)
        sender = self._ws.send
        await sender(message)

    async def _recv(self) -> str | bytes | None:
        assert self._ws is not None
        try:
            return await asyncio.wait_for(self._ws.recv(), timeout=self._first_audio_timeout_s)
        except TimeoutError:
            return None


async def _open_websockets(url: str, headers: dict[str, str], timeout: float):
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is not installed") from exc
    try:
        return await websockets.connect(url, additional_headers=headers, open_timeout=timeout)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, open_timeout=timeout)
