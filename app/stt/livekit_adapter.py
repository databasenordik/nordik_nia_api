from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.stt.base import STTProvider, TranscriptEvent


class LiveKitWhisperAdapter:
    """Streaming adapter the LiveKit agent can use without coupling to whisper.cpp.

    Maps provider events to LiveKit SpeechEvent field names. LiveKit is imported
    only when create_livekit_stt() is called.
    """

    def __init__(self, provider: STTProvider) -> None:
        self.provider = provider

    def to_livekit_payload(self, event: TranscriptEvent) -> dict[str, Any]:
        return {
            "type": "interim_transcript" if event.type == "partial" else "final_transcript",
            "alternatives": [
                {
                    "text": event.text,
                    "language": "en",
                    "start_time": event.start_ms / 1000,
                    "end_time": event.end_ms / 1000,
                    "confidence": event.confidence,
                }
            ],
        }

    async def stream_events(self, audio: AsyncIterator[bytes], *, sample_rate: int = 16000) -> AsyncIterator[dict[str, Any]]:
        async for event in self.provider.transcribe_stream(audio, sample_rate=sample_rate):
            yield self.to_livekit_payload(event)


class LiveKitStreamPump:
    """Incremental PCM → LiveKit payloads. Used by RecognizeStream and unit tests."""

    def __init__(self, provider: STTProvider) -> None:
        self.adapter = LiveKitWhisperAdapter(provider)
        self.provider = provider

    async def push(self, pcm: bytes) -> list[dict[str, Any]]:
        pusher = getattr(self.provider, "push_audio", None)
        if pusher is None:
            return []
        events = await pusher(pcm)
        return [self.adapter.to_livekit_payload(event) for event in events]

    async def flush(self) -> list[dict[str, Any]]:
        final = await self.provider.finalize_turn()
        payloads = [self.adapter.to_livekit_payload(final)] if final else []
        await self.provider.reset_session()
        return payloads


def create_livekit_stt(provider: STTProvider):
    """Create a VAD-adaptable LiveKit STT implementation.

    The HTTP Whisper service returns its authoritative transcript only when a
    turn is finalized. Advertising this implementation as a native streaming
    STT causes a deadlock: LiveKit expects a streaming provider to emit its own
    speech boundaries, while Whisper waits for LiveKit to flush the stream.

    Declaring the implementation as batch STT lets LiveKit automatically wrap
    it in ``stt.StreamAdapter`` using the AgentSession VAD. Each detected speech
    segment is then sent to Whisper and finalized without phrase-specific logic.
    """
    from livekit.agents import stt
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

    pump = LiveKitStreamPump(provider)

    def _speech_event(payload: dict[str, Any]):
        kind = payload.get("type")
        if kind == "start_of_speech":
            return stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
        if kind == "end_of_speech":
            return stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
        alt = payload["alternatives"][0]
        event_type = (
            stt.SpeechEventType.INTERIM_TRANSCRIPT
            if kind == "interim_transcript"
            else stt.SpeechEventType.FINAL_TRANSCRIPT
        )
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[
                stt.SpeechData(
                    language=alt.get("language") or "en",
                    text=alt["text"],
                    start_time=float(alt.get("start_time") or 0),
                    end_time=float(alt.get("end_time") or 0),
                    confidence=float(alt.get("confidence") or 0),
                )
            ],
        )

    class WhisperCppLiveKitSTT(stt.STT):
        def __init__(self) -> None:
            super().__init__(
                capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
            )

        @property
        def model(self) -> str:
            return "whisper.cpp"

        @property
        def provider(self) -> str:
            return "local"

        async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
            frames = buffer if isinstance(buffer, list) else [buffer]
            pcm = b"".join(bytes(frame.data) for frame in frames)
            payloads = await pump.push(pcm)
            payloads.extend(await pump.flush())
            final = next(
                (item for item in reversed(payloads) if item["type"] == "final_transcript"),
                payloads[-1] if payloads else None,
            )
            if final is None:
                return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])
            return _speech_event(final)

    return WhisperCppLiveKitSTT()
