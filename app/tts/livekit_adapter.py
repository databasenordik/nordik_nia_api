from __future__ import annotations

import re

from app.tts.base import TTSProvider
from app.tts.expressive import add_contextual_voice_tags, strip_voice_tags

SAMPLE_RATE = 24000
_SPOKEN_PRONUNCIATIONS = ((re.compile(r"\bNIA\b"), "Nia"),)


def prepare_speech_text(text: str, *, expressive: bool = False) -> str:
    """Build the TTS-only copy without altering chat or conversation text."""
    spoken = strip_voice_tags(text)
    for pattern, replacement in _SPOKEN_PRONUNCIATIONS:
        spoken = pattern.sub(replacement, spoken)
    return add_contextual_voice_tags(spoken) if expressive else spoken


def create_livekit_tts(provider: TTSProvider):
    """LiveKit TTS wrapper. LiveKit is imported only at agent runtime."""
    from livekit.agents import tts
    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

    class ProviderChunkedStream(tts.ChunkedStream):
        async def _run(self, output_emitter) -> None:
            output_emitter.initialize(
                request_id="nia-tts",
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )
            speech_text = prepare_speech_text(
                self._input_text,
                expressive=bool(getattr(provider, "supports_expressive_tags", False)),
            )
            async for chunk in provider.synthesize_stream(speech_text):
                output_emitter.push(chunk)
            output_emitter.flush()

    class ProviderLiveKitTTS(tts.TTS):
        def __init__(self) -> None:
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

        @property
        def model(self) -> str:
            return "nia-tts"

        @property
        def provider(self) -> str:
            return "local-adapter"

        def synthesize(
            self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
        ) -> tts.ChunkedStream:
            return ProviderChunkedStream(tts=self, input_text=text, conn_options=conn_options)

        def prewarm(self) -> None:
            return None

    return ProviderLiveKitTTS()
