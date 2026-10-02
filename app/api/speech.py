from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope
from app.tts.base import TTSProvider
from app.tts.factory import get_tts_provider
from app.tts.livekit_adapter import prepare_speech_text

router = APIRouter(prefix="/api/voice", tags=["voice"])


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    preference: Literal["male", "female"] = "male"


def _tts_provider() -> TTSProvider:
    return get_tts_provider()


@router.post("/speech")
async def synthesize_speech(
    body: SpeechRequest,
    _scope: AccessScope = Depends(get_access_scope),  # noqa: B008
    provider: TTSProvider = Depends(_tts_provider),  # noqa: B008
) -> StreamingResponse:
    """Stream authenticated, display-safe speech as 24 kHz mono PCM.

    The API key and expression markup remain server-side. Callers only send
    clean visible text and their session's male/female preference.
    """
    settings = get_settings()
    if settings.tts_provider.strip().lower() == "kokoro":
        voice = (
            settings.kokoro_female_voice
            if body.preference == "female"
            else settings.kokoro_male_voice
        )
    else:
        voice = (
            settings.xai_tts_female_voice
            if body.preference == "female"
            else settings.xai_tts_male_voice
        )
    speech_text = prepare_speech_text(
        body.text,
        expressive=bool(getattr(provider, "supports_expressive_tags", False)),
    )

    async def audio() -> AsyncIterator[bytes]:
        try:
            async for chunk in provider.synthesize_stream(
                speech_text,
                voice_config={"voice": voice},
            ):
                yield chunk
        finally:
            await provider.close_session()

    return StreamingResponse(
        audio(),
        media_type="audio/L16",
        headers={
            "Cache-Control": "no-store",
            "X-Audio-Sample-Rate": str(settings.xai_tts_sample_rate),
            "X-NIA-Voice": voice,
        },
    )
