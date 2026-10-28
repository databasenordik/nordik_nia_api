from __future__ import annotations

import logging

from app.config import get_settings
from app.tts.base import TTSProvider

logger = logging.getLogger("nia.tts")


def get_tts_provider(name: str | None = None) -> TTSProvider:
    settings = get_settings()
    chosen = (name or settings.tts_provider).strip().lower()
    if chosen in {"off", "none", "disabled"}:
        from app.tts.fake import FakeTTSProvider

        return FakeTTSProvider()
    if chosen in {"fake", "test"}:
        from app.tts.fake import FakeTTSProvider

        return FakeTTSProvider()
    if chosen == "silent":
        from app.tts.silent import SilentTTSProvider

        return SilentTTSProvider(sample_rate=settings.xai_tts_sample_rate)
    if chosen == "kokoro":
        from app.tts.kokoro import KokoroTTSProvider

        return KokoroTTSProvider()
    if not settings.xai_api_key:
        # Voice must still connect and stream text without a paid TTS key.
        logger.warning(
            "TTS_PROVIDER=xai but XAI_API_KEY is empty; using silent fallback TTS. "
            "Set XAI_API_KEY for spoken audio, or TTS_PROVIDER=kokoro for local speech."
        )
        from app.tts.silent import SilentTTSProvider

        return SilentTTSProvider(sample_rate=settings.xai_tts_sample_rate)
    from app.tts.xai_tts import XAITTSProvider

    return XAITTSProvider()
