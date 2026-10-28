from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from app.concurrency.cancellation import CancellationToken
from app.tts.base import TTSProvider
from app.tts.humanize import humanize_for_speech
from app.tts.phrases import PhraseBuffer, PhraseQueue, SpokenPhrase, split_phrases

PhraseHandler = Callable[[SpokenPhrase], Awaitable[None]]


class SpeechSession:
    """Preconnected TTS session with ordered phrases and barge-in cancel."""

    def __init__(
        self,
        provider: TTSProvider,
        *,
        cancellation: CancellationToken | None = None,
        max_phrases: int = 8,
        mode: str = "voice",
    ) -> None:
        if mode == "text":
            raise RuntimeError("TTS must not run in text mode")
        self.provider = provider
        self.cancellation = cancellation or CancellationToken()
        self.queue = PhraseQueue(max_phrases=max_phrases)
        self.buffer = PhraseBuffer()
        self.interrupted = False
        self.delivered_text = ""
        self.generation_id = ""
        self.connected = False

    async def connect(self) -> None:
        await self.provider.connect_session()
        self.connected = True

    def begin_turn(self) -> str:
        self.interrupted = False
        self.delivered_text = ""
        self.generation_id = self.queue.start_generation()
        self.buffer.start(self.generation_id)
        return self.generation_id

    async def enqueue_text(self, text: str) -> list[SpokenPhrase]:
        spoken = humanize_for_speech(text)
        accepted: list[SpokenPhrase] = []
        for chunk in split_phrases(spoken):
            phrase = SpokenPhrase(
                sequence_id=len(self.queue) + len(accepted) + 1,
                generation_id=self.generation_id,
                text=chunk,
            )
            if self.queue.put(phrase):
                accepted.append(phrase)
        return accepted

    async def feed_delta(self, delta: str) -> list[SpokenPhrase]:
        accepted: list[SpokenPhrase] = []
        for phrase in self.buffer.push(delta):
            cleaned = SpokenPhrase(
                sequence_id=phrase.sequence_id,
                generation_id=phrase.generation_id,
                text=humanize_for_speech(phrase.text),
            )
            if cleaned.text and self.queue.put(cleaned):
                accepted.append(cleaned)
        return accepted

    async def finish_input(self) -> list[SpokenPhrase]:
        accepted: list[SpokenPhrase] = []
        for phrase in self.buffer.flush():
            cleaned = SpokenPhrase(
                sequence_id=phrase.sequence_id,
                generation_id=phrase.generation_id,
                text=humanize_for_speech(phrase.text),
            )
            if cleaned.text and self.queue.put(cleaned):
                accepted.append(cleaned)
        return accepted

    async def play(self) -> AsyncIterator[bytes]:
        if not self.connected:
            await self.connect()
        while True:
            if self.interrupted or self.cancellation.cancelled:
                return
            phrase = self.queue.pop()
            if phrase is None:
                return
            async for chunk in self.provider.synthesize_stream(phrase.text):
                if self.interrupted or self.cancellation.cancelled:
                    return
                if not self.delivered_text:
                    self.delivered_text = phrase.text
                elif not self.delivered_text.endswith(phrase.text):
                    self.delivered_text = f"{self.delivered_text} {phrase.text}".strip()
                yield chunk

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        self.begin_turn()
        await self.enqueue_text(text)
        async for chunk in self.play():
            yield chunk

    async def interrupt(self, reason: str = "barge-in") -> None:
        self.interrupted = True
        self.queue.flush()
        await self.provider.cancel_generation()
        self.cancellation.cancel(reason)

    async def close(self) -> None:
        await self.provider.close_session()
        self.connected = False


class BargeInController:
    """Stop playback/TTS and unused LLM/DAG work without wiping query state."""

    def __init__(self, speech: SpeechSession, cancellation: CancellationToken) -> None:
        self.speech = speech
        self.cancellation = cancellation
        self.interrupted = False
        self.query_state_kept = True

    async def interrupt(self, reason: str = "barge-in") -> None:
        self.interrupted = True
        await self.speech.interrupt(reason)
        if not self.cancellation.cancelled:
            self.cancellation.cancel(reason)
