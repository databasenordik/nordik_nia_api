from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass

_ENDERS = frozenset(".!?")


@dataclass(frozen=True)
class SpokenPhrase:
    sequence_id: int
    generation_id: str
    text: str


class PhraseBuffer:
    """Emit short sentences, not word fragments."""

    def __init__(self, *, min_chars: int = 12) -> None:
        self._buf = ""
        self._seq = 0
        self._generation_id = ""
        self.min_chars = min_chars

    def start(self, generation_id: str) -> None:
        self._buf = ""
        self._seq = 0
        self._generation_id = generation_id

    def push(self, delta: str) -> list[SpokenPhrase]:
        if not delta:
            return []
        self._buf += delta
        return self._drain(flush=False)

    def flush(self) -> list[SpokenPhrase]:
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> list[SpokenPhrase]:
        emitted: list[SpokenPhrase] = []
        while self._buf:
            index = _sentence_end(self._buf)
            if index < 0:
                break
            chunk = self._buf[: index + 1].strip()
            rest = self._buf[index + 1 :].lstrip()
            if len(chunk) < self.min_chars and rest and not flush:
                break
            self._buf = rest
            if chunk:
                emitted.append(self._next(chunk))
        if flush and self._buf.strip():
            emitted.append(self._next(self._buf.strip()))
            self._buf = ""
        return emitted

    def _next(self, text: str) -> SpokenPhrase:
        self._seq += 1
        return SpokenPhrase(sequence_id=self._seq, generation_id=self._generation_id, text=text)


class PhraseQueue:
    """Bounded ordered phrase queue. Flush drops unsent items for the active generation."""

    def __init__(self, max_phrases: int = 8) -> None:
        self.max_phrases = max_phrases
        self.generation_id = ""
        self.tts_stream_id = ""
        self._items: deque[SpokenPhrase] = deque()
        self._wait = asyncio.Event()

    def start_generation(self) -> str:
        self.flush()
        self.generation_id = str(uuid.uuid4())
        self.tts_stream_id = str(uuid.uuid4())
        return self.generation_id

    def put(self, phrase: SpokenPhrase) -> bool:
        if phrase.generation_id != self.generation_id:
            return False
        if len(self._items) >= self.max_phrases:
            return False
        self._items.append(phrase)
        self._wait.set()
        return True

    def pop(self) -> SpokenPhrase | None:
        if not self._items:
            return None
        item = self._items.popleft()
        if not self._items:
            self._wait.clear()
        return item

    def flush(self) -> list[SpokenPhrase]:
        dropped = list(self._items)
        self._items.clear()
        self._wait.clear()
        return dropped

    def __len__(self) -> int:
        return len(self._items)

    def pending_texts(self) -> list[str]:
        return [item.text for item in self._items]


def split_phrases(text: str) -> list[str]:
    buffer = PhraseBuffer()
    buffer.start("split")
    phrases = buffer.push(text)
    phrases.extend(buffer.flush())
    return [item.text for item in phrases if item.text]


def _sentence_end(text: str) -> int:
    for index, char in enumerate(text):
        if char in _ENDERS:
            nxt = text[index + 1 : index + 2]
            if nxt == "" or nxt.isspace() or nxt in "\"')":
                return index
    return -1
