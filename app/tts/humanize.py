from __future__ import annotations

import re

from app.tts.expressive import strip_voice_tags

_FILLER_LEAD = re.compile(
    r"^\s*(certainly|of course|great question|sure|absolutely|right away)[!.,:—-]*\s+",
    re.IGNORECASE,
)
_SOURCE_IDS = re.compile(
    r"\[(?:file|src):\d+:[^\]]+\]|"
    r"\bfile:\d+:\S+|"
    r"\(\s*file:\d+:[^\)]+\)|"
    r"\[file\s+\d+[^\]]*\]",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s{2,}")


# Speech is real-time: every extra word is extra seconds the user waits. Roughly
# 15 characters per second of speech, so ~320 chars is about 20s worst case and
# most answers land far below it.
MAX_SPOKEN_CHARS = 320
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def humanize_for_speech(text: str, *, max_chars: int | None = MAX_SPOKEN_CHARS) -> str:
    """Spoken form: no filler openers, no raw citation IDs, no evidence payloads.

    Also bounded in length — a spoken answer costs the listener real time, so
    long prose is trimmed at a sentence boundary rather than read in full.
    """
    spoken = strip_voice_tags(text)
    spoken = _SOURCE_IDS.sub("", spoken)
    spoken = _FILLER_LEAD.sub("", spoken)
    spoken = _SPACES.sub(" ", spoken).strip()
    spoken = spoken.replace(" ,", ",").replace(" .", ".")
    if max_chars is not None:
        spoken = _trim_to_sentences(spoken, max_chars)
    return spoken


def _trim_to_sentences(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    kept: list[str] = []
    used = 0
    for sentence in _SENTENCE_SPLIT.split(text):
        if kept and used + len(sentence) + 1 > max_chars:
            break
        kept.append(sentence)
        used += len(sentence) + 1
    if not kept:
        return text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return " ".join(kept).strip()
