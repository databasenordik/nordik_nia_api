from __future__ import annotations

import re

# xAI's documented TTS tags. Keeping the complete allow-list here lets us
# remove model-supplied markup before deciding which restrained effects Nia
# should actually use.
INLINE_TAGS = (
    "pause",
    "long-pause",
    "hum-tune",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "breath",
    "inhale",
    "exhale",
    "sigh",
)
WRAPPING_TAGS = (
    "soft",
    "whisper",
    "loud",
    "build-intensity",
    "decrease-intensity",
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",
    "singing",
    "laugh-speak",
    "emphasis",
)

_INLINE = re.compile(r"\[(?:" + "|".join(map(re.escape, INLINE_TAGS)) + r")\]", re.I)
_WRAPPING = re.compile(
    r"</?(?:" + "|".join(map(re.escape, WRAPPING_TAGS)) + r")>", re.I
)
_SPACES = re.compile(r"[ \t]{2,}")
_COUNT = re.compile(
    r"(?i)\b(?:there (?:are|is)|the total is|I found|showing)\s+"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
)
_SENSITIVE = re.compile(
    r"(?i)\b(?:death|deaths|deceased|died|cause of death|grief|trauma|abuse|"
    r"not authorized|access is restricted|cannot access|can't access|"
    r"may have misheard|want to be sure I understood|could you say that another way)\b"
)
_CAUTION = re.compile(r"(?i)\b(?:important|please note|warning|be careful)\b")


def strip_voice_tags(text: str) -> str:
    """Remove supported speech markup from text intended for display/storage."""
    cleaned = _INLINE.sub(" ", text)
    cleaned = _WRAPPING.sub("", cleaned)
    cleaned = _SPACES.sub(" ", cleaned)
    return cleaned.replace(" ,", ",").replace(" .", ".").strip()


def add_contextual_voice_tags(text: str) -> str:
    """Add restrained xAI expression only when the phrase provides context.

    Deliberately excluded: laughter, giggles, sing-song, upbeat pitch changes,
    and indiscriminate emotion. Ordinary replies remain ordinary speech.
    """
    spoken = strip_voice_tags(text)
    if not spoken:
        return ""

    # Sensitive records, access boundaries, and repair prompts benefit from a
    # gentler delivery. Phrases arrive sentence-sized, so wrapping the complete
    # phrase avoids abrupt mid-sentence style changes.
    if _SENSITIVE.search(spoken):
        return f"<soft>{spoken}</soft>"

    # Warnings should be deliberate, never louder or more dramatic.
    if _CAUTION.search(spoken):
        return f"<slow>{spoken}</slow>"

    # Emphasize an answer's computed count, but not arbitrary dates or IDs.
    count = _COUNT.search(spoken)
    if count:
        start, end = count.span("number")
        return f"{spoken[:start]}<emphasis>{spoken[start:end]}</emphasis>{spoken[end:]}"

    # A colon introducing a substantial list is a natural place to breathe.
    if ": " in spoken and len(spoken) >= 80:
        return spoken.replace(": ", ": [pause] ", 1)

    return spoken

