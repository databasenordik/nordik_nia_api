from __future__ import annotations

import re
from typing import Literal

VoicePreference = Literal["male", "female"]

_ACTION_REQUEST = re.compile(
    r"(?is)\b(?:switch|change|set|use|select|choose|prefer|speak|talk|sound|make)\b"
    r".{0,36}?\b(?P<gender>female|woman(?:['’]s)?|woman|lady|male|man(?:['’]s)?|man)\b"
    r"(?:\s+voice)?"
)
_DIRECT_REQUEST = re.compile(
    r"(?is)^\s*(?:please\s+)?(?:use\s+)?(?:a\s+)?"
    r"(?P<gender>female|woman(?:['’]s)?|woman|lady|male|man(?:['’]s)?|man)"
    r"(?:\s+voice)?(?:\s+please)?[.!?]*\s*$"
)


def requested_voice_preference(text: str) -> VoicePreference | None:
    """Recognize an instruction to change voice without matching discussion of voices."""
    match = _ACTION_REQUEST.search(text) or _DIRECT_REQUEST.fullmatch(text)
    if match is None:
        return None
    gender = match.group("gender").lower()
    return "female" if gender.startswith(("f", "w", "l")) else "male"
