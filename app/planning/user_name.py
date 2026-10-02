"""Caller name capture. Not a research-record field."""

from __future__ import annotations

import re

_INTRO = re.compile(
    r"^(?:(?:hi|hey|hello|hiya)[,!\s]+)?(?:i am|i'm|im|my name is|this is|call me)\s+"
    r"(?P<name>[A-Za-z][A-Za-z.'-]{0,40}(?:\s+[A-Za-z][A-Za-z.'-]{0,40}){0,2})"
    r"\s*[.!]*$",
    re.IGNORECASE,
)
_STOP = frozenset(
    {
        "a",
        "about",
        "ambiguous",
        "an",
        "and",
        "asking",
        "confused",
        "fine",
        "from",
        "going",
        "here",
        "just",
        "looking",
        "lost",
        "not",
        "only",
        "really",
        "so",
        "sorry",
        "sure",
        "the",
        "trying",
        "unclear",
        "very",
        "wondering",
    }
)
_MY_NAME = re.compile(
    r"^(?:what(?:'s|s| is)? my name|do you know my name|who am i|remind me my name)"
    r"(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)
_ASSISTANT_IDENTITY = re.compile(
    r"^(?:who are (?:you|u)|who're you|what(?:'s|s| is)? your name|"
    r"what do (?:people call you|you call yourself)|ur name)[.!?]*$",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"^(?:i (?:did not|didn't|didnt) ask(?: (?:you )?(?:your name|that|for that|for .{1,60}))?|"
    r"i asked (?:about |for )?my name|"
    r"not your name|"
    r"you should(?: know(?: that| my name)?)?|"
    r"that(?:'s| is) not what i asked|"
    r"wrong(?: answer)?|"
    r"no(?:pe)?(?:[,.]?\s+(?:that(?:'s| is) not(?: it| what i asked)?))?|"
    r"i (?:meant|said) my name|"
    r"it should be .{1,100})\s*[.!?]*$",
    re.IGNORECASE,
)
_RESEARCH = re.compile(
    r"\b(?:list|count|how many|students?|master list|dataset|"
    r"authorized records|research records|"
    r"deceased|admitted|discharged|community|next|show more|keep going|"
    r"confirmed deaths?|quote|compare|aggregate|starts? with|beginning with|"
    r"what about|how about|file \d+|enumerate|"
    r"the [a-z] names|names starting|name starts|names with|what(?:'s|s| is) the total|"
    r"the records|causes? of death|more than that|admission|earliest|latest|repeat|provenance)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b")


def extract_introduced_name(text: str) -> str | None:
    match = _INTRO.match(" ".join((text or "").strip().split()))
    if not match:
        return None
    parts = match.group("name").split()
    if any(part.lower() in _STOP for part in parts):
        return None
    name = " ".join(part.capitalize() if part.islower() else part for part in parts)
    return name


def is_user_name_question(text: str) -> bool:
    return bool(_MY_NAME.fullmatch(" ".join((text or "").strip().split())))


def is_assistant_identity_question(text: str) -> bool:
    return bool(_ASSISTANT_IDENTITY.fullmatch(" ".join((text or "").strip().split())))


def is_conversational_correction(text: str) -> bool:
    return bool(_CORRECTION.fullmatch(" ".join((text or "").strip().split())))


_SPEAK = re.compile(
    r"^(?:can you (?:speak|talk)(?: to me)?|do you speak|speak(?: to me)?|"
    r"talk(?: to me)?|say (?:it |that )?out loud)\s*[.!?]*$",
    re.IGNORECASE,
)
_AFFIRM = re.compile(r"^(?:yes|yeah|yep|yup|ok|okay|sure|please do)\s*[.!?]*$", re.IGNORECASE)
_CONFUSED = re.compile(
    r"^(?:what|what(?:'s|s| is) that|what do you mean|huh|excuse me)\s*[.!?]*$",
    re.IGNORECASE,
)
_EXPLICIT_ROW = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b"
    r".{0,24}\b(?:one|ones|student|record|result|them)\b"
    r"|\btell me about (?:the )?(?:first|second|third|fourth|fifth|last)\b",
    re.IGNORECASE,
)


def is_speak_request(text: str) -> bool:
    return bool(_SPEAK.fullmatch(" ".join((text or "").strip().split())))


def is_bare_affirmation(text: str) -> bool:
    return bool(_AFFIRM.fullmatch(" ".join((text or "").strip().split())))


def is_confusion(text: str) -> bool:
    return bool(_CONFUSED.fullmatch(" ".join((text or "").strip().split())))


def is_explicit_row_reference(text: str) -> bool:
    return bool(_EXPLICIT_ROW.search(" ".join((text or "").strip().split())))


_NAMED_FOLLOWUP = re.compile(
    r"^(?:tell me (?:more )?about|who(?:'s| is)|details (?:on|about)|what happened to)\s+"
    r"(?!the\b)(?P<name>[A-Za-z][A-Za-z.'-]{1,40}(?:\s+[A-Za-z][A-Za-z.'-]{1,40}){0,2})"
    r"\s*[.!?]*$",
    re.IGNORECASE,
)


def extract_followup_name(text: str) -> str | None:
    match = _NAMED_FOLLOWUP.fullmatch(" ".join((text or "").strip().split()))
    if not match:
        return None
    name = match.group("name").strip().rstrip(".!?").strip()
    if name.lower() in _STOP:
        return None
    return name


def is_named_record_followup(text: str) -> bool:
    return extract_followup_name(text) is not None


def looks_like_research(text: str) -> bool:
    lowered = " ".join((text or "").strip().split())
    if not lowered:
        return False
    if _RESEARCH.search(lowered) or _YEAR.search(lowered):
        return True
    return bool(re.fullmatch(r"(?:only\s+)?[A-Za-z][.!?]?", lowered))
