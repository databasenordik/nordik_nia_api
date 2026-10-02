from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SAFE_FILLERS = re.compile(r"\b(?:uh+|um+|erm+|hmm+|mm+|ah+)\b", re.IGNORECASE)
_POLITE_PREFIX = re.compile(
    r"^\s*(?:(?:please\s+)?(?:could|can|would|will)\s+you\s+(?:please\s+)?(?:tell|show|give)\s+me\s+|"
    r"(?:please\s+)?(?:could|can|would|will)\s+you\s+|"
    r"please\s+|i\s+(?:would|'d)\s+like\s+you\s+to\s+|i\s+want\s+you\s+to\s+)",
    re.IGNORECASE,
)
_SIMPLE_CORRECTION = re.compile(
    r"\b(?P<old>[A-Za-z]|(?:1[6-9]\d{2}|20\d{2}))\b\s*[,;:\-–—]*\s*"
    r"(?P<marker>sorry|no\s+wait|actually\s*,?\s*i\s+mean|"
    r"actually(?!\s*,?\s*i\s+mean\b)|i\s+mean)\s*[,;:\-–—]*\s*"
    r"\b(?P<new>[A-Za-z]|(?:1[6-9]\d{2}|20\d{2}))\b",
    re.IGNORECASE,
)
_CORRECTION_MARKER = re.compile(
    r"\b(?:sorry|no\s+wait|wait|actually|i\s+mean|rather|instead)\b",
    re.IGNORECASE,
)
_REPEAT_PUNCT = re.compile(r"([!?.,])\1+")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationCorrection:
    old: str
    new: str
    marker: str


@dataclass(frozen=True)
class NormalizedTurn:
    """Conservative cleanup for typed text or a FINAL STT transcript.

    This layer never paraphrases the request. It normalizes Unicode/spacing,
    removes speech fillers that carry no query meaning, strips a small class of
    polite wrappers, and resolves only one-token self-corrections such as
    ``S, sorry, R`` or ``1900, actually 1920``. Semantic correction markers that
    remain visible cause the deterministic fast path to abstain.
    """

    raw_text: str
    normalized_text: str
    corrections: tuple[NormalizationCorrection, ...] = ()
    removed_fillers: tuple[str, ...] = ()
    unresolved_correction: bool = False

    @property
    def changed(self) -> bool:
        return self.raw_text.strip() != self.normalized_text


def normalize_transport(text: str) -> NormalizedTurn:
    """Unicode/spacing only. AI mode must not strip fillers or rewrite corrections."""
    raw = text or ""
    value = unicodedata.normalize("NFKC", raw)
    value = (
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )
    value = _REPEAT_PUNCT.sub(r"\1", value)
    value = _WS.sub(" ", value).strip()
    return NormalizedTurn(raw_text=raw, normalized_text=value)


def normalize_user_turn(text: str) -> NormalizedTurn:
    raw = text or ""
    value = unicodedata.normalize("NFKC", raw)
    value = (
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )

    corrections: list[NormalizationCorrection] = []

    def _apply_simple_correction(match: re.Match[str]) -> str:
        corrections.append(
            NormalizationCorrection(
                old=match.group("old"),
                new=match.group("new"),
                marker=match.group("marker"),
            )
        )
        return match.group("new")

    value = _SIMPLE_CORRECTION.sub(_apply_simple_correction, value)

    removed_fillers = tuple(match.group(0) for match in _SAFE_FILLERS.finditer(value))
    value = _SAFE_FILLERS.sub(" ", value)
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r"^[\s,;:]+", "", value)
    value = _POLITE_PREFIX.sub("", value)
    value = re.sub(r"^[\s,;:]+", "", value)
    value = _REPEAT_PUNCT.sub(r"\1", value)
    value = re.sub(r"\s*[–—]+\s*", " ", value)
    value = _WS.sub(" ", value).strip()

    return NormalizedTurn(
        raw_text=raw,
        normalized_text=value,
        corrections=tuple(corrections),
        removed_fillers=removed_fillers,
        unresolved_correction=bool(_CORRECTION_MARKER.search(value)),
    )
